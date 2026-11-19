//! Reads per-kernel metrics from an ncu report exported as csv
//! and joins them to the kernels table by launch order:
//!
//! ```text
//! ncu --import r.ncu-rep --csv --page raw --print-units base --print-kernel-base function
//! ```
//!
//! `function` makes ncu print the same short kernel names as nsys.
//!
//! ncu replays every kernel, so its durations are not real. Times always
//! come from nsys; this table only carries counts (bytes, instructions).

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use arrow_array::builder::{StringDictionaryBuilder, UInt32Builder, UInt64Builder};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, UInt32Type};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_ipc::reader::FileReader;
use arrow_schema::{DataType, Field, Schema};

use crate::trace::SCHEMA_VERSION;

/// The raw page also carries device attributes and every rollup
/// (.avg/.min/.max) of each metric. Only the per-kernel totals of the
/// requested metrics are kept, and those are whole counts.
fn is_kept_metric(column: &str) -> bool {
    column.ends_with(".sum") && !column.starts_with("profiler__")
}

pub struct NcuTable {
    pub names: Vec<String>,
    pub metrics: Vec<String>,
    /// one row per profiled kernel, one value per metric
    pub values: Vec<Vec<u64>>,
}

pub fn read_csv(path: &Path) -> Result<NcuTable> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(false)
        .from_path(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let header = reader.headers()?.clone();
    let col = |name: &str| header.iter().position(|h| h == name);
    let name_col = col("Kernel Name").context("csv has no 'Kernel Name' column")?;
    let metric_cols: Vec<usize> = (0..header.len())
        .filter(|i| is_kept_metric(&header[*i]))
        .collect();
    ensure!(!metric_cols.is_empty(), "csv has no .sum metric columns");

    let mut table = NcuTable {
        names: Vec::new(),
        metrics: metric_cols.iter().map(|i| header[*i].to_string()).collect(),
        values: Vec::new(),
    };
    for (line, record) in reader.records().enumerate() {
        let record = record?;
        // the raw page repeats a units row under the header
        if line == 0 && record.get(0).is_some_and(str::is_empty) {
            for (i, m) in metric_cols.iter().zip(&table.metrics) {
                let unit = &record[*i];
                ensure!(
                    !m.contains("bytes") || unit == "byte",
                    "{m} is in '{unit}', export with --print-units base"
                );
            }
            continue;
        }
        let mut row = Vec::with_capacity(metric_cols.len());
        for (i, m) in metric_cols.iter().zip(&table.metrics) {
            let text = record[*i].replace(',', "");
            let value: f64 = text
                .parse()
                .with_context(|| format!("row {line}: {m} = '{}'", &record[*i]))?;
            ensure!(
                value >= 0.0 && value.fract() == 0.0,
                "row {line}: {m} = {value} is not a count"
            );
            row.push(value as u64);
        }
        table.names.push(record[name_col].to_string());
        table.values.push(row);
    }
    Ok(table)
}

/// kernels.arrow's name and step columns, in seq order.
fn read_kernel_names(path: &Path) -> Result<(Vec<String>, Vec<u32>)> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let (mut names, mut steps) = (Vec::new(), Vec::new());
    for batch in FileReader::try_new(file, None)? {
        let batch = batch?;
        let name = batch
            .column_by_name("name")
            .context("kernels.arrow has no name column")?
            .as_dictionary::<Int32Type>();
        let values = name.values().as_string::<i32>();
        for key in name.keys().iter() {
            names.push(
                values
                    .value(key.context("null kernel name")? as usize)
                    .to_string(),
            );
        }
        let step = batch
            .column_by_name("step")
            .context("kernels.arrow has no step column")?
            .as_primitive::<UInt32Type>();
        steps.extend(step.values().iter().copied());
    }
    Ok((names, steps))
}

pub struct Joined {
    pub batch: RecordBatch,
    /// (step, kernels, summed values per metric) for every step in the window
    pub per_step: Vec<(u32, usize, Vec<u64>)>,
}

/// Lines ncu row i up with kernels row `skip + i` and refuses any mismatch.
pub fn join(table: &NcuTable, kernels: &Path, skip: usize) -> Result<Joined> {
    let (names, steps) = read_kernel_names(kernels)?;
    ensure!(
        skip + table.names.len() <= names.len(),
        "window ends at kernel {} but {} has {} kernels",
        skip + table.names.len(),
        kernels.display(),
        names.len()
    );
    for (i, ncu_name) in table.names.iter().enumerate() {
        let nsys_name = &names[skip + i];
        if ncu_name != nsys_name {
            bail!(
                "ncu kernel {i} is {ncu_name} but nsys kernel {} is {nsys_name}; \
                 the launch skip or the run does not match",
                skip + i
            );
        }
    }

    let n = table.names.len();
    let mut seq = UInt32Builder::with_capacity(n);
    let mut step = UInt32Builder::with_capacity(n);
    let mut name = StringDictionaryBuilder::<Int32Type>::new();
    for (i, k) in table.names.iter().enumerate() {
        seq.append_value((skip + i) as u32);
        step.append_value(steps[skip + i]);
        name.append_value(k);
    }
    let mut fields = vec![
        Field::new("seq", DataType::UInt32, false),
        Field::new("step", DataType::UInt32, false),
        Field::new(
            "name",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            false,
        ),
    ];
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(seq.finish()),
        Arc::new(step.finish()),
        Arc::new(name.finish()),
    ];
    for (m, metric) in table.metrics.iter().enumerate() {
        let mut b = UInt64Builder::with_capacity(n);
        table.values.iter().for_each(|row| b.append_value(row[m]));
        fields.push(Field::new(metric, DataType::UInt64, false));
        columns.push(Arc::new(b.finish()));
    }
    let schema = Schema::new(fields).with_metadata(
        [
            ("hotaru.table", "kernel_metrics"),
            ("hotaru.schema_version", SCHEMA_VERSION),
            ("hotaru.source", "ncu"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect::<std::collections::HashMap<_, _>>(),
    );
    let batch = RecordBatch::try_new(Arc::new(schema), columns)?;

    let mut per_step: Vec<(u32, usize, Vec<u64>)> = Vec::new();
    for (i, row) in table.values.iter().enumerate() {
        let s = steps[skip + i];
        if per_step.last().is_none_or(|(last, _, _)| *last != s) {
            per_step.push((s, 0, vec![0; table.metrics.len()]));
        }
        let (_, count, sums) = per_step.last_mut().unwrap();
        *count += 1;
        sums.iter_mut().zip(row).for_each(|(a, v)| *a += v);
    }
    debug_assert_eq!(batch.column(0).len(), n);
    Ok(Joined { batch, per_step })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{KernelRow, kernels_batch, write_batch};

    const CSV: &str = "\
\"ID\",\"Process ID\",\"Process Name\",\"Host Name\",\"Kernel Name\",\"Context\",\"Stream\",\"Block Size\",\"Grid Size\",\"Device\",\"CC\",\"device__attribute_display_name\",\"dram__bytes_op_read.avg\",\"dram__bytes_op_read.sum\",\"lts__t_bytes.sum\"
\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"\",\"byte\",\"byte\",\"byte\"
\"0\",\"42\",\"llama-completion.exe\",\"127.0.0.1\",\"quantize_q8_1\",\"1\",\"7\",\"(256, 1, 1)\",\"(16, 1, 1)\",\"0\",\"12.0\",\"NVIDIA GeForce RTX 5060 Laptop GPU\",\"4,128.25\",\"16,512\",\"40,960\"
\"1\",\"42\",\"llama-completion.exe\",\"127.0.0.1\",\"mul_mat_vec_q\",\"1\",\"7\",\"(32, 4, 1)\",\"(4096, 1, 1)\",\"0\",\"12.0\",\"NVIDIA GeForce RTX 5060 Laptop GPU\",\"2,359,296\",\"94,37,184\",\"9,500,000\"
";

    fn kernel(name: &str, step: u32) -> KernelRow {
        KernelRow {
            start_ns: 0,
            end_ns: 1,
            device: 0,
            stream: 7,
            correlation_id: 1,
            name: name.into(),
            demangled: name.into(),
            grid: [1, 1, 1],
            block: [1, 1, 1],
            registers: 0,
            shared_static: 0,
            shared_dynamic: 0,
            graph_node_id: None,
            step,
        }
    }

    fn setup(names: &[(&str, u32)]) -> (tempfile::TempDir, NcuTable, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("window.csv");
        std::fs::write(&csv_path, CSV).unwrap();
        let rows: Vec<_> = names.iter().map(|(n, s)| kernel(n, *s)).collect();
        let kpath = dir.path().join("kernels.arrow");
        write_batch(&kpath, &kernels_batch(&rows, "test").unwrap()).unwrap();
        (dir, read_csv(&csv_path).unwrap(), kpath)
    }

    #[test]
    fn keeps_only_metric_totals_and_parses_any_digit_grouping() {
        let (_dir, table, _) = setup(&[]);
        assert_eq!(table.names, ["quantize_q8_1", "mul_mat_vec_q"]);
        assert_eq!(
            table.metrics,
            ["dram__bytes_op_read.sum", "lts__t_bytes.sum"]
        );
        assert_eq!(table.values[1], [9_437_184, 9_500_000]);
    }

    #[test]
    fn joins_by_launch_order_and_sums_per_step() {
        let (_dir, table, kpath) = setup(&[
            ("rms_norm_f32", 4),
            ("quantize_q8_1", 4),
            ("mul_mat_vec_q", 5),
        ]);
        let joined = join(&table, &kpath, 1).unwrap();
        let seq = joined.batch.column(0).as_primitive::<UInt32Type>();
        assert_eq!(seq.values().as_ref(), &[1, 2]);
        assert_eq!(joined.per_step.len(), 2);
        assert_eq!(joined.per_step[0], (4, 1, vec![16_512, 40_960]));
        assert_eq!(joined.per_step[1], (5, 1, vec![9_437_184, 9_500_000]));
    }

    #[test]
    fn refuses_a_window_that_does_not_line_up() {
        let (_dir, table, kpath) = setup(&[
            ("rms_norm_f32", 4),
            ("quantize_q8_1", 4),
            ("mul_mat_vec_q", 5),
        ]);
        let err = join(&table, &kpath, 0).err().unwrap().to_string();
        assert!(err.contains("ncu kernel 0 is quantize_q8_1"), "{err}");
    }
}
