//! The hotaru trace format: one Arrow IPC file per table.
//!
//! Every engine that hotaru reads (llama.cpp through nsys today, issen
//! later) writes these same schemas, so the renderer never sees the source.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::builder::{Int64Builder, StringDictionaryBuilder, UInt32Builder};
use arrow_array::types::Int32Type;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema};

pub const SCHEMA_VERSION: &str = "1";

/// One kernel launch as it ran on the gpu. Times are nanoseconds on the
/// profiler's clock.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelRow {
    pub start_ns: i64,
    pub end_ns: i64,
    pub device: u32,
    pub stream: u32,
    pub correlation_id: u32,
    pub name: String,
    pub demangled: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub registers: u32,
    pub shared_static: u32,
    pub shared_dynamic: u32,
    pub graph_node_id: Option<i64>,
    /// Which engine step the kernel belongs to. For llama.cpp traces this is
    /// 0 for warmup, 1 for the prompt, then one step per generated token.
    pub step: u32,
}

impl KernelRow {
    pub fn duration_ns(&self) -> i64 {
        self.end_ns - self.start_ns
    }
}

fn dict() -> DataType {
    DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
}

pub fn kernels_schema(source: &str) -> Schema {
    let u32_col = |name: &str| Field::new(name, DataType::UInt32, false);
    let metadata = HashMap::from([
        ("hotaru.table".to_string(), "kernels".to_string()),
        (
            "hotaru.schema_version".to_string(),
            SCHEMA_VERSION.to_string(),
        ),
        ("hotaru.source".to_string(), source.to_string()),
    ]);
    Schema::new(vec![
        u32_col("seq"),
        u32_col("step"),
        Field::new("name", dict(), false),
        Field::new("demangled", dict(), false),
        Field::new("start_ns", DataType::Int64, false),
        Field::new("end_ns", DataType::Int64, false),
        u32_col("device"),
        u32_col("stream"),
        u32_col("correlation_id"),
        u32_col("grid_x"),
        u32_col("grid_y"),
        u32_col("grid_z"),
        u32_col("block_x"),
        u32_col("block_y"),
        u32_col("block_z"),
        u32_col("registers"),
        u32_col("shared_static"),
        u32_col("shared_dynamic"),
        Field::new("graph_node_id", DataType::Int64, true),
    ])
    .with_metadata(metadata)
}

pub fn kernels_batch(rows: &[KernelRow], source: &str) -> Result<RecordBatch> {
    let schema = Arc::new(kernels_schema(source));
    let mut name = StringDictionaryBuilder::<Int32Type>::new();
    let mut demangled = StringDictionaryBuilder::<Int32Type>::new();
    let mut start = Int64Builder::with_capacity(rows.len());
    let mut end = Int64Builder::with_capacity(rows.len());
    let mut graph = Int64Builder::with_capacity(rows.len());
    let u32s = |f: &dyn Fn(&KernelRow) -> u32| -> ArrayRef {
        let mut b = UInt32Builder::with_capacity(rows.len());
        rows.iter().for_each(|r| b.append_value(f(r)));
        Arc::new(b.finish())
    };
    for r in rows {
        name.append_value(&r.name);
        demangled.append_value(&r.demangled);
        start.append_value(r.start_ns);
        end.append_value(r.end_ns);
        graph.append_option(r.graph_node_id);
    }
    let mut seq = UInt32Builder::with_capacity(rows.len());
    (0..rows.len() as u32).for_each(|i| seq.append_value(i));

    let columns: Vec<ArrayRef> = vec![
        Arc::new(seq.finish()),
        u32s(&|r| r.step),
        Arc::new(name.finish()),
        Arc::new(demangled.finish()),
        Arc::new(start.finish()),
        Arc::new(end.finish()),
        u32s(&|r| r.device),
        u32s(&|r| r.stream),
        u32s(&|r| r.correlation_id),
        u32s(&|r| r.grid[0]),
        u32s(&|r| r.grid[1]),
        u32s(&|r| r.grid[2]),
        u32s(&|r| r.block[0]),
        u32s(&|r| r.block[1]),
        u32s(&|r| r.block[2]),
        u32s(&|r| r.registers),
        u32s(&|r| r.shared_static),
        u32s(&|r| r.shared_dynamic),
        Arc::new(graph.finish()),
    ];
    Ok(RecordBatch::try_new(schema, columns)?)
}

pub fn write_batch(path: &Path, batch: &RecordBatch) -> Result<()> {
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = FileWriter::try_new(file, &batch.schema())?;
    writer.write(batch)?;
    writer.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int64Type, UInt32Type};
    use arrow_ipc::reader::FileReader;

    fn row(name: &str, start_ns: i64, end_ns: i64) -> KernelRow {
        KernelRow {
            start_ns,
            end_ns,
            device: 0,
            stream: 7,
            correlation_id: 1,
            name: name.into(),
            demangled: format!("void {name}()"),
            grid: [64, 1, 1],
            block: [256, 1, 1],
            registers: 32,
            shared_static: 0,
            shared_dynamic: 0,
            graph_node_id: None,
            step: 0,
        }
    }

    #[test]
    fn round_trips_through_an_ipc_file() {
        let rows = vec![row("a", 0, 10), row("b", 10, 25), row("a", 30, 41)];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kernels.arrow");
        write_batch(&path, &kernels_batch(&rows, "test").unwrap()).unwrap();

        let reader = FileReader::try_new(File::open(&path).unwrap(), None).unwrap();
        assert_eq!(reader.schema().metadata()["hotaru.table"], "kernels");
        let batches: Vec<_> = reader.map(Result::unwrap).collect();
        assert_eq!(batches.len(), 1);
        let b = &batches[0];
        assert_eq!(b.num_rows(), 3);

        let names = b
            .column_by_name("name")
            .unwrap()
            .as_dictionary::<Int32Type>();
        // two distinct names, stored once each
        assert_eq!(names.values().len(), 2);
        let ends = b
            .column_by_name("end_ns")
            .unwrap()
            .as_primitive::<Int64Type>();
        assert_eq!(ends.values().as_ref(), &[10, 25, 41]);
        let seq = b
            .column_by_name("seq")
            .unwrap()
            .as_primitive::<UInt32Type>();
        assert_eq!(seq.values().as_ref(), &[0, 1, 2]);
        assert_eq!(b.column_by_name("graph_node_id").unwrap().null_count(), 3);
    }
}
