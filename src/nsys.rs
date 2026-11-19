//! Reads kernel launches from an nsys sqlite export
//! (`nsys export --type sqlite`) into the `kernels` table.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OpenFlags};

use crate::trace::KernelRow;

const TABLE: &str = "CUPTI_ACTIVITY_KIND_KERNEL";

const REQUIRED: &[&str] = &[
    "start",
    "end",
    "deviceId",
    "streamId",
    "correlationId",
    "shortName",
    "demangledName",
    "gridX",
    "gridY",
    "gridZ",
    "blockX",
    "blockY",
    "blockZ",
    "registersPerThread",
    "staticSharedMemory",
    "dynamicSharedMemory",
];

pub fn read_kernels(path: &Path) -> Result<Vec<KernelRow>> {
    let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("opening {}", path.display()))?;

    let columns = table_columns(&db, TABLE)?;
    if columns.is_empty() {
        bail!(
            "{} has no {TABLE} table. Was the run traced with --trace=cuda, \
             and did the program launch any kernels?",
            path.display()
        );
    }
    let missing: Vec<_> = REQUIRED.iter().filter(|c| !columns.contains(**c)).collect();
    if !missing.is_empty() {
        bail!("{TABLE} is missing columns {missing:?}; this nsys version is not supported yet");
    }
    // graphNodeId is only set when kernels ran inside a cuda graph and nsys
    // traced graph nodes (--cuda-graph-trace=node).
    let graph_node = if columns.contains("graphNodeId") {
        "k.graphNodeId"
    } else {
        "NULL"
    };

    let sql = format!(
        "SELECT k.start, k.end, k.deviceId, k.streamId, k.correlationId,
                s.value, d.value,
                k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
                k.registersPerThread, k.staticSharedMemory, k.dynamicSharedMemory,
                {graph_node}
         FROM {TABLE} k
         JOIN StringIds s ON s.id = k.shortName
         JOIN StringIds d ON d.id = k.demangledName
         ORDER BY k.start, k.correlationId"
    );
    let mut stmt = db.prepare(&sql)?;
    let rows = stmt.query_map([], |r| {
        Ok(KernelRow {
            start_ns: r.get(0)?,
            end_ns: r.get(1)?,
            device: r.get(2)?,
            stream: r.get(3)?,
            correlation_id: r.get(4)?,
            name: r.get(5)?,
            demangled: r.get(6)?,
            grid: [r.get(7)?, r.get(8)?, r.get(9)?],
            block: [r.get(10)?, r.get(11)?, r.get(12)?],
            registers: r.get(13)?,
            shared_static: r.get(14)?,
            shared_dynamic: r.get(15)?,
            graph_node_id: r.get(16)?,
            step: 0,
        })
    })?;
    let mut kernels = rows.collect::<rusqlite::Result<Vec<_>>>()?;

    let readbacks = readback_starts(&db)?;
    for k in &mut kernels {
        k.step = readbacks.partition_point(|&t| t < k.start_ns) as u32;
    }

    let total = count(&db, &format!("SELECT count(*) FROM {TABLE}"))?;
    if total != kernels.len() as i64 {
        bail!(
            "{TABLE} has {total} rows but only {} joined to StringIds",
            kernels.len()
        );
    }
    Ok(kernels)
}

/// Start times of every device-to-host copy, sorted.
///
/// llama.cpp copies the logits back to the host once at the end of every
/// eval (warmup, prompt, then one per generated token), so these copies
/// split the kernel stream into steps.
fn readback_starts(db: &Connection) -> Result<Vec<i64>> {
    const COPY_DTOH: i64 = 2; // ENUM_CUDA_MEMCPY_OPER
    if table_columns(db, "CUPTI_ACTIVITY_KIND_MEMCPY")?.is_empty() {
        return Ok(Vec::new());
    }
    let mut stmt = db.prepare(
        "SELECT start FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind = ?1 ORDER BY start",
    )?;
    let starts = stmt.query_map([COPY_DTOH], |r| r.get(0))?;
    Ok(starts.collect::<rusqlite::Result<_>>()?)
}

fn table_columns(db: &Connection, table: &str) -> Result<HashSet<String>> {
    let mut stmt = db.prepare(&format!("PRAGMA table_info({table})"))?;
    let names = stmt.query_map([], |r| r.get::<_, String>(1))?;
    Ok(names.collect::<rusqlite::Result<_>>()?)
}

fn count(db: &Connection, sql: &str) -> Result<i64> {
    Ok(db.query_row(sql, [], |r| r.get(0))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The subset of the nsys export schema that the reader touches.
    fn fixture(dir: &Path, graph_nodes: bool) -> std::path::PathBuf {
        let path = dir.join("run.sqlite");
        let db = Connection::open(&path).unwrap();
        let graph_col = if graph_nodes {
            ", graphNodeId INTEGER"
        } else {
            ""
        };
        db.execute_batch(&format!(
            "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
             CREATE TABLE {TABLE} (
                start INTEGER NOT NULL, end INTEGER NOT NULL,
                deviceId INTEGER NOT NULL, contextId INTEGER NOT NULL,
                streamId INTEGER NOT NULL, correlationId INTEGER,
                globalPid INTEGER, demangledName INTEGER NOT NULL,
                shortName INTEGER NOT NULL, mangledName INTEGER,
                gridX INTEGER NOT NULL, gridY INTEGER NOT NULL, gridZ INTEGER NOT NULL,
                blockX INTEGER NOT NULL, blockY INTEGER NOT NULL, blockZ INTEGER NOT NULL,
                registersPerThread INTEGER NOT NULL,
                staticSharedMemory INTEGER NOT NULL, dynamicSharedMemory INTEGER NOT NULL
                {graph_col});
             INSERT INTO StringIds VALUES
                (1, 'mul_mat_vec_q'), (2, 'void mul_mat_vec_q<(ggml_type)12, 1>(...)'),
                (3, 'rms_norm_f32'), (4, 'void rms_norm_f32<1024>(...)');"
        ))
        .unwrap();
        // inserted out of order on purpose
        let insert = format!(
            "INSERT INTO {TABLE} (start, end, deviceId, contextId, streamId, correlationId,
                globalPid, demangledName, shortName, gridX, gridY, gridZ, blockX, blockY,
                blockZ, registersPerThread, staticSharedMemory, dynamicSharedMemory)
             VALUES (?1, ?2, 0, 1, 7, ?3, 99, ?4, ?5, ?6, 1, 1, ?7, 1, 1, 40, 0, ?8)"
        );
        db.execute(&insert, (5_000, 9_500, 12, 2, 1, 4096, 32, 512))
            .unwrap();
        db.execute(&insert, (1_000, 1_800, 11, 4, 3, 1, 1024, 0))
            .unwrap();
        path
    }

    #[test]
    fn reads_kernels_in_start_order() {
        let dir = tempfile::tempdir().unwrap();
        let kernels = read_kernels(&fixture(dir.path(), false)).unwrap();
        assert_eq!(kernels.len(), 2);
        assert_eq!(kernels[0].name, "rms_norm_f32");
        assert_eq!(kernels[0].duration_ns(), 800);
        assert_eq!(kernels[1].name, "mul_mat_vec_q");
        assert_eq!(kernels[1].grid, [4096, 1, 1]);
        assert_eq!(kernels[1].block, [32, 1, 1]);
        assert_eq!(kernels[1].shared_dynamic, 512);
        assert_eq!(kernels[1].stream, 7);
        assert_eq!(kernels[1].graph_node_id, None);
        // no memcpy table in this export, so nothing splits the stream
        assert!(kernels.iter().all(|k| k.step == 0));
    }

    #[test]
    fn splits_steps_at_logit_readbacks() {
        let dir = tempfile::tempdir().unwrap();
        let path = fixture(dir.path(), false);
        // a host-to-device copy (kind 1) must not count as a step boundary
        Connection::open(&path)
            .unwrap()
            .execute_batch(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
                    start INTEGER NOT NULL, end INTEGER NOT NULL,
                    bytes INTEGER NOT NULL, copyKind INTEGER NOT NULL);
                 INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES
                    (500, 600, 64, 1), (2_000, 2_100, 607744, 2), (9_600, 9_700, 607744, 2);",
            )
            .unwrap();
        let kernels = read_kernels(&path).unwrap();
        let steps: Vec<_> = kernels.iter().map(|k| k.step).collect();
        assert_eq!(steps, [0, 1]);
    }

    #[test]
    fn reads_graph_node_ids_when_present() {
        let dir = tempfile::tempdir().unwrap();
        let path = fixture(dir.path(), true);
        Connection::open(&path)
            .unwrap()
            .execute(
                &format!("UPDATE {TABLE} SET graphNodeId = correlationId + 100"),
                [],
            )
            .unwrap();
        let kernels = read_kernels(&path).unwrap();
        assert_eq!(kernels[0].graph_node_id, Some(111));
    }

    #[test]
    fn rejects_exports_without_kernels() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.sqlite");
        Connection::open(&path)
            .unwrap()
            .execute_batch("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);")
            .unwrap();
        let err = read_kernels(&path).unwrap_err().to_string();
        assert!(err.contains("no CUPTI_ACTIVITY_KIND_KERNEL table"), "{err}");
    }
}
