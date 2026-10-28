//! Ports of meshoptimizer `tools/` programs.

pub mod codecbench;
pub mod codectest;

/// A name suffix that is unique per process and call, for temp files. Not `std::process::id`: WASI doesn't support it.
pub fn unique_suffix() -> String {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("{nanos:x}-{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Directory for temp files: `TMPDIR`, else the platform default; `/tmp` on WASI, where `std::env::temp_dir` panics.
pub fn temp_dir() -> std::path::PathBuf {
    match std::env::var_os("TMPDIR") {
        Some(dir) => dir.into(),
        None if cfg!(target_os = "wasi") => "/tmp".into(),
        None => std::env::temp_dir(),
    }
}
