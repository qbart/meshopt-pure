//! File helpers (`fileio.cpp`).

pub fn get_temp_prefix() -> String {
    #[cfg(windows)]
    {
        let temp_dir = std::env::var("TEMP").unwrap_or_else(|_| ".".to_string());
        format!("{temp_dir}\\gltfpack-temp{}", std::process::id())
    }
    #[cfg(not(windows))]
    {
        format!("/tmp/gltfpack-temp{}", std::process::id())
    }
}

/// `path` relative to the directory of `base_path`.
pub fn get_full_path(path: &str, base_path: &str) -> String {
    let slash = base_path.rfind(['/', '\\']);
    let mut result = base_path[..slash.map_or(0, |s| s + 1)].to_string();
    result.push_str(path);
    result
}

/// File name without directory and extension.
pub fn get_file_name(path: &str) -> String {
    let mut result = path;

    if let Some(slash) = result.rfind(['/', '\\']) {
        result = &result[slash + 1..];
    }

    if let Some(dot) = result.rfind('.') {
        result = &result[..dot];
    }

    result.to_string()
}

/// Lowercase extension including the dot (empty if none).
pub fn get_extension(path: &str) -> String {
    let slash = path.rfind(['/', '\\']);
    let mut dot = path.rfind('.');

    if let (Some(s), Some(d)) = (slash, dot)
        && d < s
    {
        dot = None;
    }

    let result = match dot {
        Some(d) => &path[d..],
        None => "",
    };

    result.to_ascii_lowercase()
}

/// `readFile`: fails for missing or empty files.
pub fn read_file(path: &str) -> Option<Vec<u8>> {
    let data = std::fs::read(path).ok()?;
    if data.is_empty() { None } else { Some(data) }
}

pub fn write_file(path: &str, data: &[u8]) -> bool {
    std::fs::write(path, data).is_ok()
}

pub fn remove_file(path: &str) {
    let _ = std::fs::remove_file(path);
}
