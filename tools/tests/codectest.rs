//! codectest output checked against the C++ tool: the same generated files (integer patterns, so any generator
//! reproduces them) were run through an out-of-repo build of `tools/codectest.cpp`.

use std::path::Path;

use meshoptimizer::{encode_vertex_buffer, encode_vertex_buffer_bound};
use meshoptimizer_tools::codectest::{atoi, run};

fn smooth(count: usize, stride: usize) -> Vec<u8> {
    (0..count)
        .flat_map(|v| (0..stride / 4).map(move |k| ((v * (k + 1) * 3 + k * 1000) as u32).to_le_bytes()))
        .flatten()
        .collect()
}

fn encode(data: &[u8], count: usize, stride: usize) -> Vec<u8> {
    let bound = encode_vertex_buffer_bound(count, stride).unwrap_or(0);
    let mut out = vec![0u8; bound];
    let size = encode_vertex_buffer(&mut out, bound, data, count, stride).unwrap_or(0);
    out.truncate(size);
    out
}

fn write_files(dir: &Path) -> std::io::Result<Vec<String>> {
    let noise: Vec<u8> = (0..333u32 * 8).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
    let mut bad = vec![0u8; 31];
    bad[0] = 0xa1;

    let files = [
        ("v1000_s16_smooth_R1.bin", smooth(1000, 16)),
        ("v333_s8_noise.bin", noise),
        ("v2048_s12_encoded.bin", encode(&smooth(2048, 12), 2048, 12)),
        ("v10_s4_bad.bin", bad),
    ];

    let mut paths = Vec::new();
    for (name, data) in files {
        let path = dir.join(name);
        std::fs::write(&path, data)?;
        paths.push(path.to_string_lossy().into_owned());
    }

    Ok(paths)
}

fn run_text(args: &[String], stdin: &[u8]) -> std::io::Result<(u8, Vec<u8>, String)> {
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let code = run(args, &mut &stdin[..], &mut stdout, &mut stderr)?;
    Ok((code, stdout, String::from_utf8_lossy(&stderr).into_owned()))
}

const EXPECTED_TEST: &str = "                   smooth: raw 15 KB\t vtx 0.193
                noise.bin: raw 2 KB\t vtx 1.011
              encoded.bin: raw 24 KB\t vtx 0.171
                  bad.bin: error decoding input: -3
---
3 files: vtx 0.322
total: 3 files, raw 0.04 MB, vtx 0.01 MB
";

const EXPECTED_TEST_LEVEL3: &str = "                   smooth: raw 15 KB\t vtx 0.181
                noise.bin: raw 2 KB\t vtx 1.011
              encoded.bin: raw 24 KB\t vtx 0.171
                  bad.bin: error decoding input: -3
---
3 files: vtx 0.316
total: 3 files, raw 0.04 MB, vtx 0.01 MB
";

#[test]
fn test_mode_matches_reference() -> std::io::Result<()> {
    let dir = meshoptimizer_tools::temp_dir().join(format!("codectest-{}", meshoptimizer_tools::unique_suffix()));
    std::fs::create_dir_all(&dir)?;
    let paths = write_files(&dir)?;

    let args = |extra: &[&str]| -> Vec<String> { extra.iter().map(|s| s.to_string()).chain(paths.iter().cloned()).collect() };

    let (code, out, _) = run_text(&args(&["-test"]), &[])?;
    assert_eq!((code, String::from_utf8_lossy(&out).as_ref()), (0, EXPECTED_TEST));

    let (code, out, _) = run_text(&args(&["-test", "-3"]), &[])?;
    assert_eq!((code, String::from_utf8_lossy(&out).as_ref()), (0, EXPECTED_TEST_LEVEL3));

    std::fs::remove_dir_all(&dir)
}

#[test]
fn encode_decode_roundtrip() -> std::io::Result<()> {
    let raw = smooth(2048, 12);
    let args = |a: &[&str]| -> Vec<String> { a.iter().map(|s| s.to_string()).collect() };

    let (code, encoded, _) = run_text(&args(&["12"]), &raw)?;
    assert_eq!(code, 0);
    assert_eq!(encoded, encode(&raw, 2048, 12));

    let (code, decoded, _) = run_text(&args(&["12", "2048"]), &encoded)?;
    assert_eq!((code, decoded), (0, raw));

    let (code, _, err) = run_text(&args(&["12", "2048"]), &encoded[..10])?;
    assert_eq!((code, err.as_str()), (2, "Error decoding input: -2\n"));

    for usage in [&[][..], &["0"], &["x"], &["4", "1", "2"]] {
        let (code, _, err) = run_text(&args(usage), &[])?;
        assert_eq!(code, 1);
        assert!(err.starts_with("Usage:"));
    }

    Ok(())
}

#[test]
fn atoi_matches_c() {
    assert_eq!(
        [atoi("16"), atoi(" 12abc"), atoi("-3"), atoi("+7"), atoi("x"), atoi("")],
        [16, 12, -3, 7, 0, 0]
    );
}
