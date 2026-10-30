//! Demo output checked against the C++ demo (`demo/main.cpp` + `nanite.cpp`, built out of repo): the golden files in
//! `tests/data/*.txt` are its raw output, compared after masking timings (msec, GB/s, tri/sec, ns/meshlet).
//! Not built for Wasm: the tests spawn the demo binary, which WASI can't do. `DEMO_RUNNER` (e.g. qemu) runs it when cross
//! testing.
#![cfg(not(target_family = "wasm"))]

use std::path::Path;
use std::process::Command;

/// Replaces timing-dependent numbers with `T`.
fn mask(text: &str) -> String {
    let mut out = String::new();

    for line in text.lines() {
        let words: Vec<&str> = line.split(' ').collect();
        let mut masked = Vec::with_capacity(words.len());

        for (i, word) in words.iter().enumerate() {
            let next = words.get(i + 1).copied().unwrap_or("");
            let timing = next.starts_with("msec") || next.starts_with("ns/meshlet") || next.starts_with("GB/sec");

            masked.push(if timing {
                "T"
            } else if next.starts_with("GB/s)") && word.starts_with('(') {
                "(T"
            } else if word.ends_with('B') && next.starts_with("tri/sec") {
                "TB"
            } else {
                word
            });
        }

        out.push_str(&masked.join(" "));
        out.push('\n');
    }

    out
}

fn run(dir: &Path, args: &[&str], env: &[(&str, &str)]) -> Result<String, String> {
    let exe = env!("CARGO_BIN_EXE_demo");
    let mut command = match std::env::var("DEMO_RUNNER") {
        Ok(runner) => {
            let mut command = Command::new(runner);
            command.arg(exe);
            command
        }
        Err(_) => Command::new(exe),
    };

    let output = command
        .args(args)
        .current_dir(dir)
        .env_remove("CLRT")
        .env_remove("DUMP")
        .envs(env.iter().copied())
        .output()
        .map_err(|e| e.to_string())?;

    if !output.status.success() {
        return Err(format!("demo {args:?} failed: {}", String::from_utf8_lossy(&output.stderr)));
    }

    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn check(dir: &Path, args: &[&str], env: &[(&str, &str)], golden: &str) -> Result<(), String> {
    let expected =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(golden)).map_err(|e| e.to_string())?;
    let actual = run(dir, args, env)?;

    assert_eq!(mask(&actual), mask(&expected), "demo {args:?} vs {golden}");
    Ok(())
}

fn data() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn models() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../gltfpack/tests/data/models")
}

#[test]
fn process_pirate() -> Result<(), String> {
    check(&models(), &["pirate.obj"], &[], "pirate.txt")
}

#[test]
fn process_ngons() -> Result<(), String> {
    check(&data(), &["hexes.obj"], &[], "hexes.txt")
}

#[test]
fn nanite_pirate() -> Result<(), String> {
    check(&models(), &["-n", "pirate.obj"], &[], "pirate-nanite.txt")
}

#[test]
fn nanite_pirate_rt() -> Result<(), String> {
    check(&models(), &["-n", "pirate.obj"], &[("CLRT", "0.5")], "pirate-nanite-clrt.txt")
}

#[test]
fn nanite_ngons() -> Result<(), String> {
    check(&data(), &["-n", "hexes.obj"], &[], "hexes-nanite.txt")
}

#[test]
fn dev_pirate() -> Result<(), String> {
    check(&models(), &["-d", "pirate.obj"], &[], "pirate-dev.txt")
}

#[test]
fn mask_timings() {
    let line = "IdxCodec : 9.1 bits/triangle; encode 0.09 msec (0.654 GB/s), decode time 0.007 msec, 0.741B tri/sec, 112.7 ns/meshlet, 10.1 GB/sec\n";
    assert_eq!(
        mask(line),
        "IdxCodec : 9.1 bits/triangle; encode T msec (T GB/s), decode time T msec, TB tri/sec, T ns/meshlet, T GB/sec\n"
    );
}
