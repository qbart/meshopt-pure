//! Golden tests: every case in `data/cases.txt` must produce byte-identical output to the C++ gltfpack reference
//! (`data/expected/<name>`, regenerated with `gen-golden.sh`).

use std::path::{Path, PathBuf};

struct Case {
    name: String,
    input: String,
    options: Vec<String>,
    ext: String,
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn cases() -> Vec<Case> {
    let text = std::fs::read_to_string(data_dir().join("cases.txt")).unwrap_or_default();

    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .filter_map(|l| {
            let parts: Vec<&str> = l.split('|').map(str::trim).collect();
            let [name, input, options, ext] = parts[..] else {
                return None;
            };
            Some(Case {
                name: name.to_string(),
                input: input.to_string(),
                options: options.split_whitespace().map(str::to_string).collect(),
                ext: ext.to_string(),
            })
        })
        .collect()
}

fn files(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut result: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| e.path().is_file())
                .map(|e| {
                    (
                        e.file_name().to_string_lossy().into_owned(),
                        std::fs::read(e.path()).unwrap_or_default(),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    result.sort();
    result
}

/// Runs a case and returns a description of the mismatch, if any.
fn run_case(case: &Case, tmp: &Path) -> Option<String> {
    let out_dir = tmp.join(&case.name);
    let _ = std::fs::remove_dir_all(&out_dir);
    if std::fs::create_dir_all(&out_dir).is_err() {
        return Some(format!("{}: cannot create output directory", case.name));
    }

    let input = data_dir().join(&case.input);
    let output = out_dir.join(format!("out.{}", case.ext));

    let mut args = vec![
        "-i".to_string(),
        input.to_string_lossy().into_owned(),
        "-o".to_string(),
        output.to_string_lossy().into_owned(),
    ];
    args.extend(case.options.iter().cloned());

    let rc = gltfpack::run(&args);
    if std::fs::write(out_dir.join("exitcode"), format!("{rc}\n")).is_err() {
        return Some(format!("{}: cannot write exit code", case.name));
    }

    let expected = files(&data_dir().join("expected").join(&case.name));
    let actual = files(&out_dir);

    if expected.is_empty() {
        return Some(format!("{}: no expected outputs", case.name));
    }

    let names = |f: &[(String, Vec<u8>)]| f.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
    if names(&expected) != names(&actual) {
        return Some(format!(
            "{}: files {:?} != expected {:?}",
            case.name,
            names(&actual),
            names(&expected)
        ));
    }

    for ((name, e), (_, a)) in expected.iter().zip(&actual) {
        if e != a {
            let at = e.iter().zip(a).position(|(x, y)| x != y).unwrap_or(e.len().min(a.len()));
            return Some(format!(
                "{}: {name} differs at byte {at} (sizes {} vs expected {})",
                case.name,
                a.len(),
                e.len()
            ));
        }
    }

    None
}

#[test]
fn golden_outputs_match_reference() {
    let cases = cases();
    assert!(cases.len() > 200, "missing test cases");

    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR")).join("gltfpack-golden");

    let failures = if cfg!(target_family = "wasm") {
        // no threads on WASI
        cases.iter().filter_map(|c| run_case(c, &tmp)).collect()
    } else {
        run_parallel(&cases, &tmp)
    };

    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

fn run_parallel(cases: &[Case], tmp: &Path) -> Vec<String> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = cases.len().div_ceil(threads);

    std::thread::scope(|s| {
        let handles: Vec<_> = cases
            .chunks(chunk)
            .map(|part| s.spawn(|| part.iter().filter_map(|c| run_case(c, tmp)).collect::<Vec<_>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap_or_else(|_| vec!["worker panicked".to_string()]))
            .collect()
    })
}
