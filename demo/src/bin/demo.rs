//! `demo [-d|-n] files...`: prints statistics for every algorithm on each OBJ file (`-d`: remesh only, `-n`: nanite).
//! C++ runs `demo/tests.cpp` without arguments and after processing files; those tests are ported to the crate's
//! `tests/` and run with `cargo test`.

use std::process::ExitCode;

use meshoptimizer_demo::demo::{Result, process, process_dev, process_nanite};

type Run = fn(&str) -> Result<()>;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let (run, paths): (Run, &[String]) = match args.first().map(String::as_str) {
        None => {
            println!("Usage: demo [-d|-n] files...; the C++ demo tests (demo/tests.cpp) run with `cargo test -p meshoptimizer`");
            return ExitCode::SUCCESS;
        }
        Some("-d") => (process_dev, &args[1..]),
        Some("-n") => (process_nanite, &args[1..]),
        Some(_) => (process, &args[..]),
    };

    for path in paths {
        if let Err(e) = run(path) {
            eprintln!("Error processing {path}: {e}");
            return ExitCode::FAILURE;
        }
    }

    ExitCode::SUCCESS
}
