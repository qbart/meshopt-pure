//! `codecbench [-v] [-l] [files...]`: decoder throughput on a synthetic grid, or on `v<count>_s<stride>_*` /
//! `i<count>_*` stream files.

use std::process::ExitCode;

use meshoptimizer_tools::codecbench::{bench_files, bench_synthetic};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let verbose = args.iter().any(|a| a == "-v");
    let loops = if args.iter().any(|a| a == "-l") { 100 } else { 1 };
    let inputs: Vec<&str> = args.iter().filter(|a| !a.starts_with('-')).map(String::as_str).collect();

    let result = if inputs.is_empty() {
        bench_synthetic(1000, loops, verbose)
    } else {
        bench_files(&inputs, loops)
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
