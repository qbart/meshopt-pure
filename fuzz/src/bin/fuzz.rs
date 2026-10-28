//! `fuzz <codec|simplify|cluster> [iterations] [seed]` runs the mutation fuzzer and saves a crashing input to
//! `fuzz-<target>-crash.bin`; `fuzz <target> --replay <file>` reruns a saved input.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(target) = args.get(1).and_then(|name| meshoptimizer_fuzz::target(name)) else {
        eprintln!("Usage: fuzz <codec|simplify|cluster> [iterations] [seed] | fuzz <target> --replay <file>");
        return ExitCode::from(2);
    };
    let name = args[1].as_str();

    if args.get(2).map(String::as_str) == Some("--replay") {
        let Some(data) = args.get(3).and_then(|path| std::fs::read(path).ok()) else {
            eprintln!("Error: cannot read replay file");
            return ExitCode::from(2);
        };
        target(&data);
        println!("{name}: replay ok");
        return ExitCode::SUCCESS;
    }

    let iterations = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100_000u64);
    let seed = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0u64);
    let max_len = if name == "codec" { 8192 } else { 1024 };

    match meshoptimizer_fuzz::driver::run(name, target, iterations, seed, max_len) {
        None => {
            println!("{name}: {iterations} inputs ok (seed {seed})");
            ExitCode::SUCCESS
        }
        Some(input) => {
            let path = format!("fuzz-{name}-crash.bin");
            let _ = std::fs::write(&path, &input);
            eprintln!("{name}: crash on a {} byte input, saved to {path}", input.len());
            ExitCode::FAILURE
        }
    }
}
