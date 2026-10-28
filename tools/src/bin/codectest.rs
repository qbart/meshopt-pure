//! `codectest <stride> [<count>]` encodes (or decodes) stdin to stdout; `codectest -test[z] [-<level>] files...`
//! reports compression ratios.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let code = meshoptimizer_tools::codectest::run(
        &args,
        &mut std::io::stdin().lock(),
        &mut std::io::stdout().lock(),
        &mut std::io::stderr(),
    );

    match code {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}
