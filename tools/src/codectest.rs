//! `tools/codectest.cpp`: vertex codec stream filter and compression ratio report.
//!
//! - `codectest <stride>`: encodes stdin (raw vertices) to stdout.
//! - `codectest <stride> <count>`: decodes stdin (encoded vertices) to stdout.
//! - `codectest -test[z] [-<level>] files...`: per-file ratios for `v<count>_s<stride>_<name>[_R...]` files; `-testz` also
//!   measures `lz4`/`zstd` (external programs, like the C++ `system()` calls) on raw and encoded data.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;

use meshoptimizer::{decode_vertex_buffer, encode_vertex_buffer, encode_vertex_buffer_bound, encode_vertex_buffer_level};

/// C `atoi`: optional whitespace and sign, then digits up to the first non-digit; 0 if there are none.
pub fn atoi(s: &str) -> i64 {
    let s = s.trim_start();
    let (sign, digits) = match s.as_bytes().first() {
        Some(b'-') => (-1, &s[1..]),
        Some(b'+') => (1, &s[1..]),
        _ => (1, s),
    };

    let value = digits
        .bytes()
        .take_while(u8::is_ascii_digit)
        .fold(0i64, |v, d| v.saturating_mul(10).saturating_add(i64::from(d - b'0')));
    sign * value
}

/// `%.3f` / `%.2f` like printf, including glibc's `nan`/`-nan`.
fn fixed(v: f64, precision: usize) -> String {
    if v.is_nan() {
        if v.is_sign_negative() { "-nan".into() } else { "nan".into() }
    } else {
        format!("{v:.precision$}")
    }
}

/// Compressed size of `data` from an external compressor; 0 if it can't run (C++: `system()` failure).
fn measure(tool: &str, data: &[u8]) -> usize {
    let dir = crate::temp_dir();
    let suffix = crate::unique_suffix();
    let input: PathBuf = dir.join(format!("codectest-{suffix}.in"));
    let output: PathBuf = dir.join(format!("codectest-{suffix}.out"));

    if std::fs::write(&input, data).is_err() {
        return 0;
    }

    let mut command = Command::new(tool);
    command.args(["-f", "-q"]).arg(&input);
    if tool == "zstd" {
        command.arg("-o");
    }
    command.arg(&output);

    let ok = command.status().is_ok_and(|status| status.success());
    let size = if ok {
        std::fs::metadata(&output).map_or(0, |m| m.len() as usize)
    } else {
        0
    };

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);

    size
}

fn measure_lz4(data: &[u8]) -> usize {
    measure("lz4", data)
}

fn measure_zstd(data: &[u8]) -> usize {
    measure("zstd", data)
}

#[derive(Default)]
struct Stats {
    testz: bool,
    count: f64,
    size: f64,

    ratio: f64,
    ratio_lz4: f64,
    ratio_zstd: f64,

    total: f64,
    total_lz4: f64,
    total_zstd: f64,
}

fn test_data(out: &mut impl Write, input: &[u8], count: usize, stride: usize, level: i32, stats: &mut Stats) -> std::io::Result<()> {
    let mut decoded = vec![0u8; count * stride];
    let res = decode_vertex_buffer(&mut decoded, count, stride, input, input.len());
    match res {
        // some files are not encoded
        Err(_) if input.len() == decoded.len() => decoded.copy_from_slice(input),
        Err(e) => {
            let code = if let meshoptimizer::Error::Decode(code) = e { code } else { -1 };
            return write!(out, " error decoding input: {code}");
        }
        Ok(()) => {}
    }

    let bound = encode_vertex_buffer_bound(count, stride).unwrap_or(0);
    let mut output = vec![0u8; bound];
    let size = encode_vertex_buffer_level(&mut output, bound, &decoded, count, stride, level, -1).unwrap_or(0);
    output.truncate(size);

    let raw = decoded.len() as f64;

    write!(out, " raw {} KB\t", decoded.len() / 1024)?;
    write!(out, " vtx {}", fixed(output.len() as f64 / raw, 3))?;

    stats.count += 1.0;
    stats.size += raw;

    stats.total += output.len() as f64;
    stats.ratio += (output.len() as f64 / raw).ln();

    if stats.testz {
        let decoded_lz4 = measure_lz4(&decoded) as f64;
        let output_lz4 = measure_lz4(&output) as f64;
        let decoded_zstd = measure_zstd(&decoded) as f64;
        let output_zstd = measure_zstd(&output) as f64;

        stats.total_lz4 += output_lz4;
        stats.total_zstd += output_zstd;

        stats.ratio_lz4 += (output_lz4 / raw).ln();
        stats.ratio_zstd += (output_zstd / raw).ln();

        write!(out, "\tlz4 {}", fixed(decoded_lz4 / raw, 3))?;
        write!(out, " vtx+lz4 {}", fixed(output_lz4 / raw, 3))?;

        write!(out, "\tzstd {}", fixed(decoded_zstd / raw, 3))?;
        write!(out, " vtx+zstd {}", fixed(output_zstd / raw, 3))?;
    }

    Ok(())
}

/// `sscanf(name, "v%d_s%d_", ...)`
fn scan_name(name: &str) -> Option<(i64, i64)> {
    let rest = name.strip_prefix('v')?;
    let count_len = rest.find(|c: char| !c.is_ascii_digit())?;
    let stride = rest[count_len..].strip_prefix("_s")?;
    let stride_len = stride.find(|c: char| !c.is_ascii_digit()).unwrap_or(stride.len());

    if count_len == 0 || stride_len == 0 {
        return None;
    }

    Some((atoi(&rest[..count_len]), atoi(&stride[..stride_len])))
}

fn test_file(out: &mut impl Write, path: &str, level: i32, stats: &mut Stats) -> std::io::Result<()> {
    let Ok(input) = std::fs::read(path) else {
        return Ok(());
    };

    let name = path.rsplit('/').next().unwrap_or(path);

    // the display name is between the second '_' and "_R", up to 25 characters
    let name0 = name.splitn(3, '_').nth(2).unwrap_or("");
    let name0 = name0.find("_R").map_or(name0, |end| &name0[..end]);
    let name0: String = name0.chars().take(25).collect();

    write!(out, "{name0:>25}:")?;

    match scan_name(name) {
        Some((count, stride)) if count >= 0 && stride > 0 => {
            test_data(out, &input, count as usize, stride as usize, level, stats)?;
        }
        _ => write!(out, " error parsing file name")?, // C++: assert
    }

    writeln!(out)
}

/// Runs `codectest` with `args` (without the program name); returns the exit code.
pub fn run(args: &[String], stdin: &mut impl Read, stdout: &mut impl Write, stderr: &mut impl Write) -> std::io::Result<u8> {
    if let Some(mode @ ("-test" | "-testz")) = args.first().map(String::as_str) {
        let mut stats = Stats {
            testz: mode == "-testz",
            ..Stats::default()
        };

        let level = match args.get(1).map(String::as_bytes) {
            Some([b'-', d, ..]) if d.is_ascii_digit() => atoi(&args[1][1..]) as i32,
            _ => -1,
        };

        for path in &args[if level < 0 { 1 } else { 2 }..] {
            test_file(stdout, path, if level < 0 { 2 } else { level }, &mut stats)?;
        }

        writeln!(stdout, "---")?;
        write!(
            stdout,
            "{} files: vtx {}",
            stats.count as i32,
            fixed((stats.ratio / stats.count).exp(), 3)
        )?;
        if stats.testz {
            writeln!(
                stdout,
                ", vtx+lz4 {}, vtx+zstd {}",
                fixed((stats.ratio_lz4 / stats.count).exp(), 3),
                fixed((stats.ratio_zstd / stats.count).exp(), 3)
            )?;
        } else {
            writeln!(stdout)?;
        }

        write!(
            stdout,
            "total: {} files, raw {} MB, vtx {} MB",
            stats.count as i32,
            fixed(stats.size / 1024.0 / 1024.0, 2),
            fixed(stats.total / 1024.0 / 1024.0, 2)
        )?;
        if stats.testz {
            writeln!(
                stdout,
                ", vtx+lz4 {} MB, vtx+zstd {} MB",
                fixed(stats.total_lz4 / 1024.0 / 1024.0, 2),
                fixed(stats.total_zstd / 1024.0 / 1024.0, 2)
            )?;
        } else {
            writeln!(stdout)?;
        }

        return Ok(0);
    }

    if args.is_empty() || args.len() > 2 || atoi(&args[0]) <= 0 {
        writeln!(stderr, "Usage: codectest <stride> [<count>]")?;
        return Ok(1);
    }

    let stride = atoi(&args[0]) as usize;

    let mut input = Vec::new();
    stdin.read_to_end(&mut input)?;

    if let Some(count) = args.get(1) {
        // if count is specified, we assume input is meshopt-encoded and decode it first
        let count = usize::try_from(atoi(count)).unwrap_or(0);

        let mut decoded = vec![0u8; count * stride];
        if let Err(e) = decode_vertex_buffer(&mut decoded, count, stride, &input, input.len()) {
            let code = if let meshoptimizer::Error::Decode(code) = e { code } else { -1 };
            writeln!(stderr, "Error decoding input: {code}")?;
            return Ok(2);
        }

        stdout.write_all(&decoded)?;
    } else {
        let vertex_count = input.len() / stride;
        let bound = encode_vertex_buffer_bound(vertex_count, stride).unwrap_or(0);
        let mut output = vec![0u8; bound];
        let size = encode_vertex_buffer(&mut output, bound, &input, vertex_count, stride).unwrap_or(0);

        stdout.write_all(&output[..size])?;
    }

    Ok(0)
}
