//! C standard library number parsing with glibc semantics (`atoi`, `atoll`, `atof`, `strtol`, `strtod`).

fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r')
}

fn at(s: &[u8], i: usize) -> u8 {
    s.get(i).copied().unwrap_or(0)
}

/// `strtoll(s, &end, 10)`: returns the value (saturated on overflow) and the number of bytes consumed (0 if no digits).
pub fn strtoll(s: &[u8]) -> (i64, usize) {
    let mut i = 0;
    while is_space(at(s, i)) {
        i += 1;
    }

    let negative = at(s, i) == b'-';
    if matches!(at(s, i), b'-' | b'+') {
        i += 1;
    }

    let digits_start = i;
    let mut value: i128 = 0;
    while at(s, i).is_ascii_digit() {
        if value <= i128::from(i64::MAX) + 1 {
            value = value * 10 + i128::from(at(s, i) - b'0');
        }
        i += 1;
    }

    if i == digits_start {
        return (0, 0);
    }

    let value = if negative { -value } else { value };
    let value = value.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
    (value, i)
}

/// `strtol` (`long` is 64-bit on the reference platform).
pub fn strtol(s: &[u8]) -> (i64, usize) {
    strtoll(s)
}

/// `atoi`: glibc implements it as `(int)strtol(s, NULL, 10)`.
pub fn atoi(s: &[u8]) -> i32 {
    strtol(s).0 as i32
}

/// `atoll`
pub fn atoll(s: &[u8]) -> i64 {
    strtoll(s).0
}

fn starts_with_ignore_case(s: &[u8], i: usize, pat: &[u8]) -> bool {
    pat.iter().enumerate().all(|(k, &p)| at(s, i + k).eq_ignore_ascii_case(&p))
}

fn hex_value(c: u8) -> Option<u32> {
    (c as char).to_digit(16)
}

/// Parses a hexadecimal floating point literal body (after `0x`); returns the value and consumed bytes (0 if no digits).
fn parse_hex_float(s: &[u8], start: usize) -> (f64, usize) {
    let mut i = start;
    let mut mantissa = 0.0f64;
    let mut exponent = 0i64;
    let mut any = false;

    while let Some(d) = hex_value(at(s, i)) {
        mantissa = mantissa * 16.0 + f64::from(d);
        any = true;
        i += 1;
    }

    if at(s, i) == b'.' && (any || hex_value(at(s, i + 1)).is_some()) {
        i += 1;
        while let Some(d) = hex_value(at(s, i)) {
            mantissa = mantissa * 16.0 + f64::from(d);
            exponent -= 4;
            any = true;
            i += 1;
        }
    }

    if !any {
        return (0.0, 0);
    }

    if matches!(at(s, i), b'p' | b'P') {
        let (e, n) = strtoll(&s[i + 1..]);
        // the exponent must start with a sign or digit right after 'p'
        if n > 0 && !is_space(at(s, i + 1)) {
            exponent = exponent.saturating_add(e);
            i += 1 + n;
        }
    }

    let exponent = exponent.clamp(-10000, 10000) as i32;
    (mantissa * 2f64.powi(exponent / 2) * 2f64.powi(exponent - exponent / 2), i)
}

/// `strtod(s, &end)`: returns the value and the number of bytes consumed (0 if nothing was parsed).
/// Decimal conversion is correctly rounded (like glibc); hexadecimal literals are approximated beyond 53 mantissa bits.
pub fn strtod(s: &[u8]) -> (f64, usize) {
    let mut i = 0;
    while is_space(at(s, i)) {
        i += 1;
    }

    let negative = at(s, i) == b'-';
    if matches!(at(s, i), b'-' | b'+') {
        i += 1;
    }

    let sign = if negative { -1.0 } else { 1.0 };

    if starts_with_ignore_case(s, i, b"inf") {
        let n = if starts_with_ignore_case(s, i, b"infinity") { 8 } else { 3 };
        return (sign * f64::INFINITY, i + n);
    }

    if starts_with_ignore_case(s, i, b"nan") {
        let mut end = i + 3;
        if at(s, end) == b'(' {
            let mut k = end + 1;
            while at(s, k).is_ascii_alphanumeric() || at(s, k) == b'_' {
                k += 1;
            }
            if at(s, k) == b')' {
                end = k + 1;
            }
        }
        return (sign * f64::NAN, end);
    }

    if at(s, i) == b'0' && matches!(at(s, i + 1), b'x' | b'X') {
        let (v, end) = parse_hex_float(s, i + 2);
        if end > 0 {
            return (sign * v, end);
        }
        // "0x" without hex digits parses as "0"
        return (sign * 0.0, i + 1);
    }

    let mut text = String::new();
    let int_start = i;
    while at(s, i).is_ascii_digit() {
        text.push(at(s, i) as char);
        i += 1;
    }
    let mut digits = i - int_start;

    if at(s, i) == b'.' {
        let frac_start = i + 1;
        let mut k = frac_start;
        while at(s, k).is_ascii_digit() {
            k += 1;
        }
        if digits > 0 || k > frac_start {
            text.push('.');
            for &c in &s[frac_start..k] {
                text.push(c as char);
            }
            digits += k - frac_start;
            i = k;
        }
    }

    if digits == 0 {
        return (0.0, 0);
    }

    if matches!(at(s, i), b'e' | b'E') {
        let mut k = i + 1;
        let exp_negative = at(s, k) == b'-';
        if matches!(at(s, k), b'-' | b'+') {
            k += 1;
        }
        if at(s, k).is_ascii_digit() {
            let mut exp: i64 = 0;
            while at(s, k).is_ascii_digit() {
                exp = (exp * 10 + i64::from(at(s, k) - b'0')).min(1_000_000_000);
                k += 1;
            }
            text.push('e');
            if exp_negative {
                text.push('-');
            }
            text.push_str(&exp.to_string());
            i = k;
        }
    }

    if text.starts_with('.') {
        text.insert(0, '0');
    }
    if text.ends_with('.') {
        text.push('0');
    }
    let text = text.replacen(".e", ".0e", 1);

    let value = text.parse::<f64>().unwrap_or(0.0);
    (sign * value, i)
}

/// `atof`
pub fn atof(s: &[u8]) -> f64 {
    strtod(s).0
}

/// Splits Rust's `{:e}` output (`d.ddde-X`) into mantissa and exponent.
fn split_exp(s: &str) -> (&str, i32) {
    match s.split_once('e') {
        Some((m, e)) => (m, e.parse().unwrap_or(0)),
        None => (s, 0),
    }
}

fn strip_fraction_zeros(s: &mut String) {
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
}

/// `printf("%.<precision>g", v)` with glibc semantics (exact decimal conversion, round-half-even).
pub fn format_g(v: f64, precision: usize) -> String {
    let p = precision.max(1);

    if v.is_nan() {
        return if v.is_sign_negative() { "-nan".into() } else { "nan".into() };
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf".into() } else { "inf".into() };
    }

    // exponent of the value after rounding to p significant digits
    let sci = format!("{:.*e}", p - 1, v);
    let (mantissa, x) = split_exp(&sci);

    if x < -4 || x >= p as i32 {
        let mut m = mantissa.to_string();
        strip_fraction_zeros(&mut m);
        let sign = if x < 0 { '-' } else { '+' };
        format!("{m}e{sign}{:02}", x.unsigned_abs())
    } else {
        let mut f = format!("{:.*}", (p as i32 - 1 - x) as usize, v);
        strip_fraction_zeros(&mut f);
        f
    }
}

/// `printf("%.<precision>f", v)` with glibc semantics.
pub fn format_f(v: f64, precision: usize) -> String {
    if v.is_nan() {
        return if v.is_sign_negative() { "-nan".into() } else { "nan".into() };
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf".into() } else { "inf".into() };
    }
    format!("{v:.precision$}")
}

/// `printf("%.<precision>e", v)` with glibc semantics.
pub fn format_e(v: f64, precision: usize) -> String {
    if v.is_nan() {
        return if v.is_sign_negative() { "-nan".into() } else { "nan".into() };
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf".into() } else { "inf".into() };
    }
    let sci = format!("{v:.precision$e}");
    let (m, x) = split_exp(&sci);
    let sign = if x < 0 { '-' } else { '+' };
    format!("{m}e{sign}{:02}", x.unsigned_abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers() {
        assert_eq!(atoi(b"42"), 42);
        assert_eq!(atoi(b"  -7xyz"), -7);
        assert_eq!(atoi(b"1.5"), 1);
        assert_eq!(atoi(b"true"), 0);
        assert_eq!(atoi(b"99999999999"), 99999999999i64 as i32);
        assert_eq!(atoi(b"999999999999999999999"), -1);
        assert_eq!(atoll(b"-999999999999999999999"), i64::MIN);
    }

    #[test]
    fn floats() {
        assert_eq!(atof(b"1.5"), 1.5);
        assert_eq!(atof(b"-2e3,"), -2000.0);
        assert_eq!(atof(b".5"), 0.5);
        assert_eq!(atof(b"5."), 5.0);
        assert_eq!(atof(b"1e"), 1.0);
        assert_eq!(atof(b"1e+"), 1.0);
        assert_eq!(atof(b"0x10"), 16.0);
        assert_eq!(atof(b"0x1p3"), 8.0);
        assert_eq!(atof(b"0x"), 0.0);
        assert_eq!(atof(b"-inf"), f64::NEG_INFINITY);
        assert!(atof(b"nan").is_nan());
        assert_eq!(atof(b"null"), 0.0);
        assert_eq!(atof(b"1e400"), f64::INFINITY);
        assert_eq!(strtod(b"abc"), (0.0, 0));
        assert_eq!(strtod(b"12.5e-1x"), (1.25, 7));
        assert_eq!(atof(b"0.1") as f32, 0.1f32);
    }

    #[test]
    fn printf() {
        assert_eq!(format_g(0.0, 9), "0");
        assert_eq!(format_g(-0.0, 9), "-0");
        assert_eq!(format_g(1.0, 9), "1");
        assert_eq!(format_g(0.1f32 as f64, 9), "0.100000001");
        assert_eq!(format_g(123456789.0, 9), "123456789");
        assert_eq!(format_g(1234567890.0, 9), "1.23456789e+09");
        assert_eq!(format_g(0.0001, 9), "0.0001");
        assert_eq!(format_g(0.00001, 9), "1e-05");
        assert_eq!(format_g(f32::MAX as f64, 9), "3.40282347e+38");
        assert_eq!(format_g(0.5, 1), "0.5");
        assert_eq!(format_g(2.5, 1), "2");
        assert_eq!(format_g(3.5, 1), "4");
        assert_eq!(format_f(2.5, 0), "2");
        assert_eq!(format_f(0.125, 2), "0.12");
        assert_eq!(format_e(1234.5, 2), "1.23e+03");
    }
}
