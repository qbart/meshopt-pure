//! JSON output helpers (`json.cpp`).

use crate::cstd::format_g;

/// Appends `,` unless the string is empty or ends with `[`/`{`.
pub fn comma(s: &mut String) {
    let ch = s.as_bytes().last().copied().unwrap_or(0);

    if ch != 0 && ch != b'[' && ch != b'{' {
        s.push(',');
    }
}

/// `append(std::string&, size_t)`
pub fn append_size(s: &mut String, v: usize) {
    s.push_str(&v.to_string());
}

/// `append(std::string&, float)`: `%.9g`, with +-inf sanitized to +-FLT_MAX and NaN to FLT_MAX.
pub fn append_float(s: &mut String, v: f32) {
    // it would be more consistent to use null for NaN but that makes JSON invalid, and 0 makes it hard to distinguish from valid values
    let sv = if v.abs() < f32::MAX {
        v
    } else if v < 0.0 {
        -f32::MAX
    } else {
        f32::MAX
    };

    s.push_str(&format_g(f64::from(sv), 9));
}

/// `append(std::string&, const char*)`
pub fn append(s: &mut String, v: &str) {
    s.push_str(v);
}

/// `append(std::string&, const float*, size_t)`: JSON array of floats.
pub fn append_floats(s: &mut String, data: &[f32]) {
    s.push('[');
    for (i, &v) in data.iter().enumerate() {
        if i != 0 {
            s.push(',');
        }
        append_float(s, v);
    }
    s.push(']');
}

/// `appendJson`: copies JSON text, dropping whitespace outside of quoted strings.
pub fn append_json(s: &mut String, data: &str) {
    #[derive(PartialEq)]
    enum State {
        None,
        Escape,
        Quoted,
    }

    let mut state = State::None;

    for ch in data.chars() {
        // whitespace outside of quoted strings can be ignored (C isspace in the "C" locale)
        if state != State::None || !matches!(ch, ' ' | '\t' | '\n' | '\x0b' | '\x0c' | '\r') {
            s.push(ch);
        }

        // the finite automata tracks whether we're inside a quoted string
        state = match state {
            State::None => {
                if ch == '"' {
                    State::Quoted
                } else {
                    State::None
                }
            }
            State::Quoted => {
                if ch == '"' {
                    State::None
                } else if ch == '\\' {
                    State::Escape
                } else {
                    State::Quoted
                }
            }
            State::Escape => State::Quoted,
        };
    }
}
