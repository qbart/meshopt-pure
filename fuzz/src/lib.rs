//! Ports of `tools/codecfuzz.cpp`, `tools/simplifyfuzz.cpp` and `tools/clusterfuzz.cpp`.
//! Each `test_one_input` mirrors `LLVMFuzzerTestOneInput`; C++ `assert`s are `assert!`s, and any panic (including an
//! out of bounds access inside the library) is a bug, since the library must never panic.
//! `driver` is a pure Rust mutation fuzzer that feeds them (libFuzzer would be a C++ dependency, see docs/spec/36-fuzz.md).
#![deny(unsafe_code)]

pub mod clusterfuzz;
pub mod codecfuzz;
pub mod crand;
pub mod driver;
pub mod simplifyfuzz;

/// Fuzz target by name: `codec`, `simplify` or `cluster`.
pub fn target(name: &str) -> Option<fn(&[u8])> {
    match name {
        "codec" => Some(codecfuzz::test_one_input),
        "simplify" => Some(simplifyfuzz::test_one_input),
        "cluster" => Some(clusterfuzz::test_one_input),
        _ => None,
    }
}
