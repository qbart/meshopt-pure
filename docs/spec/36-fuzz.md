# 36 — fuzz harnesses (`tools/codecfuzz.cpp`, `simplifyfuzz.cpp`, `clusterfuzz.cpp`)

Workspace crate `fuzz/` (`meshoptimizer-fuzz`, not published).

## Harnesses (1:1 ports of `LLVMFuzzerTestOneInput`)
- `codecfuzz::test_one_input`:
  - decoders on untrusted data: index buffer/sequence (u16/u32) and vertex buffer (strides 4/16/24/32), 66 elements;
  - vertex encode→decode roundtrip, with a second encode into the exact tail of the buffer
    (`encode_vertex_version(level < 4 ? 1 : 0)`, level = `data[0] % 5`);
  - meshlet decode (raw, u16/u32 vertices × u8/u32 triangles) into the C++ tail-of-buffer slices;
  - meshlet roundtrips for triangles (modulo rotation) and vertices.
- `simplifyfuzz::test_one_input`: `simplify` with 6 option sets, `simplify_with_attributes`, then
  `simplify_with_update_interleaved` (C++ passes `vb[0] + 3` as aliased attributes).
- `clusterfuzz::test_one_input`: `build_meshlets`, `_scan`, `_flex`, `_spatial` with the C++ limits, and
  `partition_clusters` (one triangle per cluster, target 12).
- The 100-vertex buffer comes from `srand(data[0])` + `rand() % 10`. `crand` ports glibc's TYPE_3 generator, so
  the inputs match the C++ harness exactly.
- C++ `assert` → `assert!`. Results that C++ ignores are ignored, `Err` included. Out-of-bounds accesses and
  other panics fail the run: they would be crashes in C++, and the library must not panic.

## Drivers
### Pure Rust mutation driver (`driver`, default)
No coverage feedback, runs on stable, no C/C++ involved:
- splitmix64 RNG;
- libFuzzer-style byte mutations: bit flips, interesting bytes, insert/erase/duplicate, truncate, splice;
- target-specific seeds. Codec seeds are valid index/sequence/vertex/meshlet encodings, so mutations get past
  the headers. Simplify/cluster seeds are a grid and random bytes.

Run it with `cargo run -p meshoptimizer-fuzz --release --bin fuzz -- <codec|simplify|cluster> [iterations] [seed]`.
A crashing input is saved to `fuzz-<target>-crash.bin`; rerun it with `--replay <file>`.

### libFuzzer (`fuzz/libfuzzer`, coverage-guided, dev only)
`libfuzzer-sys` builds LLVM's C++ libFuzzer. Approved by the human on the condition that it never reaches the
library, so:
- `fuzz/libfuzzer` is its own workspace (`[workspace]`, own `Cargo.lock`) and is in the root `exclude` list;
  `cargo build`/`test` of the workspace never builds it and `meshoptimizer` keeps zero dependencies.
- Each target is one `fuzz_target!` line calling the same `test_one_input`. The `#[no_mangle]` entry point the
  macro emits is the only unsafe-adjacent code, and it stays in that crate.

Needs nightly and `cargo install cargo-fuzz`:
`cargo +nightly fuzz run --fuzz-dir fuzz/libfuzzer <codec|simplify|cluster> -- -max_len=8192`
(AddressSanitizer on by default; `corpus/`, `artifacts/` are gitignored).

## Verification
- `fuzz/tests/fuzz.rs`: 4 seeds x (500 codec, 100 simplify, 200 cluster) inputs in `cargo test`, plus a glibc
  `rand()` sequence check.
- Long runs (release, 4 seeds each): 40M codec, 12M simplify, 6M cluster inputs without failures.
- libFuzzer + ASan, 20 min per target: 10.5M codec, 2.0M simplify, 0.36M cluster runs without failures
  (coverage 1097 / 1589 / 1384 edges).

## Makefile
`make fuzz` / `fuzz-<target>` (`ITERATIONS`, `SEED`), `make fuzz-replay TARGET= FILE=`,
`make libfuzzer` / `libfuzzer-<target>` (`FUZZ_TIME` seconds, `FUZZ_MAX_LEN`), `make fuzz-setup` (nightly + cargo-fuzz).
