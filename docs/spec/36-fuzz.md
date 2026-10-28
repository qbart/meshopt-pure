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

## Driver (instead of libFuzzer)
`cargo-fuzz`/`libfuzzer-sys` builds LLVM's C++ libFuzzer, so it needs human approval (AGENTS.md); not used.
`driver` is a pure Rust mutation fuzzer without coverage feedback:
- splitmix64 RNG;
- libFuzzer-style byte mutations: bit flips, interesting bytes, insert/erase/duplicate, truncate, splice;
- target-specific seeds. Codec seeds are valid index/sequence/vertex/meshlet encodings, so mutations get past
  the headers. Simplify/cluster seeds are a grid and random bytes.

Run it with `cargo run -p meshoptimizer-fuzz --release --bin fuzz -- <codec|simplify|cluster> [iterations] [seed]`.
A crashing input is saved to `fuzz-<target>-crash.bin`; rerun it with `--replay <file>`.
The harnesses keep the libFuzzer signature (`fn(&[u8])`), so they can be wired to `libfuzzer-sys` if it is approved.

## Verification
- `fuzz/tests/fuzz.rs`: 4 seeds x (500 codec, 100 simplify, 200 cluster) inputs in `cargo test`, plus a glibc
  `rand()` sequence check.
- Long runs (release, 4 seeds each): 40M codec, 12M simplify, 6M cluster inputs without failures.
