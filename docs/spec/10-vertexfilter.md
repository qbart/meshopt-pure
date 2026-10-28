# 10 — Vertex filters (`src/vertexfilter.cpp`)

| C | Rust |
|---|---|
| `meshopt_decodeFilterOct/Quat/Exp/Color` | `decode_filter_oct/quat/exp/color(buffer: &mut [u8], count, stride) -> Result<()>` |
| `meshopt_encodeFilterOct/Quat/Color` | `encode_filter_oct/quat/color(destination: &mut [u8], count, stride, bits, data: &[f32]) -> Result<()>` |
| `meshopt_encodeFilterExp` | `encode_filter_exp(destination, count, stride, bits, data, EncodeExpMode) -> Result<()>` |
| (`destination == data`) | `encode_filter_exp_in_place(data: &mut [f32], count, stride, bits, mode)` |
| `meshopt_EncodeExpMode` | `EncodeExpMode { Separate, SharedVector, SharedComponent, Clamped }` |

- Buffers are little-endian byte slices (C `void*`); 8/16-bit lanes chosen by stride.
- Scalar paths and the SSE2 decoders (spec 38). The C SIMD decoders round slightly differently from the C scalar ones; each port path matches the corresponding C build bit-for-bit (`tests/vertexfilter_reference.rs`).
- Known divergence on invalid input only: color decode with alpha == 0 converts ±inf to int (UB in C; x86 yields 0, Rust/ARM saturate).
- Tests: all 15 filter tests → `tests/vertexfilter.rs`; diff-tested (all bit widths/modes, random raw buffers).
