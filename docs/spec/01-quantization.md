# 01 — Quantization (`src/quantization.cpp`)

| C | Rust | Notes |
|---|---|---|
| `meshopt_quantizeHalf` | `quantize_half(f32) -> u16` | infallible |
| `meshopt_quantizeFloat` | `quantize_float(f32, i32) -> Result<f32>` | `N ∉ 0..=23` → `InvalidArgument` |
| `meshopt_dequantizeHalf` | `dequantize_half(u16) -> f32` | infallible |
| `meshopt_computePositionExponent` | `compute_position_exponent(&[f32;3], &[f32;3], i32, i32) -> Result<i32>` | asserts → `InvalidArgument` |

`frexpf`/`ldexpf` are reimplemented in `src/math.rs` (bit-exact, denormal aware).

Tests: `quantizeFloat`, `quantizeHalf`, `dequantizeHalf`, `computePositionExponent` → `tests/quantization.rs`.

Header inline helpers: `meshopt_quantizeUnorm` → `quantize_unorm(f32, i32) -> i32`, `meshopt_quantizeSnorm` → `quantize_snorm(f32, i32) -> i32`
(infallible; shifts use `wrapping_*` so out-of-range `N` cannot panic).
