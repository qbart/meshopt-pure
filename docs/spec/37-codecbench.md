# 37 — codecbench (`tools/codecbench.cpp`)

Workspace crate `tools/` (`meshoptimizer-tools`, not published), binary `codecbench [-v] [-l] [files...]`
(`make codecbench ARGS=...`).

## Port
- Synthetic mode (no files): 1001² grid of 32-byte vertices (murmur3 k-bit words), 2M triangles; best of 50 decodes
  for vertex/index codecs, meshlet codec with and without vertex references (64/96, level-3 meshlet optimization,
  vertex fetch reorder), and filters oct8/oct12/quat12/col8/col12/exp on zeroes. Same table output; `-v` per attempt,
  `-l` loops 100 times.
- File mode: `v<count>_s<stride>_*` (decoded or raw stream, re-encoded) and `i<count>_*` (encoded indices,
  stride 2 if `count <= 0xffff`); `sscanf` patterns parsed by hand.
- `lib`: setup (`make_grid`, `prepare_codecs`, `prepare_meshlets`, `read_file`) is split from timing loops for tests.
- Divergences: C++ `assert`s and decode failures → `Err` printed to stderr, exit 1 (C++: assert/abort). Unknown
  file names error instead of asserting. Timing uses `Instant` (C++: `clock_gettime(CLOCK_MONOTONIC)`).

## Verification
- `tools/tests/codecbench.rs`: FNV-1a 64 of optimized vb/ib, encoded vertex/index streams and packed meshlets for
  N = 1, 7, 64, 100, matching an out-of-repo C++ build with `N` overridden; file mode roundtrip; smoke runs.
- Out of repo: `-v` setup lines (encoded sizes, meshlet counts, packed sizes) identical to C++ at N = 1000; file
  mode ratio identical on generated float streams.

## Performance (Ryzen AI 9 HX 370, GB/s, best of 50)
| | vtx | idx | mlet | mletΔ | oct8 | oct12 | quat12 | col8 | col12 | exp |
|---|---|---|---|---|---|---|---|---|---|---|
| C++ SIMD (`-O3`) | 6.56 | 6.73 | 8.20 | 7.33 | 4.44 | 7.88 | 9.57 | 5.70 | 9.59 | 27.66 |
| C++ `MESHOPTIMIZER_NO_SIMD` | 1.51 | 6.71 | 2.05 | 1.43 | 2.30 | 3.64 | 3.41 | 2.65 | 4.23 | 27.23 |
| Rust scalar, initial port | 1.03 | 6.43 | 1.81 | 1.38 | 0.82 | 1.65 | 1.75 | 1.08 | 1.76 | 5.00 |
| Rust scalar, tuned | 1.61 | 6.43 | 1.81 | 1.38 | 1.16 | 1.73 | 2.19 | 1.15 | 2.74 | 27.50 |
| Rust scalar (`make codecbench-scalar`) | 1.64 | 6.43 | 1.80 | 1.38 | 1.16 | 1.76 | 2.21 | 1.19 | 2.18 | 27.67 |
| Rust SIMD (`make codecbench`, spec 38) | 6.08 | 6.42 | 7.38 | 5.37 | 4.46 | 7.99 | 9.74 | 5.67 | 9.63 | 28.06 |

Scalar tuning (safe code only):
- vertex codec: byte groups specialized per bit width (const generics, 24-byte window), C-like delta loop with
  const width, decode scratch buffers allocated once per call. Now 1.07x C++ scalar.
- filters: elements as `[u8; 4]`/`[u8; 8]` chunks with constant component offsets. `exp` vectorizes (parity).
  oct/quat/color stay at 0.4–0.65x C++ scalar: Rust's saturating `f32 as i32` blocks the vectorization GCC applies to
  these loops (non-saturating casts need `unsafe`; measured 1.6–2.8x faster), so they are left to the SIMD paths.
  col12 dropped to 2.18 when color adopted C++'s x86 out-of-range conversion (spec 38).

SIMD (spec 38) vs C++ SIMD: filters at parity (0.99–1.10x), vertex decode 0.92x, meshlets 0.90x, meshlets without
vertex references 0.70x. The meshlet triangle loop is sensitive to code layout (5.2–5.7 GB/s across equivalent variants).
