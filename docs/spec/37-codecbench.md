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
| Rust (release, scalar) | 1.03 | 6.43 | 1.81 | 1.38 | 0.82 | 1.65 | 1.75 | 1.08 | 1.76 | 5.00 |

Index decoding is at parity. The Rust scalar code trails even C++ scalar on vertex decode (0.7x) and filters
(0.2–0.5x), so there is headroom before SIMD.
