# 40 — demo (`demo/main.cpp`, `demo/nanite.cpp`, `extern/sdefl.h`)

Workspace crate `demo/` (`meshoptimizer-demo`, not published), binary `demo [-d|-n] files...`.

## Port
- `demo.rs` = `main.cpp`: OBJ loading (gltfpack's `fast_obj` port) and indexing, then every statistics section in the
  C++ order: optimize, stripify, meshlets (scan/regular/uniform/flex/spatial) + bounds, shadow/filter/tessellation/
  provoking, index/sequence/vertex codecs, meshlet codec, the DXR cluster encoding, all simplifiers, spatial sorts,
  fuzzy reindex, coverage, tangents, normals, remesh, deinterleaved pipeline. `-d` runs remesh only, `-n` runs nanite.
- `nanite.rs` = `nanite.cpp`: clusterlod DAG build and per-level statistics (components, boundary, SAH overhead with
  `CLRT`), BVH cut, `DUMP` cuts as OBJ on stderr.
- `sdefl.rs` = `extern/sdefl.h`: deflate compressor for the post-deflate sizes, byte-identical to C.
- `Vertex` keeps the C++ layout (8 floats, 32 bytes). Byte-oriented APIs get little-endian copies (`bytes`/`floats`),
  and the packed formats are built byte by byte with the C++ field layout.
- Printing follows C `printf`: `%f` via gltfpack's `format_f` (glibc `nan`/`-nan`), and float expressions are
  evaluated in `f32` before promotion, as in C.

## Divergences
- No arguments: C++ runs `demo/tests.cpp` (`runTests`). Those tests are ported 1:1 to `tests/` of the root crate, so
  the binary prints usage instead, and doesn't rerun them after processing files.
- `meshopt_setAllocatorCache` in nanite is skipped (allocator API not ported, spec 00).
- In-place C++ calls (same source and destination) use copies: optimize_vertex_fetch, remap_vertex_buffer, vertex
  cache/overdraw. `simplifyWithUpdate` on aliased position/attribute pointers uses `simplify_with_update_interleaved`.
- C++ `assert`s → `assert!`; library errors stop processing with `Error processing <path>: ...` (C++ would crash or
  assert).
- sdefl: all writes are bounds-checked, and the functions return 0 instead of overflowing. This covers a too-small
  output buffer, a level outside 0..=8, and input longer than `i32::MAX`. The sequence buffer grows instead of
  overflowing. The C `sdefl_bound` is too small for multi-megabyte incompressible input (about 25 bytes of overhead
  per 256 KB block, e.g. 5000486 bytes written against a bound of 5000406). There C overflows the heap, and the port
  returns 0. Mesh data in the demo compresses far below the bound.

## Verification
- Out of repo, against the C++ demo (`main.cpp` + `tests.cpp` + `nanite.cpp`, same sources): stdout with timings
  masked, stderr and exit codes identical for 18 runs. Meshes: pirate.obj, a 40k-triangle quad torus, a
  positions-only noisy sphere, a hexagon n-gon grid with degenerate faces, and a 200k-triangle terrain. Modes:
  default, `-n`, `-d`, `CLRT`, `DUMP=1`, `DUMP=-1` (1.4 MB of OBJ on stderr), a missing file, and multiple files.
- `demo/tests/demo.rs`: the binary vs raw C++ output (`tests/data/*.txt`, timings masked) for pirate.obj and a small
  n-gon mesh (`tests/data/hexes.obj`) in default, `-n`, `-n` + `CLRT` and `-d` modes.
- sdefl, out of repo: byte-identical to C at all 9 levels, for `sdeflate` and `zsdeflate`. Inputs: 366 files
  (window/block edges, up to 9 MB random, runs, alphabets, text, pirate.obj, gltfpack test data) and 400 fuzz-style
  files. Every output inflates back to its input (miniz_oxide, Python zlib). The C oracle runs clean under ASan/UBSan.
- `demo/tests/sdefl.rs`: sdefl golden sizes/hashes from the C library for 10 generated inputs x 9 levels x both
  functions, plus `sdefl_bound` values and the error returns.
