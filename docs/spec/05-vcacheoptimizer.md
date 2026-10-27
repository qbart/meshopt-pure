# 05 — Vertex cache optimizer (`src/vcacheoptimizer.cpp`)

| C | Rust |
|---|---|
| `meshopt_optimizeVertexCache` | `optimize_vertex_cache` (+ `_in_place`) |
| `meshopt_optimizeVertexCacheStrip` | `optimize_vertex_cache_strip` (+ `_in_place`) |
| `meshopt_optimizeVertexCacheFifo` | `optimize_vertex_cache_fifo` (+ `_in_place`) |

- `TriangleAdjacency`/`build_triangle_adjacency` are `pub(crate)` for reuse (overdraw, stripifier, ...).
- Empty mesh guard runs before index validation, as in C.
- Tests: `emptyMesh` (vcache part) in `tests/optimizer.rs`; no other C tests.

## Verification
Differential oracle (C++ reference vs Rust, kept outside the repo): bit-identical output for pirate.obj,
regular/shuffled grids and random index soups — also covers indexanalyzer, indexgenerator, vfetchoptimizer.
