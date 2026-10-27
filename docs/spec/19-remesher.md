# 19 — Voxel remesher (`src/remesher.cpp`, experimental)

| C | Rust |
|---|---|
| `meshopt_RemeshShell`, `meshopt_RemeshSolve` | `REMESH_SHELL`, `REMESH_SOLVE` |
| `meshopt_RemeshInternalDebug` | `REMESH_INTERNAL_DEBUG` (`#[doc(hidden)]`) |
| `meshopt_remesh` | `remesh(destination: Option<&mut [f32]>, max_triangle_count, indices, index_count, positions, vertex_count, stride, resolution, options) -> Result<usize>` |

- Static-initialized triangle tables (`buildRemeshTables`) → lazily built once via `std::sync::OnceLock` (safe, thread-safe).
- Grid/rowmap/voxel arrays live in a `Grid` struct; `voxel_index` replaces the C `rowmap[row] + grid[idx] - 1` lookups.
- `NULL` destination → `None` (counting mode, `max_triangle_count` must be 0).
- No C tests; sanity test in `tests/remesher.rs`; diff-tested bit-identical (resolutions 4/16/64/129 × all options, full and truncated output).
