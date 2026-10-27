# 11 — Spatial order (`src/spatialorder.cpp`)

| C | Rust |
|---|---|
| `meshopt_spatialSortRemap` | `spatial_sort_remap(destination, vertex_positions, vertex_count, stride) -> Result<()>` |
| `meshopt_spatialSortTriangles` | `spatial_sort_triangles` (+ `_in_place`) |
| `meshopt_spatialClusterPoints` | `spatial_cluster_points(destination, vertex_positions, vertex_count, stride, cluster_size) -> Result<()>` |

- Scratch memory (`order`/`sides`/`keyk` aliasing one allocation in C) uses separate `Vec`s.
- `splitPoints` recursion uses `split_at_mut` on the three axis orders.
- No C tests; sanity tests in `tests/spatialorder.rs`; diff-tested bit-identical (cluster sizes 1/3/64/256).
