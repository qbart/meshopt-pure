# 12 — Clusterizer (`src/clusterizer.cpp`)

| C | Rust |
|---|---|
| `meshopt_Meshlet` | `Meshlet { vertex_offset, triangle_offset, vertex_count, triangle_count }` |
| `meshopt_buildMeshletsBound` | `build_meshlets_bound(index_count, max_vertices, max_triangles) -> Result<usize>` |
| `meshopt_buildMeshlets` | `build_meshlets(meshlets, meshlet_vertices, meshlet_triangles, indices, index_count, positions, vertex_count, stride, max_vertices, max_triangles, cone_weight) -> Result<usize>` |
| `meshopt_buildMeshletsFlex` | `build_meshlets_flex(..., max_vertices, min_triangles, max_triangles, cone_weight, split_factor)` |
| `meshopt_buildMeshletsScan` | `build_meshlets_scan(..., indices, index_count, vertex_count, max_vertices, max_triangles)` |
| `meshopt_buildMeshletsSpatial` | `build_meshlets_spatial(..., max_vertices, min_triangles, max_triangles, fill_weight)` |

- Output buffers are written with bounds checks: C callers (and the C tests) often pass buffers smaller than the
  documented worst case, so overflow surfaces as `Err(BufferTooSmall(..))` instead of being rejected up front.
- `KDNode` union/bitfields → `KdNode { value: u32 (f32 bits | index), axis, children }`.
- SSE/NEON `boxMerge` → scalar variant (same evaluation order, identical for finite input).
- `clearUsed` sparse optimization is replaced by a full `-1` fill (identical results).
- `MESHOPTIMIZER_CLUSTERIZER_INDEXLIMIT` compile option is not ported.
- Tests: `meshletsEmpty/Dense/Sparse/Flex/Spatial/SpatialDeep` → `tests/clusterizer.rs` (`meshletsMax` needs meshletutils).
- Diff-tested bit-identical for 9 builder configurations across all meshes (incl. pirate.obj).
