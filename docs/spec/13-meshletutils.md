# 13 — Meshlet utilities (`src/meshletutils.cpp`)

| C | Rust |
|---|---|
| `meshopt_Bounds` | `Bounds { center, radius, cone_apex, cone_axis, cone_cutoff, cone_axis_s8: [i8; 3], cone_cutoff_s8: i8 }` |
| `meshopt_computeClusterBounds` | `compute_cluster_bounds(indices, index_count, positions, vertex_count, stride) -> Result<Bounds>` |
| `meshopt_computeMeshletBounds` | `compute_meshlet_bounds(meshlet_vertices, meshlet_triangles, triangle_count, positions, vertex_count, stride) -> Result<Bounds>` |
| `meshopt_computeSphereBounds` | `compute_sphere_bounds(positions, count, stride, radii: Option<&[f32]>, radii_stride) -> Result<Bounds>` |
| `meshopt_optimizeMeshlet` | `optimize_meshlet(meshlet_vertices, meshlet_triangles, triangle_count, vertex_count) -> Result<()>` |
| `meshopt_optimizeMeshletLevel` | `optimize_meshlet_level(meshlet_vertices, vertex_count, meshlet_triangles, triangle_count, level) -> Result<()>` |
| `meshopt_extractMeshletIndices` | `extract_meshlet_indices(vertices, triangles, indices, index_count) -> Result<usize>` |

- `computeBoundingSphere<Axes>` → runtime axis count with a `Spheres` view (points/stride, optional radii, optional indices).
- `extract_meshlet_indices` reports >256 unique vertices or a short `vertices` slice as `Err` mid-way (C asserts).
- Tests: `clusterBoundsDegenerate`, `sphereBounds`, `extractMeshlet`, `meshletsMax` → `tests/meshletutils.rs`.
- Diff-tested bit-identical: bounds, extraction and optimizer levels 0/1/3/9 on every meshlet of every test mesh.
