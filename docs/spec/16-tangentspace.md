# 16 — Tangent space & normals (`src/tangentspace.cpp`)

| C | Rust |
|---|---|
| `meshopt_TangentCompatible`, `meshopt_TangentZeroFallback` | `TANGENT_COMPATIBLE`, `TANGENT_ZERO_FALLBACK` (`u32` flags) |
| `meshopt_generateTangents` | `generate_tangents(result, indices: Option<&[u32]>, index_count, positions, vertex_count, stride, normals, normals_stride, uvs, uvs_stride, options) -> Result<()>` |
| `meshopt_generateNormals` (experimental) | `generate_normals(result, indices: Option<&[u32]>, index_count, positions, vertex_count, stride, crease_angle, smoothing) -> Result<()>` |

- Interleaved streams are passed as offset slices (`&vb[3..]`) with byte strides, like C pointers into one buffer.
- `VertexHasherF<PositionOnly>` → one hasher with optional normal/uv streams; uses shared `hash.rs` lookup (identical to `hashLookup4`).
- `optacos` keeps the C constant `3.1415926f` (not `PI`) for bit-exactness; `cosf` → `f32::cos`.
- Tests: `tangentsBasic`, `tangentDegenerate`, `normalsBasic`, `normalsDegenerate` → `tests/tangentspace.rs`.
- Diff-tested bit-identical (all option combos, crease/smoothing grid, indexed and unindexed).
