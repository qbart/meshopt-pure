# 15 — Meshlet codec (`src/meshletcodec.cpp`)

| C / C++ template | Rust |
|---|---|
| `meshopt_encodeMeshlet` | `encode_meshlet(buffer, buffer_size, vertices, vertex_count, triangles, triangle_count) -> Result<usize>` |
| `meshopt_encodeMeshletBound` | `encode_meshlet_bound(max_vertices, max_triangles) -> usize` |
| `meshopt_decodeMeshlet` (+ `<V, T>`) | `decode_meshlet<V: IndexType, T: MeshletTriangleType>(vertices, vertex_count, triangles, triangle_count, buffer, buffer_size) -> Result<()>` |
| `meshopt_decodeMeshletRaw` | `decode_meshlet_raw(vertices: &mut [u32], vertex_count, triangles: &mut [u32], triangle_count, buffer, buffer_size)` |

- `vertex_size` (2/4) is `V::SIZE` (`u16`/`u32`); `triangle_size` 3/4 is `T = u8` (3 bytes per triangle) / `T = u32` (packed).
- Output slices need exactly `vertex_count` / `triangle_count * T::PER_TRIANGLE` elements (the C SIMD alignment padding is not required).
- Scalar decoder only. On corrupted input the C SIMD decoder may return `-3` where C scalar (and this port) return `-2`.
- Encoder returns `Ok(0)` when the buffer is too small (C semantics).
- Tests: `encodeMeshletBound`, `decodeMeshletSafety`, `decodeMeshletBasic`, `decodeMeshletTypical` → `tests/meshletcodec.rs`.
- Diff-tested: encodings byte-identical; ~1000 corrupted streams decode identically to the C scalar build.
