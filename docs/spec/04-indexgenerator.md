# 04 — Index generator (`src/indexgenerator.cpp`)

| C | Rust |
|---|---|
| `meshopt_Stream` | `Stream<'a> { data: &[u8], size, stride }` |
| `meshopt_generateVertexRemap` / `Multi` / `Custom` | `generate_vertex_remap` / `_multi` / `_custom` |
| `meshopt_remapVertexBuffer` | `remap_vertex_buffer` + `remap_vertex_buffer_in_place` |
| `meshopt_remapIndexBuffer` | `remap_index_buffer` + `remap_index_buffer_in_place` |
| `meshopt_filterIndexBuffer` / `Multi` | `filter_index_buffer` / `_multi` |
| `meshopt_generateShadowIndexBuffer` / `Multi` | `generate_shadow_index_buffer` / `_multi` |
| `meshopt_generatePositionRemap` | `generate_position_remap` |
| `meshopt_generateAdjacencyIndexBuffer` | `generate_adjacency_index_buffer` |
| `meshopt_generateTessellationIndexBuffer` | `generate_tessellation_index_buffer` |
| `meshopt_generateProvokingIndexBuffer` | `generate_provoking_index_buffer` |

- Hash table (`hashBuckets`/`hashLookup`, MurmurHash2) lives in `src/hash.rs` for reuse; lookup returns a slot index.
- Custom callback + `void* context` → `Option<&mut dyn FnMut(u32, u32) -> bool>` (context captured by closure).
- `NULL` indices → `None`; requires `index_count == vertex_count` (C assert).
- In-place C modes are separate `_in_place` functions.
- Tests: `remapCustom`, `filterTriangles`, `adjacency`, `tessellation`, `provoking` → `tests/indexgenerator.rs`.
