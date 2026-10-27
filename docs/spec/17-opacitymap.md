# 17 — Opacity micromaps (`src/opacitymap.cpp`)

| C | Rust |
|---|---|
| `meshopt_opacityMapMeasure` | `opacity_map_measure(levels, sources, omm_indices: &mut [i32], indices, index_count, uvs, vertex_count, stride, width, height, max_level, target_edge) -> Result<usize>` |
| `meshopt_opacityMapEntrySize` | `opacity_map_entry_size(level, states) -> Result<usize>` |
| `meshopt_opacityMapRasterize` | `opacity_map_rasterize(result, level, states, uv0: &[f32; 2], uv1, uv2, texture_data: &[u8], stride, pitch, width, height) -> Result<()>` |
| `meshopt_opacityMapCompact` | `opacity_map_compact(data, data_size, levels, offsets, omm_count, omm_indices, triangle_count, states) -> Result<usize>` |

- `rasterizeOpacity*<States>` templates → runtime `states` parameter.
- Texture slice is validated to cover the last addressed texel (C trusts the caller).
- Speculative-write hashing (measure/compact) kept: data is written, then looked up with a short-lived hasher borrow.
- `log2f` → `f32::log2`.
- Tests: `opacityMap`, `opacityMapRasterize0`, `opacityMapSpecial` → `tests/opacitymap.rs`.
- Diff-tested bit-identical (2/4 states, fixed and adaptive levels, wrapped UVs, RGBA texture with stride 4).
