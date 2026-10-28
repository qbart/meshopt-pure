# 20 — Index type adapters (`meshopt_IndexAdapter`, C++ `template <typename T>` wrappers)

C++ overloads every index-buffer API with `template <typename T>`; for `T != unsigned int` the indices are widened into a
temporary `unsigned int` buffer, the C function runs, and outputs are narrowed back with `T(data[i])`.

## Mapping
- Every templated C++ API becomes generic `fn foo<T: IndexType>(…)` with the same name; `T` is `u16` or `u32`.
- Index slices typed `T` exactly where the C++ wrapper uses `T*`; buffers that stay `unsigned int*` in C++
  (remap tables, `reorder`, `meshlet_vertices`, `cluster_index_counts`) stay `u32`.
- `stripify`/`unstripify` take `restart_index: T` (C++: `unsigned(restart_index)` on the way in).
- Non-templated C APIs stay `u32`: `generate_vertex_remap_custom`, `extract_meshlet_indices`, `opacity_map_measure`,
  `compute_meshlet_bounds`, `optimize_meshlet*`, `spatial_sort_remap`, `simplify_points`.
- Rust-only `_in_place` variants are generic as well.

## Adapter (`src/index.rs`)
- `index::input` → `Cow<[u32]>`: borrowed for `u32` (C++ `IndexAdapter<T, true>`, zero copy), widened copy for `u16`.
- `index::output(dst, f)`: `u32` runs `f` directly; `u16` runs `f` on a widened copy of `dst` and narrows it back
  (truncating). The copy is seeded from `dst` so entries the C function doesn't write keep their value.
- Length validation runs inside the `u32` implementation on equal-length buffers, so errors match the `u32` path.
- Zero-copy specialisation uses sealed-trait methods (`as_u32`), no `unsafe`.

## Notes
- Calls whose index type is not inferable (`&mut []`, `None`) need a turbofish, e.g. `optimize_vertex_cache::<u32>`.

## Verification
- `tests/indextype.rs`: every generic API gives identical results for `u16` and `u32` inputs on a scrambled grid.
