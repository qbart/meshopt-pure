# 18 — Simplifier (`src/simplifier.cpp`)

| C | Rust |
|---|---|
| `meshopt_Simplify*` options | `SIMPLIFY_LOCK_BORDER`, `SIMPLIFY_SPARSE`, `SIMPLIFY_ERROR_ABSOLUTE`, `SIMPLIFY_PRUNE`, `SIMPLIFY_REGULARIZE`, `SIMPLIFY_PERMISSIVE`, `SIMPLIFY_REGULARIZE_LIGHT`, `SIMPLIFY_PRESERVE_FOLDS`, `SIMPLIFY_ERROR_CLAMPED` (`u32`) |
| `meshopt_SimplifyVertex_*` | `SIMPLIFY_VERTEX_LOCK`, `SIMPLIFY_VERTEX_PROTECT`, `SIMPLIFY_VERTEX_PRIORITY` (`u8`) |
| `meshopt_simplify` | `simplify(destination, indices, index_count, positions, vertex_count, stride, target_index_count, target_error, options, result_error: Option<&mut f32>) -> Result<usize>` |
| `meshopt_simplifyWithAttributes` | `simplify_with_attributes(..., attributes, attributes_stride, weights, attribute_count, vertex_lock: Option<&[u8]>, ...)` |
| `meshopt_simplifyWithUpdate` | `simplify_with_update(indices, ..., positions: &mut [f32], ..., attributes: &mut [f32], ...)` |
| (attributes aliasing positions) | `simplify_with_update_interleaved(indices, ..., vertex_data: &mut [f32], ..., vertex_attributes_offset, ...)` |
| `meshopt_simplifySloppy` | `simplify_sloppy(..., vertex_lock: Option<&[u8]>, target_index_count, target_error, result_error)` |
| `meshopt_simplifyPrune` | `simplify_prune(destination, indices, index_count, positions, vertex_count, stride, target_error)` |
| `meshopt_simplifyPoints` | `simplify_points(destination, positions, vertex_count, stride, colors: Option<&[f32]>, colors_stride, color_weight, target_vertex_count)` |
| `meshopt_simplifyScale` | `simplify_scale(positions, vertex_count, stride) -> Result<f32>` |
| `meshopt_SimplifyInternalDebug` | `SIMPLIFY_INTERNAL_DEBUG` (`#[doc(hidden)]`, development only) |

- `meshopt_simplifyEdge` → private `simplify_edge` that returns solve data; `finalizeVertices` runs after the read-only borrow ends,
  which is how the destructive update works without aliasing.
- `Collapse` union (`bidi`/`error`/`errorui`) → `value: u32` holding either the flag or f32 bits.
- The per-file hash helpers (`hashLookup2`, `hashBuckets2`) reuse `src/hash.rs`.
- `simplify_sloppy` validates `destination` lazily (bounded by the triangle count it will write), matching C tests that pass `NULL` for empty results.
- `TRACE` statistics are not ported.
- Tests: all 26 simplifier test invocations → `tests/simplifier.rs`.
- Diff-tested bit-identical: 14 option sets × 3 targets (simplify/attributes/update incl. vertex data), sloppy with/without locks, prune, points (with colors), scale.
