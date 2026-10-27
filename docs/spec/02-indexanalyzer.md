# 02 — Index analyzer (`src/indexanalyzer.cpp`)

| C | Rust |
|---|---|
| `meshopt_analyzeVertexCache` | `analyze_vertex_cache(indices, index_count, vertex_count, cache_size, warp_size, primgroup_size) -> Result<VertexCacheStatistics>` |
| `meshopt_analyzeVertexFetch` | `analyze_vertex_fetch(indices, index_count, vertex_count, vertex_size) -> Result<VertexFetchStatistics>` |

- Asserts (`index_count % 3`, `cache_size >= 3`, `warp_size`, `vertex_size` 1..=256, index range) → `Err`.
- Counter arithmetic uses `wrapping_*` to mirror `unsigned int` semantics.
- No 1:1 C tests exist; sanity tests in `tests/analyzer.rs`.
