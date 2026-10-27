# 03 — Vertex fetch optimizer (`src/vfetchoptimizer.cpp`)

| C | Rust |
|---|---|
| `meshopt_optimizeVertexFetchRemap` | `optimize_vertex_fetch_remap(destination, indices, index_count, vertex_count) -> Result<usize>` |
| `meshopt_optimizeVertexFetch` | `optimize_vertex_fetch(destination, indices, index_count, vertices, vertex_count, vertex_size) -> Result<usize>` |
| (`destination == vertices`) | `optimize_vertex_fetch_in_place(vertices, indices, index_count, vertex_count, vertex_size) -> Result<usize>` |

- Rust forbids aliasing `&mut`/`&`, so the C in-place mode is a separate function (copies input like C does).
- Vertex data is `&[u8]`; all slices validated before writing.
- Only allocator-counting C tests use it (pending allocator decision); sanity tests in `tests/analyzer.rs`.
