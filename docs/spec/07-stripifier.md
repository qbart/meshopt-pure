# 07 — Stripifier (`src/stripifier.cpp`)

| C | Rust |
|---|---|
| `meshopt_stripify` | `stripify(destination, indices, index_count, vertex_count, restart_index) -> Result<usize>` |
| `meshopt_stripifyBound` | `stripify_bound(index_count) -> Result<usize>` |
| `meshopt_unstripify` | `unstripify(destination, indices, index_count, restart_index) -> Result<usize>` |
| `meshopt_unstripifyBound` | `unstripify_bound(index_count) -> Result<usize>` |

- `destination != indices` is enforced by the borrow checker.
- `destination` must hold the bound (validated up front).
- No C tests; sanity roundtrip in `tests/optimizer.rs`, diff-tested bit-identical vs C++ (both restart modes).
