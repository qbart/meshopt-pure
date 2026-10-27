# 06 — Overdraw optimizer & rasterizer (`src/overdrawoptimizer.cpp`, `src/rasterizer.cpp`)

| C | Rust |
|---|---|
| `meshopt_optimizeOverdraw` | `optimize_overdraw` (+ `_in_place`) |
| `meshopt_analyzeOverdraw` | `analyze_overdraw -> Result<OverdrawStatistics>` |
| `meshopt_analyzeCoverage` | `analyze_coverage -> Result<CoverageStatistics>` |

- `OverdrawBuffer` (1 MiB) is heap allocated as flat `[y][x][sign]` vectors.
- Fixed-point edge math uses `wrapping_*` to mirror C's unsigned-cast shifts without overflow panics.
- Lint `too_many_arguments` allowed crate-wide: signatures mirror C.
- Tests: `emptyMesh` now complete in `tests/optimizer.rs`; diff-tested bit-identical vs C++ (incl. coverage/overdraw stats).
