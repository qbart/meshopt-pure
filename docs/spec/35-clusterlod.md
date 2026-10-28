# 35 — clusterlod (`demo/clusterlod.h`)

Workspace crate `clusterlod/`: Nanite-style cluster LOD DAG and per-level BVH built on the `meshoptimizer` crate.
Output must be bit-identical to the C++ header (ref `f0a684f`).

## API mapping
| C++ | Rust |
|---|---|
| `clodConfig`, `clodMesh`, `clodBounds`, `clodCluster`, `clodGroup`, `clodNode` | `ClodConfig`, `ClodMesh<'a>`, `ClodBounds`, `ClodCluster<'a>`, `ClodGroup`, `ClodNode` (same fields) |
| `clodDefaultConfig`, `clodDefaultConfigRT` | `clod_default_config`, `clod_default_config_rt` → `Result` (assert `4..=256`) |
| `clodBuild(config, mesh, context, callback)` / template overload | `clod_build(config, mesh, FnMut(ClodGroup, &[ClodCluster], &[f32]) -> i32) -> Result<()>` |
| `clodLocalIndices` | `clod_local_indices` (= `extract_meshlet_indices`) |
| `clodBuildHierarchyBound`, `clodBuildHierarchy` | `clod_build_hierarchy_bound`, `clod_build_hierarchy` → `Result<usize>` |

- Pointers become slices; `NULL` attributes/locks/weights → `None`. `ClodCluster::indices` borrows the build's
  internal buffer for the duration of the callback (C++: valid until the callback returns).
- A `NULL` callback (C++: refined = -1) is a closure returning `-1`.

## Divergences
- `simplify_dilate_borders`: C++ writes dilated positions through the `const float*` input. The port keeps the
  input immutable and dilates a working copy, which every later step uses (clusterize, bounds, simplify), and passes
  it to the callback as the third argument. Callers copy positions from it, as the C++ comment requires anyway.
- C++ asserts → `Err` before any callback: attribute stride and mask checks, positions/indices/lock/attribute
  buffer sizes, indices `< vertex_count`. Hierarchy: `node_width >= 2` (C++ divides by zero / loops forever),
  `nodes.len() >= bound`, every level `0..level_count` must have a group (C++ reads `row[0]` of an empty row).
- `boundsMerge` / `spatialClusterPoints` strided struct reads are rebuilt as packed `[x,y,z,r]` / `[x,y,z]` arrays
  (same values, same order); the sloppy fallback uses a packed 12-byte position subset instead of 16-byte
  `SloppyVertex` (the stride doesn't affect the result).

## Verification
- `clusterlod/tests/clusterlod.rs`: 26 golden hashes (pirate.obj and a seamed grid, both serialized as
  `tests/data/*.bin` by the C++ demo loader, x 13 configurations covering every config flag, no attributes, vertex
  locks, dilation), hashing all callback outputs, dilated positions, `clod_local_indices` and hierarchies of width
  8/4/2. Plus sanity tests (dilation moves vertices, argument errors).
- Out of repo: the same 13 configurations on a 90k-triangle noisy torus with holes (up to 343 groups) and a 12k-triangle
  variant are byte-identical to the C++ header.
