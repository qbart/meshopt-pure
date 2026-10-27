# Roadmap

Pure Rust port of meshoptimizer v1.3 (`ref/meshopt`, commit `f0a684f`). Specs/audit log: `docs/spec/`.

Legend: ✅ done · 🚧 in progress · ⬜ todo · ❓ needs human decision

## Core library (`src/*.cpp`)

| Module | Status | Tests |
|---|---|---|
| quantization | ✅ | 4/4 |
| indexanalyzer (analyzeVertexCache/Fetch) | ✅ | no C tests; 2 sanity |
| vfetchoptimizer | ✅ | no C tests; 1 sanity |
| indexgenerator | ✅ | 5/5 |
| vcacheoptimizer | ✅ | emptyMesh + diff-tested |
| overdrawoptimizer | ✅ | emptyMesh + diff-tested |
| rasterizer (analyzeOverdraw/Coverage) | ✅ | diff-tested |
| stripifier | ✅ | diff-tested + 1 sanity |
| indexcodec | ✅ | 21/21 + diff/fuzz-tested |
| vertexcodec | ✅ | 19/19 + diff/fuzz-tested |
| vertexfilter | ✅ | 15/15 + diff-tested |
| spatialorder | ✅ | diff-tested + 2 sanity |
| clusterizer | ✅ | 7/7 + diff-tested |
| meshletutils | ✅ | 3/3 + diff-tested |
| meshletcodec | ✅ | 4/4 + diff/fuzz-tested |
| partition | ✅ | 3/3 + diff-tested |
| simplifier | ✅ | 26/26 + diff-tested |
| tangentspace | ✅ | 4/4 + diff-tested |
| remesher | ✅ | diff-tested + 1 sanity |
| opacitymap | ✅ | 3/3 + diff-tested |
| allocator (`setAllocator`, `setAllocatorCache`) | ❓ | needs `unsafe` for raw allocators |
| u16 index adapters (C++ templates) | ⬜ | – |

**Status:** all `src/*.cpp` algorithms ported; 125 Rust tests (C `demo/tests.cpp` ports + sanity tests) passing.
Every module was additionally diff-tested bit-for-bit against the C++ reference (pirate.obj, grids, random soups,
fuzzed codec streams) using an out-of-repo harness (C code is not allowed in this repository).

Missing C tests: `customAllocator`, `customAllocatorCache` (blocked on the allocator decision below).

## Next steps

1. ❓ Allocator API: `meshopt_setAllocator`/`meshopt_setAllocatorCache` take raw `malloc`/`free`-style callbacks. A faithful port needs
   `unsafe` (raw memory) or the unstable `allocator_api`. Options: (a) skip — Rust users set `#[global_allocator]`; (b) expose
   observer callbacks (alloc/free notifications with sizes) so the C allocator-counting tests can be ported. Needs human approval.
2. ⬜ u16 index support for the remaining APIs (C++ template wrappers `meshopt_*<T>`); codecs already accept `IndexType`.
3. ⬜ gltfpack port (KTX via `qbart/ktx-rs`; BasisU/WebP/meshopt encoders must be pure Rust).

## Tools

| Component | Status |
|---|---|
| gltfpack (uses KTX via `qbart/ktx-rs`) | ⬜ |
