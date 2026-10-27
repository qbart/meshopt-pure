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
| clusterizer | ✅ | 6/7 (meshletsMax pending meshletutils) + diff-tested |
| meshletutils | ⬜ | – |
| meshletcodec | ⬜ | – |
| partition | ⬜ | – |
| simplifier | ⬜ | – |
| tangentspace | ⬜ | – |
| remesher | ⬜ | – |
| opacitymap | ⬜ | – |
| allocator (`setAllocator`, `setAllocatorCache`) | ❓ | needs `unsafe` for raw allocators |
| u16 index adapters (C++ templates) | ⬜ | – |

## Tools

| Component | Status |
|---|---|
| gltfpack (uses KTX via `qbart/ktx-rs`) | ⬜ |
