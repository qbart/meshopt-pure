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
| vcacheoptimizer | ✅ | emptyMesh (partial) + diff-tested |
| overdrawoptimizer | ⬜ | – |
| rasterizer (analyzeOverdraw/Coverage) | ⬜ | – |
| stripifier | ⬜ | – |
| indexcodec | ⬜ | – |
| vertexcodec | ⬜ | – |
| vertexfilter | ⬜ | – |
| spatialorder | ⬜ | – |
| clusterizer | ⬜ | – |
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
