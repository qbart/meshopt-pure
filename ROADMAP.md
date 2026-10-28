# Roadmap

Pure Rust port of meshoptimizer v1.3 (`ref/meshopt`, commit `f0a684f`). Specs/audit log: `docs/spec/`.

Legend: ✅ done · 🚧 in progress · ⬜ todo · ❓ needs human decision · ➖ intentionally not ported · ⏸ deliberately deferred

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
| allocator (`setAllocator`, `setAllocatorCache`) | ➖ not ported | use `#[global_allocator]`; tests N/A |
| u16 index adapters (C++ templates) | ✅ | 9 u16-vs-u32 equivalence tests |

**Status:** all `src/*.cpp` algorithms and C++ `template <typename T>` index wrappers ported; 134 Rust tests (C `demo/tests.cpp` ports + sanity tests) passing.
Every module was additionally diff-tested bit-for-bit against the C++ reference (pirate.obj, grids, random soups,
fuzzed codec streams) using an out-of-repo harness (C code is not allowed in this repository).

Not applicable C tests: `customAllocator`, `customAllocatorCache` (allocator API intentionally not ported, see `docs/spec/00-conventions.md`).

## Next steps

1. ✅ gltfpack port: done; WebP (`-tw`) deliberately deferred (see below).
2. ✅ clusterlod (`demo/clusterlod.h`): done (spec 35).
3. ✅ fuzz harnesses (`tools/codecfuzz`, `simplifyfuzz`, `clusterfuzz`) with a pure Rust mutation driver, plus
   coverage-guided libFuzzer targets in a separate dev-only crate that never reaches the library (spec 36).
4. ✅ codec tools: `codecbench` (spec 37), `codectest` (spec 39).
5. ✅ SIMD decoders (x86-64 SSE2/SSSE3/SSE4.1, runtime detection, default `simd` feature; spec 38).
   ⬜ NEON and Wasm SIMD paths (need ARM/Wasm test environments); AVX-512 vertex path skipped (no gain measured).
6. ⬜ demo (`demo/main.cpp`, `nanite.cpp` statistics) as examples. Not planned: `vcachetuner`/`vcachetester`, `js/` wasm wrappers, `wasmstubs`.

## Tools

### codectest (`tools/codectest.cpp`) — spec `docs/spec/39-codectest.md`

| Component | Status | Verification |
|---|---|---|
| encode/decode filter, `-test`/`-testz` reports | ✅ | golden C++ output in tests; out of repo stdout identical in all modes |

### codecbench (`tools/codecbench.cpp`) — spec `docs/spec/37-codecbench.md`

| Component | Status | Verification |
|---|---|---|
| synthetic + file modes | ✅ | setup hashes vs C++ (N = 1/7/64/100); N = 1000 sizes and file ratios identical |
| decoder throughput | ✅ | SIMD (spec 38): filters 0.99–1.10x, vtx 0.92x, mlet 0.70–0.90x of C++ SIMD; idx at parity |

### fuzz (`tools/*fuzz.cpp`) — spec `docs/spec/36-fuzz.md`

| Component | Status | Verification |
|---|---|---|
| codecfuzz, simplifyfuzz, clusterfuzz harnesses + glibc `rand` | ✅ | smoke runs in `cargo test`; 58M inputs in long runs |
| coverage-guided fuzzing (libFuzzer, `fuzz/libfuzzer`) | ✅ | approved as dev-only, excluded from the workspace; ASan, 20 min each: 10.5M codec, 2.0M simplify, 0.36M cluster runs without failures |

### clusterlod (`demo/clusterlod.h`) — spec `docs/spec/35-clusterlod.md`

| Component | Status | Verification |
|---|---|---|
| clod_build, clod_local_indices, clod_build_hierarchy(_bound), default configs | ✅ | 26 golden hashes vs C++ (pirate + grid x 13 configs); 90k-tri torus identical out of repo |

### gltfpack (`gltf/*.cpp`) — spec `docs/spec/30-gltfpack.md`

| Component | Status | Verification |
|---|---|---|
| cgltf + jsmn (`extern/cgltf.h`) | ✅ | dump-diffed vs C on samples + 3.5k fuzzed files |
| printf/strtod helpers (`cstd`) | ✅ | 2M random floats vs glibc |
| libstdc++ `std::sort`, x86 float casts (`cxx`) | ✅ | 3000 random arrays with ties vs libstdc++ |
| fast_obj (`extern/fast_obj.h`) | ✅ | dump-diffed vs C (OBJ/MTL edge cases) |
| micro_ply (`extern/micro_ply.h`) | ✅ | dump-diffed vs C (ASCII/binary/malformed) |
| parsegltf / parseobj / splats | ✅ | byte-identical output vs C++ gltfpack |
| mesh / stream / animation / node / material / image | ✅ | byte-identical output vs C++ gltfpack |
| json / write / fileio / gltfpack (CLI) | ✅ | 306 golden cases in repo; 73 models x 35 option sets identical out of repo |
| encodebasis (KTX2 via `qbart/ktx-rs`) | ✅ | 18 golden cases; 16 textured models x 12 option sets identical to C++ BasisU build (spec 34) |
| encodewebp (`-tw`) | ⏸ deferred | 1 golden case for the "built without WebP support" error |

**WebP deferred (human decision):** gltfpack needs libwebp's lossy encoder (`emulate_jpeg_size`,
`WebPPictureRescale`). The pure Rust crates (`zenwebp` AGPL, `oxideav-webp`) are independent encoders, so `-tw`
could not be byte-identical to C++ gltfpack, and the C `libwebp` bindings are not allowed. It stays unavailable,
like a C++ build without `WITH_LIBWEBP`, until a pure Rust libwebp encoder port is worth doing (spec 30).
