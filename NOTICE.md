# NOTICE

This is an unofficial pure Rust port of [meshoptimizer](https://github.com/zeux/meshoptimizer), translated from
upstream commit `f0a684f` (v1.3). It is not affiliated with or endorsed by the meshoptimizer authors.

The Rust sources are translations of the C and C++ sources listed below and have been modified from the originals.
All of them are MIT licensed (micro_ply and sdefl are MIT or public domain; MIT is used). The copyright notices and the
license text are in [LICENSE.md](LICENSE.md); every published crate ships a copy.

| Rust port | Translated from | Copyright | License |
|---|---|---|---|
| `src/` (crate `meshopt-pure`) | meshoptimizer `src/` | Arseny Kapoulkine | MIT |
| `clusterlod/` | meshoptimizer `demo/clusterlod.h` | Arseny Kapoulkine | MIT |
| `gltfpack/` (except below) | meshoptimizer `gltf/` | Arseny Kapoulkine | MIT |
| `gltfpack/src/cgltf` | cgltf 1.15 (`extern/cgltf.h`, with jsmn) | Johannes Kuhlmann | MIT |
| `gltfpack/src/fast_obj.rs` | fast_obj 1.3 (`extern/fast_obj.h`) | Richard Knight | MIT |
| `gltfpack/src/micro_ply.rs` | micro_ply (`extern/micro_ply.h`) | Nick Klingensmith | MIT / public domain |
| `demo/` (not published) | meshoptimizer `demo/main.cpp`, `nanite.cpp` | Arseny Kapoulkine | MIT |
| `demo/src/sdefl.rs` (not published) | sdefl (`extern/sdefl.h`) | Micha Mettke | MIT / public domain |
| `tools/`, `fuzz/` (not published) | meshoptimizer `tools/` | Arseny Kapoulkine | MIT |

KTX2 encoding is provided by the `ktx-rs` crate and Zstandard by `zstd-pure-rs`, under their own licenses.

## Test data

Test data is versioned in the repository but not part of the published crates:

- `gltfpack/tests/data/models/*`: glTF sample models from
  [KhronosGroup/glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets), CC0-1.0 or CC-BY-4.0; see each
  directory's `LICENSE.md`.
- `pirate.obj`: Pirate by Clint Bellanger (https://opengameart.org/content/pirate), CC-BY-SA 3.0 (from meshoptimizer
  `demo/`).
- `gltfpack/tests/data/expected/*`: C++ gltfpack outputs for the inputs above; they keep the input's license.
- Everything else (generated meshes, synthetic edge cases, golden text outputs): produced for this port, MIT.
