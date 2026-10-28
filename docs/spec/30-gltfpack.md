# 30 — gltfpack (`gltf/*.cpp`, v1.3)

Workspace crate `gltfpack/` (lib + `gltfpack` binary). Same CLI, same output bytes as the C++ tool.

## Dependencies
| C/C++ dependency | Rust replacement |
|---|---|
| `extern/cgltf.h` (+ jsmn) | ported: `gltfpack/src/cgltf/` (spec 31) |
| `extern/fast_obj.h` | ported: `gltfpack/src/fast_obj.rs` (spec 32) |
| `extern/micro_ply.h` | ported: `gltfpack/src/micro_ply.rs` (spec 33) |
| `../src/meshoptimizer.h` | `meshoptimizer` crate (this repo) |
| BasisU encoder (`WITH_BASISU`) | `ktx-rs` (qbart/ktx-rs, pure Rust Basis ETC1S/UASTC) |
| libwebp (`WITH_LIBWEBP`) | ❓ no pure Rust lossy WebP encoder; needs human decision |
| `extern/sdefl.h` | not used by gltfpack |

## Module mapping
`gltfpack.cpp` → `main.rs`/`gltfpack.rs` (CLI + pipeline), `parsegltf.cpp`, `parseobj.cpp`, `splats.cpp`, `mesh.cpp`,
`stream.cpp`, `material.cpp`, `image.cpp`, `animation.cpp`, `node.cpp`, `json.cpp`, `fileio.cpp`, `write.cpp` →
modules of the same name. `encodebasis.cpp` → `encodebasis.rs` (ktx-rs). `wasistubs.cpp`, `library.js`, `cli.js` (wasm/npm) not ported.

## Conventions
- `std::string` byte buffers → `Vec<u8>`; JSON output assembled as bytes, formatted exactly like the C++ `append` helpers.
- Pointers into `cgltf_data` → indices; `std::vector<T*>` → `Vec<usize>`.
- `fprintf(stderr, …)`/`printf` messages kept verbatim (tests compare stdout/stderr where relevant).
- No panics: malformed inputs that are UB in C++ (asserts disabled) return an error instead.

## Verification
- Golden tests: sample models (versioned under `gltfpack/tests/data/`) packed with several option sets; outputs compared
  byte-for-byte with files produced by the C++ gltfpack 1.3 (built without BasisU/WebP), stored next to the inputs.
