# 30 — gltfpack (`gltf/*.cpp`, v1.3)

Workspace crate `gltfpack/` (lib + `gltfpack` binary). Same CLI, same output bytes as the C++ tool.

## Dependencies
| C/C++ dependency | Rust replacement |
|---|---|
| `extern/cgltf.h` (+ jsmn) | ported: `gltfpack/src/cgltf/` (spec 31) |
| `extern/fast_obj.h` | ported: `gltfpack/src/fast_obj.rs` (spec 32) |
| `extern/micro_ply.h` | ported: `gltfpack/src/micro_ply.rs` (spec 33) |
| `../src/meshoptimizer.h` | `meshoptimizer` crate (this repo) |
| BasisU encoder (`WITH_BASISU`) | `ktx-rs` (qbart/ktx-rs, pure Rust Basis ETC1S/UASTC) + `zstd-pure-rs` (spec 34) |
| libwebp (`WITH_LIBWEBP`) | deferred, not built (see below) |
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

## C++ semantics preserved (`cxx.rs`, `cstd.rs`)
- `std::min`/`std::max` on floats (NaN order), x86-64 float→int conversions, libstdc++ `std::sort` (introsort;
  equal bone weights keep the libstdc++ order), glibc `printf("%.9g")`/`atof`/`atoi`.
- Math via Rust std (`expf`, `powf`, `acosf`, `cosf`, `sinf` resolve to the platform libm like the C++ build).

## Divergences (C++ undefined behavior / crashes only)
- Malformed PLY (property before element, truncated binary data) → "invalid PLY" instead of crash/garbage.
- OBJ faces with < 3 vertices (index count wraps, C++ aborts on allocation) are skipped with a warning.
- Out-of-range OBJ/accessor reads read as zero; failing meshoptimizer calls on asserted-invalid arguments leave
  the affected stream empty/zeroed.
- BasisU divergences are listed in spec 34.

## Deferred: WebP (`-tw`, human decision)
Behaves like C++ gltfpack built without `WITH_LIBWEBP`: "gltfpack was built without WebP support", exit 3.
Why: `encodewebp.cpp` uses libwebp's lossy VP8 encoder with `emulate_jpeg_size` and `WebPPictureRescale`.
- C bindings (`webp`, `webpx`, `webp-rs`) are not allowed (AGENTS.md).
- `image-webp` encodes lossless only; `zenwebp` (AGPL-3.0 since 0.4) and `oxideav-webp` are independent lossy
  encoders, so `-tw` output could not be byte-identical to C++ gltfpack, which every other path guarantees.
- A byte-identical result needs a port of the libwebp encoder (VP8 + alpha, quality mapping, rescaler): large and
  low priority next to KTX2, the recommended gltfpack texture path.
Revisit if a pure Rust libwebp port appears or non-identical WebP output becomes acceptable.

## Verification
- `gltfpack/tests/golden.rs`: 306 cases (18 with KTX2 textures, 1 for `-tw`) (`tests/data/cases.txt`: 18 Khronos sample models, pirate.obj, OBJ/PLY edge
  cases x up to 14 option sets, .glb and .gltf output) compared byte-for-byte (all output files + exit code) with
  `tests/data/expected/`, produced by the C++ gltfpack 1.3 reference via `tests/gen-golden.sh`.
- Out-of-repo diff runs: 73 sample model files x 35 option sets, plus stdout/stderr, all identical.
