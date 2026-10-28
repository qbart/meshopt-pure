# 38 — SIMD decoders (`SIMD_SSE` / `SIMD_NEON` paths of vertexcodec, vertexfilter, meshletcodec)

Feature `simd` (default on). Without it, or on targets other than x86-64 and AArch64, the crate has no `unsafe` and
uses the scalar paths (C++: `MESHOPTIMIZER_NO_SIMD`, or a target without SIMD support).

## Dispatch (C++: `SIMD_FALLBACK` + cpuid)
`src/simd/mod.rs` selects `x86.rs`, `neon.rs` or `fallback.rs`; ISA-independent lookup tables and window helpers
live in `common.rs`. Each path has a zero-sized token whose `detect()` checks
the CPU at runtime with `is_x86_feature_detected!`. The fallback tokens are uninhabited enums, so `detect()` is
`None` and the SIMD branches compile away.

| Token | Features | C++ | Used by |
|---|---|---|---|
| `VertexDecoder` | ssse3, popcnt | `decodeVertexBlockSimd` (+`SIMD_LATENCYOPT`) | `decode_vertex_buffer` |
| `FilterDecoder` | sse2 (x86-64 baseline) | `decodeFilter{Oct,Quat,Exp,Color}Simd` + `dispatchSimd` | `decode_filter_*` |
| `MeshletDecoder` | sse4.1 | `decodeMeshletSimd` | `decode_meshlet`, `decode_meshlet_raw` |

On AArch64 the same three tokens require `neon`, which is always present (C++ enables `SIMD_NEON` unconditionally).
The NEON code mirrors the C++ NEON paths: vertex groups via `vtbl1` shuffles and the `neonMoveMask` multiply, filters
with `vfmaq_f32`, the `3 << 22` snap rounding and `frsqrte`/`frecpe` estimates, meshlets via `vqtbl1q`. Loads and
stores use `vcreate`/`vcombine` and lane extracts instead of pointer intrinsics.

## `unsafe` policy (AGENTS.md rule 9)
- The only `unsafe` blocks are the token methods calling `#[target_feature]` functions. Owning a token proves
  that `detect()` found the features.
- Intrinsics are safe inside `#[target_feature]` functions. Loads and stores avoid raw pointers: `load16`/`store16`
  go through `&[u8; 16]` (they compile to `movdqu`), and every input window is a bounds-checked slice
  (`first_chunk`). A malformed stream can therefore fail a bounds check (`None` → `Err(Decode)`), but cannot read out
  of bounds.

## Divergences
- Lookup tables (`kDecodeBytesGroupShuffle`, `kDecodeTableMasks/Extra/Verts/Length`) are `const` items, not
  built at startup.
- Meshlet tails: C++ writes 16-bit vertices and 3-byte triangles rounded up to 4 bytes (its API requires that
  space). The Rust API takes exactly `count` elements, so tails are written exactly. The decoded values are the
  same.
- Filter tails (`dispatchSimd`): the padded copy is processed as a whole group of 4 elements (C++ passes
  `count - count4`, but its loop rounds up).
- Not ported: the AVX-512 vertex path. C++ only enables it at compile time, and on Zen 5 it measured 6.61 vs
  6.56 GB/s with SSSE3. Not ported: the NEON `rebase` latency trick (C++ only uses it on Apple/Windows; it doesn't
  change results), 32-bit ARM paths, Wasm SIMD (⬜).

## Output equivalence
- Vertex and meshlet codecs: SIMD and scalar decode the same format. Unit tests (`simd_matches_scalar` in
  `vertexcodec`/`meshletcodec`) compare both paths on valid and corrupted streams, for all strides, versions and
  levels, and for all four meshlet output types. On corrupt data the error codes may differ, as in C++: the
  meshlet SIMD path checks bounds per triangle pair, and the vertex SIMD path checks literal runs against the
  aligned count.
- Filters: as in C++, SIMD output differs from scalar (`cvtps2dq` rounding; oct8/color8 use `rsqrtps`/`rcpps`).
  `tests/vertexfilter_reference.rs` compares hashes of 13.4k random buffers per filter with the C++ builds: x86-64
  SSE, x86-64 scalar and AArch64 NEON (run under qemu). All 18 hashes matched. The test skips SSE oct8/color8
  because `rsqrtps`/`rcpps` precision depends on the CPU vendor; the NEON estimates are architecturally defined.
- The C++ *scalar* build for AArch64 (clang via zig) gives different oct16/quat/color hashes from x86 scalar. Color
  differs because `int(float)` saturates on ARM; the others most likely because clang contracts `a*b + c` into FMA by
  default (not confirmed). Rust never contracts, and color uses the x86 conversion, so the Rust scalar path gives
  the x86 results on every target.
- Scalar color decoding now uses C++'s x86 `int(float)` semantics (`INT_MIN` for out of range values and NaN,
  reachable with a zero alpha scale). It was saturating before and diverged on malformed data.

## Verification
- Workspace tests with and without `simd`; codec fuzzing (pure Rust driver and libFuzzer) with the SIMD paths
  active; `cargo check` for wasm32 (fallback).
- AArch64 under qemu-user (`make cross-setup`, `make test-aarch64`, `make fuzz-aarch64`, see
  `scripts/setup-cross-aarch64.sh`): `aarch64-unknown-linux-musl` linked by zig, run by `qemu-aarch64-static`, no root
  needed. All 37 workspace test binaries pass with NEON (including the 306 gltfpack golden cases, the clusterlod
  hashes, and the NEON-vs-scalar differential tests), and the scalar build passes too. 2M codec fuzz inputs, no
  failures.
