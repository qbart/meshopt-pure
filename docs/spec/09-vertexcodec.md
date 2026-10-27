# 09 — Vertex codec (`src/vertexcodec.cpp`)

| C | Rust |
|---|---|
| `meshopt_encodeVertexBuffer` | `encode_vertex_buffer(buffer, buffer_size, vertices: &[u8], vertex_count, vertex_size) -> Result<usize>` |
| `meshopt_encodeVertexBufferLevel` | `encode_vertex_buffer_level(..., level, version) -> Result<usize>` |
| `meshopt_encodeVertexBufferBound` | `encode_vertex_buffer_bound(vertex_count, vertex_size) -> Result<usize>` |
| `meshopt_encodeVertexVersion` | `encode_vertex_version(i32) -> Result<()>` |
| `meshopt_decodeVertexVersion` | `decode_vertex_version(buffer, buffer_size) -> Result<i32>` |
| `meshopt_decodeVertexBuffer` | `decode_vertex_buffer(destination: &mut [u8], vertex_count, vertex_size, buffer, buffer_size) -> Result<()>` |

- Only the scalar decoder (`decodeVertexBlock`) is ported; SSE/AVX/NEON/WASM paths need `unsafe` intrinsics.
  `MESHOPTIMIZER_VERTEXCODEC_ZEROCOPY` is not supported (always decodes via the transposed scratch block).
- Generic `encodeDeltas4<T, Xor>`/`decodeDeltas4<T, Xor>` → runtime element width (1/2/4) with masked `u32` math.
- `TRACE` statistics are not ported.
- Global encoder version is an `AtomicI32`; tests touching it serialize via a mutex.
- Tests: all 19 vertex codec tests (12 run for both versions) + `decodeVersion` → `tests/vertexcodec.rs`.
- Diff-tested: encodings byte-identical for levels 0–3 × versions 0/1 × vertex sizes 4/8/12/32.
  Corrupted-stream decoding matches the C scalar build exactly; the C SIMD decoder can report `-2` where scalar reports `-3`.
