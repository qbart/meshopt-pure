# 08 — Index codec (`src/indexcodec.cpp`)

| C / C++ template | Rust |
|---|---|
| `meshopt_encodeIndexBuffer` (+ `<T>`) | `encode_index_buffer<T: IndexType>(buffer, buffer_size, indices, index_count) -> Result<usize>` |
| `meshopt_encodeIndexBufferBound` | `encode_index_buffer_bound(index_count, vertex_count) -> Result<usize>` |
| `meshopt_encodeIndexVersion` | `encode_index_version(i32) -> Result<()>` |
| `meshopt_decodeIndexBuffer` (+ `<T>`) | `decode_index_buffer<T: IndexType>(destination, index_count, buffer, buffer_size) -> Result<()>` |
| `meshopt_decodeIndexVersion` | `decode_index_version(buffer, buffer_size) -> Result<i32>` |
| `meshopt_encodeIndexSequence` (+ `<T>`) | `encode_index_sequence<T: IndexType>(...) -> Result<usize>` |
| `meshopt_encodeIndexSequenceBound` | `encode_index_sequence_bound(index_count, vertex_count) -> usize` |
| `meshopt_decodeIndexSequence` (+ `<T>`) | `decode_index_sequence<T: IndexType>(...) -> Result<()>` |

- `IndexType` (sealed, `u16`/`u32`) replaces both `void* + index_size` and the C++ `T*` templates; `index_size == T::SIZE`.
- Encoders keep C semantics: `Ok(0)` when the output buffer is too small.
- Decoder status codes: `Err(Error::Decode(-1 | -2 | -3))`; version query returns `Err(Decode(-1))` for bad headers.
- Global encoder version is an `AtomicI32` (no `unsafe`, thread-safe).
- Tests: all 21 index codec tests → `tests/indexcodec.rs`.
- Diff-tested: v0/v1 encodings byte-identical; ~600 truncated/corrupted streams decode with identical status/output, no panics.
