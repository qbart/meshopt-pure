# 00 — Porting conventions

Reference: `ref/meshopt` (zeux/meshoptimizer, v1.3, commit `f0a684f`). Not versioned; clone with
`git clone https://github.com/zeux/meshoptimizer.git ref/meshopt`.

## API mapping
- `meshopt_fooBar` → `meshoptimizer::foo_bar`; parameter order and names kept.
- Caller owns all output buffers (`&mut [T]`), exactly like the C API; nothing returns a `Vec`.
- Element counts and byte strides are kept as explicit parameters; slices are validated against them.
- `const void*` vertex data → `&[u8]`; `const float*` positions → `&[f32]` + byte stride (multiple of 4).
- Optional pointers (`NULL`) → `Option<&[T]>`.
- Index buffers are generic over `IndexType` (`u16`/`u32`) wherever C++ has a `template <typename T>` wrapper; see `20-index-types.md`.
- C structs keep their fields (`meshopt_Bounds` → `Bounds`, …).

## Errors
- No `unwrap`/`expect`/`panic!` in the library (clippy-denied), no `unsafe` (`unsafe_code = deny`).
- Every C `assert` precondition becomes a returned `Err(Error::…)`; checks run before any output is written.
- Fallible functions return `Result<T>` (Go-style `value, err` maps onto `Ok`/`Err`).
- Decoders returning C status codes return `Err(Error::Decode(code))` with the original negative code.

## Memory
- Temporary storage uses `Vec`; it is released when the call returns (RAII), same lifetime as the C temporaries.
- `meshopt_setAllocator`/`meshopt_setAllocatorCache` are **not ported** (human decision, 2026-10-28): raw allocator
  callbacks need `unsafe`; Rust users customise allocation via `#[global_allocator]`. The C tests `customAllocator`
  and `customAllocatorCache` are therefore not applicable. Same choice as the bindings crate `meshopt`, which exposes
  `setAllocator` only as a raw `unsafe` FFI symbol.

## Numerics
- Only scalar code paths are ported (SIMD paths need `unsafe` intrinsics); outputs must be bit-identical.
- `float` math stays `f32`; integer wraparound semantics from C (`unsigned`) use `wrapping_*`.

## Tests
- `demo/tests.cpp` is ported 1:1 into `tests/*.rs`, one Rust test per C++ test function, same name in snake_case.
