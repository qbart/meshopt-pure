# 31 — cgltf 1.15 + jsmn (`extern/cgltf.h`)

Module `gltfpack::cgltf`. Port of the parser parts used by gltfpack (parse, load buffers, validate, accessors, transforms).

## Mapping
- `cgltf_xxx` struct → `cgltf::Xxx`, `cgltf_xxx_yyy` enum → `cgltf::XxxYyy`; field names kept.
- `T* ptr` (object reference) → `Option<usize>` index into the owning `Data` vector; `cgltf_xxx_index(data, p)` is the identity.
- `T* array` + `array_count` → `Vec<T>`; `char*` → `Option<String>` (raw JSON string bytes, escapes not decoded, like C).
- `cgltf_extras.data`/`cgltf_extension.data` → raw JSON text `Option<String>`.
- `cgltf_filter_type`/`cgltf_wrap_mode` stay `i32` (C stores any JSON integer via cast).
- `cgltf_camera.data` union → separate `perspective`/`orthographic` fields.
- `cgltf_options` keeps `type_` and `json_token_count`; memory/file callbacks dropped (Rust allocation, `std::fs`).
- `cgltf_parse*` → `Result<Data, CgltfResult>`; `cgltf_free(data)` consumes the value.
- Buffer data: `Option<Vec<u8>>`; the GLB BIN chunk is moved into buffer 0 (`data_free_method == None`).

## Parser
- jsmn (strict mode, parent links) ported exactly, incl. the two-pass token counting.
- Token access is bounds-checked: reads past the token array yield an `UNDEFINED` token (C relies on one sentinel).
- `CGLTF_PTRINDEX`/`CGLTF_PTRFIXUP` semantics kept: JSON index `-1` (or a non-primitive token) means "no reference",
  other out-of-range indices fail fixup with `invalid_gltf`.
- `atoi`/`atoll`/`atof` reproduce glibc semantics on the token prefix (`strtol` truncation to `int`, `strtoll`
  saturation, `strtod` decimal/inf/nan/hex; decimal conversion is correctly rounded).

## Divergences (C UB / crashes only)
- Null dereferences on invalid files (e.g. attribute without accessor after a failed fixup) return `invalid_gltf`.
- Out-of-bounds buffer reads in accessor helpers read as failure/zero instead of UB.
- Non-UTF-8 bytes in JSON strings are replaced (lossy); glTF requires UTF-8.
