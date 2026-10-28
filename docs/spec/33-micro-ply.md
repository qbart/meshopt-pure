# 33 — micro_ply (`extern/micro_ply.h`)

Module `gltfpack::micro_ply`.

## Mapping
- `ply_file_t`/`ply_element_t`/`ply_prop_t`/`ply_map_t` → `PlyFile`/`PlyElement`/`PlyProp`/`PlyMap`.
- `ply_read(data) -> Option<PlyFile>`; `ply_convert(file, name, map, stride) -> Option<(Vec<u8>, i32)>`; `ply_free` consumes.
- Fixed C name buffers (`name[32]`, `name[64]`, word `[128]`) keep their truncation lengths; names are `String`.
- `MICRO_PLY_ATOF` = `atof`, `atol` = `strtol` (glibc semantics via `cstd`).

## Divergences (C undefined behavior)
- `property` before any `element` (C: `elements[-1]`) and truncated binary bodies (C: reads past the buffer)
  return `None` ("invalid PLY").

## Verification
- Dump-diffed against C on ASCII/binary/list/splat-layout files and malformed headers.
