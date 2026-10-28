# 32 — fast_obj 1.3 (`extern/fast_obj.h`)

Module `gltfpack::fast_obj`.

## Mapping
- `fastObjMesh`/`fastObjMaterial`/`fastObjTexture`/`fastObjGroup`/`fastObjIndex` → `FastObj*` structs, same fields
  (C arrays → `Vec`, `*_count` fields kept and filled at the end like C).
- `char*` names/paths → `Option<Vec<u8>>` raw bytes (`strlen`/`strcmp` semantics applied where C uses them).
- `face_lines` NULL → empty `Vec`.
- `fast_obj_read(path) -> Option<FastObjMesh>`; `fast_obj_destroy` consumes. File callbacks dropped (std::fs).

## Behavior kept
- 64KB chunked parsing (`BUFFER_SIZE`, 2x buffer, newline handling): a chunk without any newline stops parsing.
- `parse_float`/`parse_int` digit loops (double accumulation, `POWER_10_*` tables, `MAX_POWER` cut-off, int wrap).
- `illum` is always parsed as 0 (C passes the whitespace before the value to `parse_int`).
- Platform separators (`/` vs `\`) as in C.

## Verification
- Dump-diffed against C on pirate.obj, synthetic OBJ/MTL (relative indices, colors, lines, groups, CRLF, missing
  EOL, >64KB lines).
