# 39 — codectest (`tools/codectest.cpp`)

`tools/` crate, binary `codectest` (`make codectest ARGS=...`). `codectest::run(args, stdin, stdout, stderr)`
returns the exit code, so tests can drive it in memory.

## Modes (same output format as C++)
- `codectest <stride>`: encodes stdin with `encode_vertex_buffer` and writes the stream to stdout.
- `codectest <stride> <count>`: decodes stdin. A decode failure prints `Error decoding input: <code>` and
  exits 2. Bad arguments print usage and exit 1.
- `codectest -test[z] [-<level>] files...`: for each `v<count>_s<stride>_<name>[_R...]` file, re-encodes the stream
  at `level` (default 2), reading raw data when the file isn't an encoded stream. Prints the raw size, the vtx ratio
  and the geometric mean / totals. `-testz` adds `lz4` / `zstd` ratios on raw and encoded data.

## Divergences
- `lz4`/`zstd` run as external programs through `std::process::Command` (C++: `system()` through `sh`; no C code
  linked). Temp files get a unique time-based name (`codectest-<suffix>.in/.out` in the temp dir; `process::id` is unsupported on WASI) instead of fixed `/tmp` names.
  A missing tool counts as size 0, as in C++, but without the shell's "not found" message.
- A file name that doesn't match `v%d_s%d_` prints ` error parsing file name` (C++: assert / undefined behavior).
  The display name is empty when there is no second `_` (C++: null dereference).
- `atoi` and the `%25.*s` / `%.3f` / `nan` formatting are reproduced by hand.

## Verification
- `tools/tests/codectest.rs`: `-test` and `-test -3` on 4 generated files (smooth, noise, pre-encoded, corrupt) with
  C++ output as golden text; stdin encode/decode roundtrip, decode error and usage exit codes; `atoi`.
- Out of repo: stdout identical to the C++ tool for `-test`, `-test -0/-3`, `-testz`, `-testz -1` on 6 files
  (including a 25+ character name and `_R` suffixes); identical stdin encode/decode bytes and exit codes.
