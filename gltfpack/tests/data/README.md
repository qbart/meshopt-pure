# gltfpack test data

- `models/*`: glTF sample models from [KhronosGroup/glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets),
  CC0-1.0 or CC-BY-4.0; see each directory's `LICENSE.md` for credits.
- `models/pirate.obj`: Pirate by Clint Bellanger (https://opengameart.org/content/pirate), CC-BY-SA 3.0
  (from meshoptimizer `demo/`).
- `synthetic/*`: small hand-written OBJ/MTL edge cases and a random 3DGS splat PLY (public domain).
- `cases.txt`: golden test matrix (`name | input | options | output extension`).
- `expected/<name>/`: outputs of the C++ gltfpack 1.3 reference (built without BasisU/WebP) for each case,
  regenerated with `gltfpack/tests/gen-golden.sh`.
