#!/bin/bash
# Regenerates gltfpack/tests/data/expected with the C++ reference gltfpack (ref/meshopt, see AGENTS.md).
# The reference is built without BasisU/WebP to match the Rust port's feature set.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA=$ROOT/gltfpack/tests/data
BUILD=$ROOT/target/gltfpack-ref
mkdir -p "$BUILD"
if [ ! -x "$BUILD/gltfpack" ]; then
  g++ -O2 -std=c++11 -o "$BUILD/gltfpack" "$ROOT"/ref/meshopt/gltf/*.cpp "$ROOT"/ref/meshopt/src/*.cpp
fi
rm -rf "$DATA/expected"
grep -v '^#' "$DATA/cases.txt" | while IFS='|' read -r name input options ext; do
  name=$(echo $name); input=$(echo $input); ext=$(echo $ext)
  out="$DATA/expected/$name"
  mkdir -p "$out"
  (cd "$out" && "$BUILD/gltfpack" -i "$DATA/$input" -o "out.$ext" $options > /dev/null 2>&1; echo $? > exitcode)
done
