#!/bin/bash
# Regenerates gltfpack/tests/data/expected with the C++ reference gltfpack (ref/meshopt, see AGENTS.md).
# The reference is built with BasisU from KTX-Software@2e690dd (basisu 2.10, the version ktx-rs ports), checked out
# into ref/KTX-Software, and without WebP to match the Rust port's feature set.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DATA=$ROOT/gltfpack/tests/data
BUILD=$ROOT/target/gltfpack-ref
KTXSW=$ROOT/ref/KTX-Software
KTXSW_REV=2e690dd

if [ ! -d "$KTXSW/external/basis_universal" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/KhronosGroup/KTX-Software "$KTXSW"
  git -C "$KTXSW" sparse-checkout set external/basis_universal
  git -C "$KTXSW" checkout "$KTXSW_REV"
fi

mkdir -p "$BUILD/obj"
if [ ! -x "$BUILD/gltfpack" ]; then
  B=$KTXSW/external/basis_universal
  BASISU="-DNDEBUG -DBASISU_SUPPORT_SSE=0 -DBASISU_SUPPORT_OPENCL=0 -DBASISD_SUPPORT_KTX2_ZSTD=1"
  {
    for f in "$B"/encoder/*.cpp "$B"/encoder/3rdparty/android_astc_decomp.cpp "$B"/encoder/3rdparty/tinyexr.cpp "$B"/transcoder/basisu_transcoder.cpp; do
      echo "g++ -O2 -std=c++17 $BASISU -I$B -c $f -o $BUILD/obj/$(basename "$f").o"
    done
    echo "gcc -O2 -DNDEBUG -c $B/zstd/zstd.c -o $BUILD/obj/zstd.o"
    for f in "$ROOT"/ref/meshopt/gltf/*.cpp "$ROOT"/ref/meshopt/src/*.cpp; do
      echo "g++ -O2 -std=c++17 -DWITH_BASISU -DBASISU_SUPPORT_SSE=0 -DBASISU_SUPPORT_OPENCL=0 -I$B -c $f -o $BUILD/obj/g_$(basename "$f").o"
    done
  } | xargs -P "$(nproc)" -I{} sh -c "{}"
  g++ -o "$BUILD/gltfpack" "$BUILD"/obj/*.o -lpthread
fi

rm -rf "$DATA/expected"
grep -v '^#' "$DATA/cases.txt" | while IFS='|' read -r name input options ext; do
  name=$(echo $name); input=$(echo $input); ext=$(echo $ext)
  out="$DATA/expected/$name"
  mkdir -p "$out"
  (cd "$out" && "$BUILD/gltfpack" -i "$DATA/$input" -o "out.$ext" $options > /dev/null 2>&1; echo $? > exitcode)
done
