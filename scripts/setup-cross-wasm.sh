#!/bin/sh
# Sets up Wasm testing in ref/cross (gitignored), without root:
# - wasmtime (prebuilt release) as the runner for wasm32-wasip1
# - the Rust wasm32-wasip1 target
# The C++ Wasm oracles (spec 38) are built with the zig from setup-cross-aarch64.sh.
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
cross="$root/ref/cross"
wasmtime_version=v26.0.1

mkdir -p "$cross/dl" "$cross/wasi-tmp"
cd "$cross/dl"

if [ ! -x "$cross/wasmtime-$wasmtime_version-x86_64-linux/wasmtime" ]; then
  curl -fsSLo wasmtime.tar.xz "https://github.com/bytecodealliance/wasmtime/releases/download/$wasmtime_version/wasmtime-$wasmtime_version-x86_64-linux.tar.xz"
  tar -xf wasmtime.tar.xz -C "$cross"
fi

rustup target add wasm32-wasip1
