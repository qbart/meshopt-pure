#!/bin/sh
# Sets up AArch64 testing under qemu in ref/cross (gitignored), without root:
# - qemu-aarch64-static, extracted from the distro package (`apt download`, Debian/Ubuntu hosts)
# - zig 0.14.1 (checksum verified) as C/C++ cross compiler and as linker for aarch64-unknown-linux-musl
# - the Rust aarch64-unknown-linux-musl target
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
cross="$root/ref/cross"
zig_version=0.14.1

mkdir -p "$cross/dl"
cd "$cross/dl"

if [ ! -x "$cross/qemu/usr/bin/qemu-aarch64-static" ]; then
  apt download qemu-user-static
  dpkg-deb -x qemu-user-static_*.deb "$cross/qemu"
fi

if [ ! -x "$cross/zig-x86_64-linux-$zig_version/zig" ]; then
  set -- $(curl -fsS https://ziglang.org/download/index.json | python3 -c "import json, sys; d = json.load(sys.stdin)['$zig_version']['x86_64-linux']; print(d['tarball'], d['shasum'])")
  curl -fsSLo zig.tar.xz "$1"
  echo "$2  zig.tar.xz" | sha256sum -c
  tar -xf zig.tar.xz -C "$cross"
fi

# zig's lld driver rejects the Cortex-A53 erratum flag that rustc passes; it doesn't matter under qemu
cat > "$cross/zig-cc-aarch64" <<WRAPPER
#!/bin/sh
for arg; do
  shift
  case "\$arg" in -Wl,--fix-cortex-a53-843419|--fix-cortex-a53-843419) ;; *) set -- "\$@" "\$arg" ;; esac
done
exec "$cross/zig-x86_64-linux-$zig_version/zig" cc -target aarch64-linux-musl "\$@"
WRAPPER
chmod +x "$cross/zig-cc-aarch64"

rustup target add aarch64-unknown-linux-musl
