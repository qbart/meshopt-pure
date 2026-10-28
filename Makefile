FUZZ_TARGETS := codec simplify cluster

# AArch64 (NEON) testing under qemu; tools live in ref/cross, see `make cross-setup`
CROSS := $(CURDIR)/ref/cross
AARCH64 := aarch64-unknown-linux-musl
AARCH64_ENV := CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER=$(CROSS)/zig-cc-aarch64 \
	CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_RUNNER=$(CROSS)/qemu/usr/bin/qemu-aarch64-static \
	CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_RUSTFLAGS="-C link-self-contained=no"

# codecbench flags/inputs, e.g. make codecbench ARGS=-v
ARGS ?=

# pure Rust mutation driver (stable)
ITERATIONS ?= 1000000
SEED ?= 0

# libFuzzer (nightly + cargo-fuzz, see `make fuzz-setup`)
FUZZ_TIME ?= 600
FUZZ_MAX_LEN ?= 8192

.PHONY: codecbench codecbench-scalar codectest cross-setup test-aarch64 fuzz-aarch64 fuzz $(addprefix fuzz-,$(FUZZ_TARGETS)) fuzz-replay \
	libfuzzer $(addprefix libfuzzer-,$(FUZZ_TARGETS)) fuzz-setup

fuzz: $(addprefix fuzz-,$(FUZZ_TARGETS))

$(addprefix fuzz-,$(FUZZ_TARGETS)): fuzz-%:
	cargo run -p meshoptimizer-fuzz --release --bin fuzz -- $* $(ITERATIONS) $(SEED)

# make fuzz-replay TARGET=codec FILE=fuzz-codec-crash.bin
fuzz-replay:
	cargo run -p meshoptimizer-fuzz --release --bin fuzz -- $(TARGET) --replay $(FILE)

libfuzzer: $(addprefix libfuzzer-,$(FUZZ_TARGETS))

$(addprefix libfuzzer-,$(FUZZ_TARGETS)): libfuzzer-%:
	cargo +nightly fuzz run --fuzz-dir fuzz/libfuzzer $* -- -max_total_time=$(FUZZ_TIME) -max_len=$(FUZZ_MAX_LEN)

fuzz-setup:
	rustup toolchain install nightly --profile minimal
	cargo install cargo-fuzz --locked

codecbench:
	cargo run -p meshoptimizer-tools --release --bin codecbench -- $(ARGS)

# make codectest ARGS="-test v*.bin", or ARGS=<stride> [<count>] with stdin/stdout redirected
codectest:
	cargo run -q -p meshoptimizer-tools --release --bin codectest -- $(ARGS)

# scalar decoders only (C++: MESHOPTIMIZER_NO_SIMD)
codecbench-scalar:
	cargo run -p meshoptimizer-tools --release --no-default-features --bin codecbench -- $(ARGS)

cross-setup:
	scripts/setup-cross-aarch64.sh

# NEON paths, then the scalar fallback
test-aarch64:
	$(AARCH64_ENV) cargo test --workspace --release --target $(AARCH64)
	$(AARCH64_ENV) cargo test --release --target $(AARCH64) --no-default-features

fuzz-aarch64:
	$(AARCH64_ENV) cargo run -p meshoptimizer-fuzz --release --target $(AARCH64) --bin fuzz -- codec $(ITERATIONS) $(SEED)
