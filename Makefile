FUZZ_TARGETS := codec simplify cluster

# codecbench flags/inputs, e.g. make codecbench ARGS=-v
ARGS ?=

# pure Rust mutation driver (stable)
ITERATIONS ?= 1000000
SEED ?= 0

# libFuzzer (nightly + cargo-fuzz, see `make fuzz-setup`)
FUZZ_TIME ?= 600
FUZZ_MAX_LEN ?= 8192

.PHONY: codecbench codecbench-scalar fuzz $(addprefix fuzz-,$(FUZZ_TARGETS)) fuzz-replay \
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

# scalar decoders only (C++: MESHOPTIMIZER_NO_SIMD)
codecbench-scalar:
	cargo run -p meshoptimizer-tools --release --no-default-features --bin codecbench -- $(ARGS)
