#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| meshoptimizer_fuzz::simplifyfuzz::test_one_input(data));
