//! SIMD decoders (C++ `SIMD_SSE` paths), enabled by the default `simd` feature.
//!
//! Like C++ (`SIMD_FALLBACK` + cpuid), the ISA is detected at runtime: `detect()` returns a zero-sized token only when
//! the CPU supports the required features, and the token's methods are the only places that call `#[target_feature]`
//! functions (`unsafe` by language rules). Loads and stores go through bounds-checked arrays, so the SIMD code itself
//! is safe Rust. Without the feature or on other architectures the tokens are uninhabited and `detect()` is `None`.

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod x86;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(crate) use x86::*;

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
mod fallback;
#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
pub(crate) use fallback::*;
