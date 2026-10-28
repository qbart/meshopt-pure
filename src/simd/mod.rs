//! SIMD decoders (C++ `SIMD_SSE`, `SIMD_NEON` and `SIMD_WASM` paths), enabled by the default `simd` feature.
//!
//! Like C++ (`SIMD_FALLBACK` + cpuid), the ISA is detected at runtime (Wasm: at compile time, from `simd128`): `detect()` returns a zero-sized token only when
//! the CPU supports the required features, and the token's methods are the only places that call `#[target_feature]`
//! functions (`unsafe` by language rules). Loads and stores go through bounds-checked arrays, so the SIMD code itself
//! is safe Rust. Without the feature or on other architectures the tokens are uninhabited and `detect()` is `None`.

#[cfg(all(
    feature = "simd",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )
))]
mod common;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod x86;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(crate) use x86::*;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod neon;
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub(crate) use neon::*;

#[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;
#[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) use wasm::*;

#[cfg(not(all(
    feature = "simd",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )
)))]
mod fallback;
#[cfg(not(all(
    feature = "simd",
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )
)))]
pub(crate) use fallback::*;
