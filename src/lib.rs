//! Pure Rust port of [meshoptimizer](https://github.com/zeux/meshoptimizer).
#![deny(unsafe_code)]

mod error;
mod math;
mod quantization;

pub use error::{Error, Result};
pub use quantization::*;
