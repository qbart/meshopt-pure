//! Pure Rust port of [meshoptimizer](https://github.com/zeux/meshoptimizer).
#![deny(unsafe_code)]

mod error;
mod indexanalyzer;
mod math;
mod quantization;
mod validate;
mod vfetchoptimizer;

pub use error::{Error, Result};
pub use indexanalyzer::*;
pub use quantization::*;
pub use vfetchoptimizer::*;
