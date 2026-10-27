//! Pure Rust port of [meshoptimizer](https://github.com/zeux/meshoptimizer).
#![deny(unsafe_code)]

mod error;
mod hash;
mod indexanalyzer;
mod indexgenerator;
mod math;
mod quantization;
mod validate;
mod vcacheoptimizer;
mod vfetchoptimizer;

pub use error::{Error, Result};
pub use indexanalyzer::*;
pub use indexgenerator::*;
pub use quantization::*;
pub use vcacheoptimizer::*;
pub use vfetchoptimizer::*;
