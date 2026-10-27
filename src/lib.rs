//! Pure Rust port of [meshoptimizer](https://github.com/zeux/meshoptimizer).
#![deny(unsafe_code)]

mod error;
mod hash;
mod index;
mod indexanalyzer;
mod indexcodec;
mod indexgenerator;
mod math;
mod overdrawoptimizer;
mod quantization;
mod rasterizer;
mod stripifier;
mod validate;
mod vcacheoptimizer;
mod vertexcodec;
mod vfetchoptimizer;

pub use error::{Error, Result};
pub use index::IndexType;
pub use indexanalyzer::*;
pub use indexcodec::*;
pub use indexgenerator::*;
pub use overdrawoptimizer::*;
pub use quantization::*;
pub use rasterizer::*;
pub use stripifier::*;
pub use vcacheoptimizer::*;
pub use vertexcodec::*;
pub use vfetchoptimizer::*;
