//! Pure Rust port of [meshoptimizer](https://github.com/zeux/meshoptimizer).
#![deny(unsafe_code)]

mod clusterizer;
mod error;
mod hash;
mod index;
mod indexanalyzer;
mod indexcodec;
mod indexgenerator;
mod math;
mod meshletcodec;
mod meshletutils;
mod overdrawoptimizer;
mod partition;
mod quantization;
mod rasterizer;
mod spatialorder;
mod stripifier;
mod validate;
mod vcacheoptimizer;
mod vertexcodec;
mod vertexfilter;
mod vfetchoptimizer;

pub use clusterizer::*;
pub use error::{Error, Result};
pub use index::IndexType;
pub use indexanalyzer::*;
pub use indexcodec::*;
pub use indexgenerator::*;
pub use meshletcodec::*;
pub use meshletutils::*;
pub use overdrawoptimizer::*;
pub use partition::*;
pub use quantization::*;
pub use rasterizer::*;
pub use spatialorder::*;
pub use stripifier::*;
pub use vcacheoptimizer::*;
pub use vertexcodec::*;
pub use vertexfilter::*;
pub use vfetchoptimizer::*;
