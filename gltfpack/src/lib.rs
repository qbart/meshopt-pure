//! Pure Rust port of gltfpack (https://github.com/zeux/meshoptimizer/tree/master/gltf), version 1.3.
#![deny(unsafe_code)]
// index loops and expressions mirror the C++ sources for review against the reference
#![allow(clippy::needless_range_loop, clippy::explicit_counter_loop, clippy::int_plus_one)]

pub mod cgltf;
pub mod cstd;
pub mod cxx;
pub mod fast_obj;
pub mod micro_ply;

pub mod animation;
pub mod encodebasis;
pub mod fileio;
#[allow(clippy::module_inception)]
pub mod gltfpack;
pub mod image;
pub mod json;
pub mod material;
pub mod mesh;
pub mod node;
pub mod parsegltf;
pub mod parseobj;
pub mod splats;
pub mod stream;
pub mod types;
pub mod write;

pub use gltfpack::{defaults, get_version, gltfpack, run};
