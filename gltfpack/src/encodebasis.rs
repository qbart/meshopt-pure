//! KTX2/BasisU texture encoding (`encodebasis.cpp`).
//!
//! Not wired up yet: behaves like a C++ gltfpack build without `WITH_BASISU`.

use crate::cgltf::Data;
use crate::types::*;

/// Whether BasisU encoding is available (C++: `WITH_BASISU`).
pub const SUPPORTED: bool = false;

/// `encodeImagesBasis`: fills `encoded[i]` with KTX2 data or an `error...` message.
pub fn encode_images_basis(_encoded: &mut [Vec<u8>], _data: &Data, _images: &[ImageInfo], _input_path: Option<&str>, _settings: &Settings) {
}
