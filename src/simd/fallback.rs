//! No SIMD: uninhabited tokens, so every SIMD call site compiles away.

use crate::vertexcodec::DecodeScratch;

/// `decodeVertexBlockSimd` (SSSE3 + POPCNT on x86).
#[derive(Clone, Copy)]
pub(crate) enum VertexDecoder {}

impl VertexDecoder {
    pub(crate) fn detect() -> Option<Self> {
        None
    }

    pub(crate) fn decode_vertex_block(
        self,
        _scratch: &mut DecodeScratch,
        _data: &[u8],
        _pos: usize,
        _vertex_data: &mut [u8],
        _vertex_count: usize,
        _vertex_size: usize,
        _last_vertex: &mut [u8; 256],
        _channels: Option<&[u8]>,
        _version: i32,
    ) -> Option<usize> {
        match self {}
    }
}
