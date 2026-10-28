//! No SIMD: uninhabited tokens, so every SIMD call site compiles away.

use crate::index::IndexType;
use crate::meshletcodec::MeshletTriangleType;
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

/// `decodeFilter*Simd` (SSE2 on x86).
#[derive(Clone, Copy)]
pub(crate) enum FilterDecoder {}

impl FilterDecoder {
    pub(crate) fn detect() -> Option<Self> {
        None
    }

    pub(crate) fn decode_oct(self, _data: &mut [u8], _count: usize, _stride: usize) {
        match self {}
    }

    pub(crate) fn decode_quat(self, _data: &mut [u8], _count: usize) {
        match self {}
    }

    pub(crate) fn decode_exp(self, _data: &mut [u8], _count: usize) {
        match self {}
    }

    pub(crate) fn decode_color(self, _data: &mut [u8], _count: usize, _stride: usize) {
        match self {}
    }
}

/// `decodeMeshletSimd` (SSE4.1 on x86).
#[derive(Clone, Copy)]
pub(crate) enum MeshletDecoder {}

impl MeshletDecoder {
    pub(crate) fn detect() -> Option<Self> {
        None
    }

    pub(crate) fn decode_vertices<V: IndexType>(
        self,
        _vertices: &mut [V],
        _ctrl: &[u8],
        _buffer: &[u8],
        _data: usize,
        _bound: usize,
        _vertex_count: usize,
    ) -> Option<usize> {
        match self {}
    }

    pub(crate) fn decode_triangles<T: MeshletTriangleType>(
        self,
        _triangles: &mut [T],
        _codes: &[u8],
        _buffer: &[u8],
        _extra: usize,
        _bound: usize,
        _triangle_count: usize,
    ) -> Option<usize> {
        match self {}
    }
}
