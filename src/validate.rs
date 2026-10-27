//! Shared precondition checks; each replaces a C `assert` with an `Err`.

use crate::error::{Error, Result, ensure};

/// `indices` holds at least `index_count` entries, all `< vertex_count`.
pub(crate) fn indices(indices: &[u32], index_count: usize, vertex_count: usize) -> Result<&[u32]> {
    let indices = indices.get(..index_count).ok_or(Error::BufferTooSmall("indices"))?;
    ensure(indices.iter().all(|&i| (i as usize) < vertex_count), Error::IndexOutOfRange)?;
    Ok(indices)
}

/// `index_count` describes a triangle list.
pub(crate) fn triangles(index_count: usize) -> Result<()> {
    ensure(
        index_count.is_multiple_of(3),
        Error::InvalidArgument("index_count must be a multiple of 3"),
    )
}

/// Output slice holds at least `count` elements.
pub(crate) fn output<T>(destination: &mut [T], count: usize) -> Result<&mut [T]> {
    destination.get_mut(..count).ok_or(Error::BufferTooSmall("destination"))
}

/// Byte buffer holds `count` elements of `size` bytes spaced `stride` bytes apart.
pub(crate) fn bytes(data: &[u8], count: usize, size: usize, stride: usize) -> Result<()> {
    let required = if count == 0 { 0 } else { (count - 1) * stride + size };
    ensure(data.len() >= required, Error::BufferTooSmall("vertex data"))
}

/// Float vertex stream: `stride` in bytes (multiple of 4, `min_size..=256`), enough data for `vertex_count` elements.
pub(crate) fn floats(data: &[f32], vertex_count: usize, stride: usize, min_size: usize) -> Result<()> {
    ensure(
        stride >= min_size && stride <= 256,
        Error::InvalidArgument("vertex stride out of range"),
    )?;
    ensure(
        stride.is_multiple_of(4),
        Error::InvalidArgument("vertex stride must be a multiple of 4"),
    )?;
    let required = if vertex_count == 0 {
        0
    } else {
        ((vertex_count - 1) * stride + min_size) / 4
    };
    ensure(data.len() >= required, Error::BufferTooSmall("vertex positions"))
}
