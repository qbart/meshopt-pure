use crate::error::{Error, Result, ensure};
use crate::validate;

/// Generates vertex remap to reduce the amount of GPU memory fetches during vertex processing.
/// Returns the number of unique vertices; `destination` must hold `vertex_count` entries.
pub fn optimize_vertex_fetch_remap(destination: &mut [u32], indices: &[u32], index_count: usize, vertex_count: usize) -> Result<usize> {
    let indices = validate::indices(indices, index_count, vertex_count)?;
    let destination = validate::output(destination, vertex_count)?;

    destination.fill(!0u32);

    let mut next_vertex = 0u32;

    for &index in indices {
        let remap = &mut destination[index as usize];

        if *remap == !0u32 {
            *remap = next_vertex;
            next_vertex += 1;
        }
    }

    Ok(next_vertex as usize)
}

/// Reorders vertices and changes indices to reduce the amount of GPU memory fetches during vertex processing.
/// Returns the number of unique vertices; `destination` must hold `vertex_count * vertex_size` bytes.
/// Use [`optimize_vertex_fetch_in_place`] when the destination is the vertex buffer itself.
pub fn optimize_vertex_fetch(
    destination: &mut [u8],
    indices: &mut [u32],
    index_count: usize,
    vertices: &[u8],
    vertex_count: usize,
    vertex_size: usize,
) -> Result<usize> {
    ensure(
        vertex_size > 0 && vertex_size <= 256,
        Error::InvalidArgument("vertex_size must be in 1..=256"),
    )?;
    validate::indices(indices, index_count, vertex_count)?;
    validate::bytes(vertices, vertex_count, vertex_size, vertex_size)?;
    let destination = validate::output(destination, vertex_count * vertex_size)?;

    // build vertex remap table
    let mut vertex_remap = vec![!0u32; vertex_count];

    let mut next_vertex = 0u32;

    for index in indices.iter_mut().take(index_count) {
        let remap = &mut vertex_remap[*index as usize];

        if *remap == !0u32 {
            // vertex was not added to destination VB
            let src = *index as usize * vertex_size;
            let dst = next_vertex as usize * vertex_size;
            destination[dst..dst + vertex_size].copy_from_slice(&vertices[src..src + vertex_size]);

            *remap = next_vertex;
            next_vertex += 1;
        }

        // modify indices in place
        *index = *remap;
    }

    Ok(next_vertex as usize)
}

/// In-place variant of [`optimize_vertex_fetch`] (C: `destination == vertices`).
pub fn optimize_vertex_fetch_in_place(
    vertices: &mut [u8],
    indices: &mut [u32],
    index_count: usize,
    vertex_count: usize,
    vertex_size: usize,
) -> Result<usize> {
    validate::bytes(vertices, vertex_count, vertex_size, vertex_size)?;
    let vertices_copy = vertices[..vertex_count * vertex_size].to_vec();
    optimize_vertex_fetch(vertices, indices, index_count, &vertices_copy, vertex_count, vertex_size)
}
