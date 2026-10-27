// This work is based on:
// Matthias Teschner, Bruno Heidelberger, Matthias Mueller, Danat Pomeranets, Markus Gross. Optimized Spatial Hashing for Collision Detection of Deformable Objects. 2003
// John McDonald, Mark Kilgard. Crack-Free Point-Normal Triangles using Adjacent Edge Normals. 2010
// John Hable. Variable Rate Shading with Visibility Buffer Rendering. 2024

use crate::error::{Error, Result, ensure};
use crate::hash::{Hasher, hash_buckets, hash_lookup, hash_update};
use crate::validate;

/// Vertex attribute stream (C: `meshopt_Stream`).
/// Each element takes `size` bytes, beginning at `data`, with `stride` controlling the spacing between successive elements (`stride >= size`).
#[derive(Debug, Clone, Copy)]
pub struct Stream<'a> {
    pub data: &'a [u8],
    pub size: usize,
    pub stride: usize,
}

/// Custom vertex equality callback for [`generate_vertex_remap_custom`] (C: callback + context).
pub type VertexEqualCallback<'a> = &'a mut dyn FnMut(u32, u32) -> bool;

struct VertexHasher<'a> {
    vertices: &'a [u8],
    vertex_size: usize,
    vertex_stride: usize,
}

impl VertexHasher<'_> {
    fn bytes(&self, index: u32) -> &[u8] {
        let offset = index as usize * self.vertex_stride;
        &self.vertices[offset..offset + self.vertex_size]
    }
}

impl Hasher<u32> for VertexHasher<'_> {
    fn hash(&self, index: &u32) -> usize {
        hash_update(0, self.bytes(*index)) as usize
    }

    fn equal(&mut self, lhs: &u32, rhs: &u32) -> bool {
        self.bytes(*lhs) == self.bytes(*rhs)
    }
}

struct VertexStreamHasher<'a> {
    streams: &'a [Stream<'a>],
}

fn stream_bytes<'a>(s: &Stream<'a>, index: u32) -> &'a [u8] {
    let offset = index as usize * s.stride;
    &s.data[offset..offset + s.size]
}

impl Hasher<u32> for VertexStreamHasher<'_> {
    fn hash(&self, index: &u32) -> usize {
        let mut h = 0;

        for s in self.streams {
            h = hash_update(h, stream_bytes(s, *index));
        }

        h as usize
    }

    fn equal(&mut self, lhs: &u32, rhs: &u32) -> bool {
        self.streams.iter().all(|s| stream_bytes(s, *lhs) == stream_bytes(s, *rhs))
    }
}

/// Hashes float3 positions by their binary representation (C: `VertexHasher` over positions).
struct PositionBitsHasher<'a> {
    vertex_positions: &'a [f32],
    vertex_stride_float: usize,
}

impl PositionBitsHasher<'_> {
    fn bits(&self, index: u32) -> [u32; 3] {
        let p = &self.vertex_positions[index as usize * self.vertex_stride_float..];
        [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]
    }
}

impl Hasher<u32> for PositionBitsHasher<'_> {
    fn hash(&self, index: &u32) -> usize {
        let [x, y, z] = self.bits(*index);
        let mut key = [0u8; 12];
        key[0..4].copy_from_slice(&x.to_le_bytes());
        key[4..8].copy_from_slice(&y.to_le_bytes());
        key[8..12].copy_from_slice(&z.to_le_bytes());
        hash_update(0, &key) as usize
    }

    fn equal(&mut self, lhs: &u32, rhs: &u32) -> bool {
        self.bits(*lhs) == self.bits(*rhs)
    }
}

struct VertexCustomHasher<'a, 'b> {
    vertex_positions: &'a [f32],
    vertex_stride_float: usize,
    callback: Option<VertexEqualCallback<'b>>,
}

impl VertexCustomHasher<'_, '_> {
    fn position(&self, index: u32) -> &[f32] {
        let offset = index as usize * self.vertex_stride_float;
        &self.vertex_positions[offset..offset + 3]
    }
}

impl Hasher<u32> for VertexCustomHasher<'_, '_> {
    fn hash(&self, index: &u32) -> usize {
        let p = self.position(*index);
        let (mut x, mut y, mut z) = (p[0].to_bits(), p[1].to_bits(), p[2].to_bits());

        // replace negative zero with zero
        x = if x == 0x8000_0000 { 0 } else { x };
        y = if y == 0x8000_0000 { 0 } else { y };
        z = if z == 0x8000_0000 { 0 } else { z };

        // scramble bits to make sure that integer coordinates have entropy in lower bits
        x ^= x >> 17;
        y ^= y >> 17;
        z ^= z >> 17;

        // Optimized Spatial Hashing for Collision Detection of Deformable Objects
        (x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663) ^ z.wrapping_mul(83492791)) as usize
    }

    fn equal(&mut self, lhs: &u32, rhs: &u32) -> bool {
        let (lp, rp) = (self.position(*lhs), self.position(*rhs));

        if lp[0] != rp[0] || lp[1] != rp[1] || lp[2] != rp[2] {
            return false;
        }

        match self.callback.as_mut() {
            Some(callback) => callback(*lhs, *rhs),
            None => true,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct TriangleKey {
    a: u32,
    b: u32,
    c: u32,
}

struct TriangleKeyHasher;

impl Hasher<TriangleKey> for TriangleKeyHasher {
    fn hash(&self, k: &TriangleKey) -> usize {
        // Optimized Spatial Hashing for Collision Detection of Deformable Objects
        (k.a.wrapping_mul(73856093) ^ k.b.wrapping_mul(19349663) ^ k.c.wrapping_mul(83492791)) as usize
    }

    fn equal(&mut self, lhs: &TriangleKey, rhs: &TriangleKey) -> bool {
        lhs == rhs
    }
}

struct EdgeHasher<'a> {
    remap: &'a [u32],
}

impl Hasher<u64> for EdgeHasher<'_> {
    fn hash(&self, edge: &u64) -> usize {
        let e0 = (edge >> 32) as u32;
        let e1 = *edge as u32;

        let mut h1 = self.remap[e0 as usize];
        let mut h2 = self.remap[e1 as usize];

        const M: u32 = 0x5bd1e995;

        // MurmurHash64B finalizer
        h1 ^= h2 >> 18;
        h1 = h1.wrapping_mul(M);
        h2 ^= h1 >> 22;
        h2 = h2.wrapping_mul(M);
        h1 ^= h2 >> 17;
        h1 = h1.wrapping_mul(M);
        h2 ^= h1 >> 19;
        h2 = h2.wrapping_mul(M);

        h2 as usize
    }

    fn equal(&mut self, lhs: &u64, rhs: &u64) -> bool {
        let (l0, l1) = ((lhs >> 32) as usize, *lhs as u32 as usize);
        let (r0, r1) = ((rhs >> 32) as usize, *rhs as u32 as usize);

        self.remap[l0] == self.remap[r0] && self.remap[l1] == self.remap[r1]
    }
}

fn build_position_remap(remap: &mut [u32], vertex_positions: &[f32], vertex_count: usize, vertex_positions_stride: usize) {
    let mut vertex_hasher = PositionBitsHasher {
        vertex_positions,
        vertex_stride_float: vertex_positions_stride / 4,
    };

    let mut vertex_table = vec![!0u32; hash_buckets(vertex_count)];

    for (i, r) in remap.iter_mut().enumerate().take(vertex_count) {
        let index = i as u32;
        let entry = hash_lookup(&vertex_table, &mut vertex_hasher, &index, &!0u32);

        if vertex_table[entry] == !0u32 {
            vertex_table[entry] = index;
        }

        *r = vertex_table[entry];
    }
}

fn generate_vertex_remap_impl<H: Hasher<u32>>(
    remap: &mut [u32],
    indices: Option<&[u32]>,
    index_count: usize,
    vertex_count: usize,
    hash: &mut H,
) -> usize {
    remap[..vertex_count].fill(!0u32);

    let mut table = vec![!0u32; hash_buckets(vertex_count)];

    let mut next_vertex = 0u32;

    for i in 0..index_count {
        let index = indices.map_or(i as u32, |ib| ib[i]);

        if remap[index as usize] != !0u32 {
            continue;
        }

        let entry = hash_lookup(&table, hash, &index, &!0u32);

        if table[entry] == !0u32 {
            table[entry] = index;
            remap[index as usize] = next_vertex;
            next_vertex += 1;
        } else {
            remap[index as usize] = remap[table[entry] as usize];
        }
    }

    next_vertex as usize
}

fn generate_shadow_buffer<H: Hasher<u32>>(destination: &mut [u32], indices: &[u32], vertex_count: usize, hash: &mut H) {
    let mut remap = vec![!0u32; vertex_count];

    let mut table = vec![!0u32; hash_buckets(vertex_count)];

    for (d, &index) in destination.iter_mut().zip(indices) {
        if remap[index as usize] == !0u32 {
            let entry = hash_lookup(&table, hash, &index, &!0u32);

            if table[entry] == !0u32 {
                table[entry] = index;
            }

            remap[index as usize] = table[entry];
        }

        *d = remap[index as usize];
    }
}

fn filter_index_buffer_impl(destination: &mut [u32], indices: &[u32], remap: &[u32]) -> usize {
    let mut hasher = TriangleKeyHasher;
    let empty = TriangleKey {
        a: !0u32,
        b: !0u32,
        c: !0u32,
    };

    let face_count = indices.len() / 3;
    let mut table = vec![empty; hash_buckets(face_count)];

    let mut write = 0;

    for tri in indices.as_chunks::<3>().0 {
        let [a, b, c] = *tri;

        let (mut ra, mut rb, mut rc) = (remap[a as usize], remap[b as usize], remap[c as usize]);

        // degenerate triangle
        if ra == rb || ra == rc || rb == rc {
            continue;
        }

        // canonicalize triangle wrt winding preserving rotation
        if rb < ra && rb < rc {
            // abc -> bca
            (ra, rb, rc) = (rb, rc, ra);
        } else if rc < ra && rc < rb {
            // abc -> cab
            (ra, rb, rc) = (rc, ra, rb);
        }

        let key = TriangleKey { a: ra, b: rb, c: rc };
        let entry = hash_lookup(&table, &mut hasher, &key, &empty);

        // duplicate triangle
        if table[entry].a != !0u32 {
            continue;
        }

        table[entry] = key;

        destination[write..write + 3].copy_from_slice(tri);
        write += 3;
    }

    write
}

fn validate_optional_indices(indices: Option<&[u32]>, index_count: usize, vertex_count: usize) -> Result<()> {
    match indices {
        Some(ib) => validate::indices(ib, index_count, vertex_count).map(|_| ()),
        None => ensure(
            index_count == vertex_count,
            Error::InvalidArgument("index_count must equal vertex_count without indices"),
        ),
    }
}

fn validate_streams<'a, 'b>(streams: &'a [Stream<'b>], stream_count: usize, vertex_count: usize) -> Result<&'a [Stream<'b>]> {
    ensure(
        stream_count > 0 && stream_count <= 16,
        Error::InvalidArgument("stream_count must be in 1..=16"),
    )?;
    let streams = streams.get(..stream_count).ok_or(Error::BufferTooSmall("streams"))?;

    for s in streams {
        ensure(
            s.size > 0 && s.size <= 256,
            Error::InvalidArgument("stream size must be in 1..=256"),
        )?;
        ensure(s.size <= s.stride, Error::InvalidArgument("stream size must be <= stride"))?;
        validate::bytes(s.data, vertex_count, s.size, s.stride)?;
    }

    Ok(streams)
}

/// Generates a vertex remap table from the vertex buffer and an optional index buffer and returns number of unique vertices.
/// All vertices that are binary equivalent map to the same (new) location, with no gaps in the resulting sequence.
/// `destination` must hold `vertex_count` entries; `indices` can be `None` if the input is unindexed.
pub fn generate_vertex_remap(
    destination: &mut [u32],
    indices: Option<&[u32]>,
    index_count: usize,
    vertices: &[u8],
    vertex_count: usize,
    vertex_size: usize,
) -> Result<usize> {
    validate_optional_indices(indices, index_count, vertex_count)?;
    ensure(
        vertex_size > 0 && vertex_size <= 256,
        Error::InvalidArgument("vertex_size must be in 1..=256"),
    )?;
    validate::bytes(vertices, vertex_count, vertex_size, vertex_size)?;
    let destination = validate::output(destination, vertex_count)?;

    let mut hasher = VertexHasher {
        vertices,
        vertex_size,
        vertex_stride: vertex_size,
    };

    Ok(generate_vertex_remap_impl(
        destination,
        indices,
        index_count,
        vertex_count,
        &mut hasher,
    ))
}

/// Same as [`generate_vertex_remap`] but compares vertex attributes split across multiple streams.
pub fn generate_vertex_remap_multi(
    destination: &mut [u32],
    indices: Option<&[u32]>,
    index_count: usize,
    vertex_count: usize,
    streams: &[Stream],
    stream_count: usize,
) -> Result<usize> {
    validate_optional_indices(indices, index_count, vertex_count)?;
    let streams = validate_streams(streams, stream_count, vertex_count)?;
    let destination = validate::output(destination, vertex_count)?;

    let mut hasher = VertexStreamHasher { streams };

    Ok(generate_vertex_remap_impl(
        destination,
        indices,
        index_count,
        vertex_count,
        &mut hasher,
    ))
}

/// Same as [`generate_vertex_remap`] but uses positions (float3, equal by value) plus an optional callback for remaining attributes.
pub fn generate_vertex_remap_custom(
    destination: &mut [u32],
    indices: Option<&[u32]>,
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    callback: Option<VertexEqualCallback>,
) -> Result<usize> {
    validate_optional_indices(indices, index_count, vertex_count)?;
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    let destination = validate::output(destination, vertex_count)?;

    let mut hasher = VertexCustomHasher {
        vertex_positions,
        vertex_stride_float: vertex_positions_stride / 4,
        callback,
    };

    Ok(generate_vertex_remap_impl(
        destination,
        indices,
        index_count,
        vertex_count,
        &mut hasher,
    ))
}

/// Generates vertex buffer from the source vertex buffer and remap table generated by [`generate_vertex_remap`].
/// `destination` must hold `unique_vertex_count * vertex_size` bytes. Use [`remap_vertex_buffer_in_place`] for C in-place mode.
pub fn remap_vertex_buffer(destination: &mut [u8], vertices: &[u8], vertex_count: usize, vertex_size: usize, remap: &[u32]) -> Result<()> {
    ensure(
        vertex_size > 0 && vertex_size <= 256,
        Error::InvalidArgument("vertex_size must be in 1..=256"),
    )?;
    validate::bytes(vertices, vertex_count, vertex_size, vertex_size)?;
    let remap = remap.get(..vertex_count).ok_or(Error::BufferTooSmall("remap"))?;
    ensure(
        remap.iter().all(|&r| r == !0u32 || (r as usize) < vertex_count),
        Error::IndexOutOfRange,
    )?;

    let required = remap
        .iter()
        .filter(|&&r| r != !0u32)
        .map(|&r| (r as usize + 1) * vertex_size)
        .max()
        .unwrap_or(0);
    ensure(destination.len() >= required, Error::BufferTooSmall("destination"))?;

    for (i, &r) in remap.iter().enumerate() {
        if r != !0u32 {
            let (dst, src) = (r as usize * vertex_size, i * vertex_size);
            destination[dst..dst + vertex_size].copy_from_slice(&vertices[src..src + vertex_size]);
        }
    }

    Ok(())
}

/// In-place variant of [`remap_vertex_buffer`] (C: `destination == vertices`).
pub fn remap_vertex_buffer_in_place(vertices: &mut [u8], vertex_count: usize, vertex_size: usize, remap: &[u32]) -> Result<()> {
    validate::bytes(vertices, vertex_count, vertex_size, vertex_size)?;
    let vertices_copy = vertices[..vertex_count * vertex_size].to_vec();
    remap_vertex_buffer(vertices, &vertices_copy, vertex_count, vertex_size, remap)
}

/// Generates index buffer from the source index buffer and remap table generated by [`generate_vertex_remap`].
/// `indices` can be `None` if the input is unindexed. Use [`remap_index_buffer_in_place`] for C in-place mode.
pub fn remap_index_buffer(destination: &mut [u32], indices: Option<&[u32]>, index_count: usize, remap: &[u32]) -> Result<()> {
    if let Some(ib) = indices {
        ensure(ib.len() >= index_count, Error::BufferTooSmall("indices"))?;
    }
    let destination = validate::output(destination, index_count)?;

    for (i, d) in destination.iter_mut().enumerate() {
        let index = indices.map_or(i as u32, |ib| ib[i]);
        let r = *remap.get(index as usize).ok_or(Error::IndexOutOfRange)?;
        ensure(r != !0u32, Error::InvalidArgument("remap entry is unused"))?;

        *d = r;
    }

    Ok(())
}

/// In-place variant of [`remap_index_buffer`] (C: `destination == indices`).
pub fn remap_index_buffer_in_place(indices: &mut [u32], index_count: usize, remap: &[u32]) -> Result<()> {
    let indices = validate::output(indices, index_count)?;
    // validate everything before writing so that errors leave the buffer intact
    for &index in indices.iter() {
        let r = *remap.get(index as usize).ok_or(Error::IndexOutOfRange)?;
        ensure(r != !0u32, Error::InvalidArgument("remap entry is unused"))?;
    }

    for index in indices.iter_mut() {
        *index = remap[*index as usize];
    }

    Ok(())
}

/// Removes degenerate and duplicate triangles, comparing the first `vertex_size` bytes of each vertex (spaced `vertex_stride` apart).
/// Returns the number of indices written; triangles keep their original rotation and order.
pub fn filter_index_buffer(
    destination: &mut [u32],
    indices: &[u32],
    index_count: usize,
    vertices: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    vertex_stride: usize,
) -> Result<usize> {
    validate::triangles(index_count)?;
    ensure(
        vertex_size > 0 && vertex_size <= 256,
        Error::InvalidArgument("vertex_size must be in 1..=256"),
    )?;
    ensure(
        vertex_size <= vertex_stride,
        Error::InvalidArgument("vertex_size must be <= vertex_stride"),
    )?;
    let indices = validate::indices(indices, index_count, vertex_count)?;
    validate::bytes(vertices, vertex_count, vertex_size, vertex_stride)?;
    let destination = validate::output(destination, index_count)?;

    let mut hasher = VertexHasher {
        vertices,
        vertex_size,
        vertex_stride,
    };

    let mut remap = vec![0u32; vertex_count];
    generate_vertex_remap_impl(&mut remap, Some(indices), index_count, vertex_count, &mut hasher);

    Ok(filter_index_buffer_impl(destination, indices, &remap))
}

/// Same as [`filter_index_buffer`] but compares vertex attributes split across multiple streams.
pub fn filter_index_buffer_multi(
    destination: &mut [u32],
    indices: &[u32],
    index_count: usize,
    vertex_count: usize,
    streams: &[Stream],
    stream_count: usize,
) -> Result<usize> {
    validate::triangles(index_count)?;
    let streams = validate_streams(streams, stream_count, vertex_count)?;
    let indices = validate::indices(indices, index_count, vertex_count)?;
    let destination = validate::output(destination, index_count)?;

    let mut hasher = VertexStreamHasher { streams };

    let mut remap = vec![0u32; vertex_count];
    generate_vertex_remap_impl(&mut remap, Some(indices), index_count, vertex_count, &mut hasher);

    Ok(filter_index_buffer_impl(destination, indices, &remap))
}

/// Generates index buffer that can be used for more efficient rendering when only a subset of the vertex attributes is necessary.
/// All vertices that are binary equivalent (wrt first `vertex_size` bytes) map to the first vertex in the original vertex buffer.
pub fn generate_shadow_index_buffer(
    destination: &mut [u32],
    indices: &[u32],
    index_count: usize,
    vertices: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    vertex_stride: usize,
) -> Result<()> {
    validate::triangles(index_count)?;
    ensure(
        vertex_size > 0 && vertex_size <= 256,
        Error::InvalidArgument("vertex_size must be in 1..=256"),
    )?;
    ensure(
        vertex_size <= vertex_stride,
        Error::InvalidArgument("vertex_size must be <= vertex_stride"),
    )?;
    let indices = validate::indices(indices, index_count, vertex_count)?;
    validate::bytes(vertices, vertex_count, vertex_size, vertex_stride)?;
    let destination = validate::output(destination, index_count)?;

    let mut hasher = VertexHasher {
        vertices,
        vertex_size,
        vertex_stride,
    };

    generate_shadow_buffer(destination, indices, vertex_count, &mut hasher);
    Ok(())
}

/// Same as [`generate_shadow_index_buffer`] but compares vertex attributes split across multiple streams.
pub fn generate_shadow_index_buffer_multi(
    destination: &mut [u32],
    indices: &[u32],
    index_count: usize,
    vertex_count: usize,
    streams: &[Stream],
    stream_count: usize,
) -> Result<()> {
    validate::triangles(index_count)?;
    let streams = validate_streams(streams, stream_count, vertex_count)?;
    let indices = validate::indices(indices, index_count, vertex_count)?;
    let destination = validate::output(destination, index_count)?;

    let mut hasher = VertexStreamHasher { streams };

    generate_shadow_buffer(destination, indices, vertex_count, &mut hasher);
    Ok(())
}

/// Generates a remap table that maps all vertices with the same position (by value, -0 == 0) to the first such vertex.
pub fn generate_position_remap(
    destination: &mut [u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> Result<()> {
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    let destination = validate::output(destination, vertex_count)?;

    let mut hasher = VertexCustomHasher {
        vertex_positions,
        vertex_stride_float: vertex_positions_stride / 4,
        callback: None,
    };

    let mut table = vec![!0u32; hash_buckets(vertex_count)];

    for (i, d) in destination.iter_mut().enumerate() {
        let entry = hash_lookup(&table, &mut hasher, &(i as u32), &!0u32);

        if table[entry] == !0u32 {
            table[entry] = i as u32;
        }

        *d = table[entry];
    }

    Ok(())
}

/// Generates index buffer that can be used as a geometry shader input with triangle adjacency topology.
/// `destination` must hold `index_count * 2` entries.
pub fn generate_adjacency_index_buffer(
    destination: &mut [u32],
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> Result<()> {
    validate::triangles(index_count)?;
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    let indices = validate::indices(indices, index_count, vertex_count)?;
    let destination = validate::output(destination, index_count * 2)?;

    const NEXT: [usize; 4] = [1, 2, 0, 1];

    // build position remap: for each vertex, which other (canonical) vertex does it map to?
    let mut remap = vec![0u32; vertex_count];
    build_position_remap(&mut remap, vertex_positions, vertex_count, vertex_positions_stride);

    // build edge set; this stores all triangle edges but we can look these up by any other wedge
    let mut edge_hasher = EdgeHasher { remap: &remap };

    let edge_table_size = hash_buckets(index_count);
    let mut edge_table = vec![!0u64; edge_table_size];
    let mut edge_vertex_table = vec![!0u32; edge_table_size];

    for tri in indices.as_chunks::<3>().0 {
        for e in 0..3 {
            let i0 = tri[e];
            let i1 = tri[NEXT[e]];
            let i2 = tri[NEXT[e + 1]];

            let edge = (u64::from(i0) << 32) | u64::from(i1);
            let entry = hash_lookup(&edge_table, &mut edge_hasher, &edge, &!0u64);

            if edge_table[entry] == !0u64 {
                edge_table[entry] = edge;

                // store vertex opposite to the edge
                edge_vertex_table[entry] = i2;
            }
        }
    }

    // build resulting index buffer: 6 indices for each input triangle
    for (tri, patch) in indices.as_chunks::<3>().0.iter().zip(destination.as_chunks_mut::<6>().0) {
        for e in 0..3 {
            let i0 = tri[e];
            let i1 = tri[NEXT[e]];

            // note: this refers to the opposite edge!
            let edge = (u64::from(i1) << 32) | u64::from(i0);
            let oppe = hash_lookup(&edge_table, &mut edge_hasher, &edge, &!0u64);

            patch[e * 2] = i0;
            patch[e * 2 + 1] = if edge_table[oppe] == !0u64 { i0 } else { edge_vertex_table[oppe] };
        }
    }

    Ok(())
}

/// Generates index buffer that can be used for PN-AEN tessellation with crack-free displacement.
/// `destination` must hold `index_count * 4` entries.
pub fn generate_tessellation_index_buffer(
    destination: &mut [u32],
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> Result<()> {
    validate::triangles(index_count)?;
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    let indices = validate::indices(indices, index_count, vertex_count)?;
    let destination = validate::output(destination, index_count * 4)?;

    const NEXT: [usize; 3] = [1, 2, 0];

    // build position remap: for each vertex, which other (canonical) vertex does it map to?
    let mut remap = vec![0u32; vertex_count];
    build_position_remap(&mut remap, vertex_positions, vertex_count, vertex_positions_stride);

    // build edge set; this stores all triangle edges but we can look these up by any other wedge
    let mut edge_hasher = EdgeHasher { remap: &remap };

    let mut edge_table = vec![!0u64; hash_buckets(index_count)];

    for tri in indices.as_chunks::<3>().0 {
        for e in 0..3 {
            let i0 = tri[e];
            let i1 = tri[NEXT[e]];

            let edge = (u64::from(i0) << 32) | u64::from(i1);
            let entry = hash_lookup(&edge_table, &mut edge_hasher, &edge, &!0u64);

            if edge_table[entry] == !0u64 {
                edge_table[entry] = edge;
            }
        }
    }

    // build resulting index buffer: 12 indices for each input triangle
    for (tri, patch) in indices.as_chunks::<3>().0.iter().zip(destination.as_chunks_mut::<12>().0) {
        for e in 0..3 {
            let i0 = tri[e];
            let i1 = tri[NEXT[e]];

            // note: this refers to the opposite edge!
            let edge = (u64::from(i1) << 32) | u64::from(i0);
            let mut oppe = edge_table[hash_lookup(&edge_table, &mut edge_hasher, &edge, &!0u64)];

            // use the same edge if opposite edge doesn't exist (border)
            oppe = if oppe == !0u64 { edge } else { oppe };

            // triangle index (0, 1, 2)
            patch[e] = i0;

            // opposite edge (3, 4; 5, 6; 7, 8)
            patch[3 + e * 2] = oppe as u32;
            patch[3 + e * 2 + 1] = (oppe >> 32) as u32;

            // dominant vertex (9, 10, 11)
            patch[9 + e] = remap[i0 as usize];
        }
    }

    Ok(())
}

/// Generates index buffer for rendering with flat shading via provoking vertex; returns the number of entries in `reorder`.
/// `destination` must hold `index_count` entries, `reorder` must hold `vertex_count + index_count / 3` entries.
pub fn generate_provoking_index_buffer(
    destination: &mut [u32],
    reorder: &mut [u32],
    indices: &[u32],
    index_count: usize,
    vertex_count: usize,
) -> Result<usize> {
    validate::triangles(index_count)?;
    let indices = validate::indices(indices, index_count, vertex_count)?;
    let destination = validate::output(destination, index_count)?;
    let reorder = validate::output(reorder, vertex_count + index_count / 3)?;

    let mut remap = vec![!0u32; vertex_count];

    // compute vertex valence; this is used to prioritize least used corner
    // note: we use 8-bit counters for performance; for outlier vertices the valence is incorrect but that just affects the heuristic
    let mut valence = vec![0u8; vertex_count];

    for &index in indices {
        valence[index as usize] = valence[index as usize].wrapping_add(1);
    }

    let mut reorder_offset = 0usize;

    // assign provoking vertices; leave the rest for the next pass
    for (tri, dst) in indices.as_chunks::<3>().0.iter().zip(destination.as_chunks_mut::<3>().0) {
        let [mut a, mut b, mut c] = *tri;

        // try to rotate triangle such that provoking vertex hasn't been seen before
        // if multiple vertices are new, prioritize the one with least valence
        // this reduces the risk that a future triangle will have all three vertices seen
        let va = if remap[a as usize] == !0u32 {
            u32::from(valence[a as usize])
        } else {
            !0u32
        };
        let vb = if remap[b as usize] == !0u32 {
            u32::from(valence[b as usize])
        } else {
            !0u32
        };
        let vc = if remap[c as usize] == !0u32 {
            u32::from(valence[c as usize])
        } else {
            !0u32
        };

        if vb != !0u32 && vb <= va && vb <= vc {
            // abc -> bca
            (a, b, c) = (b, c, a);
        } else if vc != !0u32 && vc <= va && vc <= vb {
            // abc -> cab
            (a, b, c) = (c, a, b);
        }

        let newidx = reorder_offset as u32;

        // now remap[a] = ~0u or all three vertices are old
        // recording remap[a] makes it possible to remap future references to the same index, conserving space
        if remap[a as usize] == !0u32 {
            remap[a as usize] = newidx;
        }

        // we need to clone the provoking vertex to get a unique index
        // if all three are used the choice is arbitrary since no future triangle will be able to reuse any of these
        reorder[reorder_offset] = a;
        reorder_offset += 1;

        // note: first vertex is final, the other two will be fixed up in next pass
        *dst = [newidx, b, c];

        // update vertex valences for corner heuristic
        valence[a as usize] = valence[a as usize].wrapping_sub(1);
        valence[b as usize] = valence[b as usize].wrapping_sub(1);
        valence[c as usize] = valence[c as usize].wrapping_sub(1);
    }

    // remap or clone non-provoking vertices (skipping provoking vertices)
    for dst in destination.as_chunks_mut::<3>().0 {
        for index in &mut dst[1..] {
            if remap[*index as usize] == !0u32 {
                // we haven't seen the vertex before as a provoking vertex
                // to maintain the reference to the original vertex we need to clone it
                remap[*index as usize] = reorder_offset as u32;
                reorder[reorder_offset] = *index;
                reorder_offset += 1;
            }

            *index = remap[*index as usize];
        }
    }

    Ok(reorder_offset)
}
