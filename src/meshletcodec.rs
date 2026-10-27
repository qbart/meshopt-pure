//! Meshlet codec; only the scalar decoder of the C++ implementation is ported.

use crate::error::{Error, Result, ensure};
use crate::index::IndexType;

type EdgeFifo8 = [[u32; 2]; 8];

mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u32 {}
}

/// Triangle output element for [`decode_meshlet`]: `u8` stores 3 bytes per triangle, `u32` stores packed `a | b << 8 | c << 16`.
pub trait MeshletTriangleType: Copy + sealed::Sealed {
    /// Number of elements per triangle (3 for `u8`, 1 for `u32`).
    const PER_TRIANGLE: usize;

    #[doc(hidden)]
    fn write(triangles: &mut [Self], i: usize, fifo: u32);
}

impl MeshletTriangleType for u8 {
    const PER_TRIANGLE: usize = 3;

    fn write(triangles: &mut [Self], i: usize, fifo: u32) {
        triangles[i * 3] = (fifo >> 8) as u8;
        triangles[i * 3 + 1] = (fifo >> 16) as u8;
        triangles[i * 3 + 2] = (fifo >> 24) as u8;
    }
}

impl MeshletTriangleType for u32 {
    const PER_TRIANGLE: usize = 1;

    fn write(triangles: &mut [Self], i: usize, fifo: u32) {
        // output triangle is stored without extra edge vertex (0xcbac => 0xcba)
        triangles[i] = fifo >> 8;
    }
}

fn rotate_triangle(a: u32, b: u32, c: u32) -> usize {
    if a > b && a > c {
        1
    } else if b > c {
        2
    } else {
        0
    }
}

fn get_edge_fifo8(fifo: &EdgeFifo8, a: u32, b: u32, c: u32, offset: usize) -> i32 {
    for i in 0..8 {
        let index = offset.wrapping_sub(1 + i) & 7;

        let [e0, e1] = fifo[index];

        if e0 == a && e1 == b {
            return (i as i32) << 2;
        }
        if e0 == b && e1 == c {
            return ((i as i32) << 2) | 1;
        }
        if e0 == c && e1 == a {
            return ((i as i32) << 2) | 2;
        }
    }

    -1
}

fn push_edge_fifo8(fifo: &mut EdgeFifo8, a: u32, b: u32, offset: &mut usize) {
    fifo[*offset] = [a, b];
    *offset = (*offset + 1) & 7;
}

fn encode_triangles(codes: &mut [u8], extra: &mut [u8], triangles: &[u8], triangle_count: usize) -> usize {
    let mut edgefifo: EdgeFifo8 = [[!0u32; 2]; 8];

    let mut edgefifooffset = 0;

    let mut next = 0u32;

    // 4-bit triangle codes give us 16 options that we use as follows:
    // 3*2 edge reuse (2 edges * 3 last triangles) * 2 next/explicit = 12 options
    // 4 remaining options = next bits; 000, 001, 011, 111.
    // triangles are rotated to make next bits line up.
    codes[..triangle_count.div_ceil(2)].fill(0);

    const ROTATIONS: [usize; 5] = [0, 1, 2, 0, 1];

    let mut extra_size = 0;
    let mut push_extra = |v: u32| {
        extra[extra_size] = v as u8;
        extra_size += 1;
    };

    for (i, tri) in triangles.as_chunks::<3>().0.iter().take(triangle_count).enumerate() {
        let tri = [u32::from(tri[0]), u32::from(tri[1]), u32::from(tri[2])];

        let fer = get_edge_fifo8(&edgefifo, tri[0], tri[1], tri[2], edgefifooffset);

        if fer >= 0 && (fer >> 2) < 6 {
            // note: getEdgeFifo8 implicitly rotates triangles by matching a/b to existing edge
            let order = &ROTATIONS[(fer & 3) as usize..];

            let (a, b, c) = (tri[order[0]], tri[order[1]], tri[order[2]]);

            let fec = if c == next {
                next += 1;
                0
            } else {
                1
            };

            let code = (fer >> 2) as u32 * 2 + fec;

            codes[i / 2] |= (code << ((i & 1) * 4)) as u8;

            if fec != 0 {
                push_extra(c);
            }

            push_edge_fifo8(&mut edgefifo, c, b, &mut edgefifooffset);
            push_edge_fifo8(&mut edgefifo, a, c, &mut edgefifooffset);
        } else {
            // rotate triangles to minimize the need for extra vertices
            let rotation = rotate_triangle(tri[0], tri[1], tri[2]);
            let order = &ROTATIONS[rotation..];

            let (a, b, c) = (tri[order[0]], tri[order[1]], tri[order[2]]);

            // fe must be continuous: once a vertex is encoded with next, further vertices must also be encoded with next
            let fea = if a == next && b == next + 1 && c == next + 2 {
                next += 1;
                0
            } else {
                1
            };
            let feb = if b == next && c == next + 1 {
                next += 1;
                0
            } else {
                1
            };
            let fec = if c == next {
                next += 1;
                0
            } else {
                1
            };

            let code = 12 + (fea + feb + fec);

            codes[i / 2] |= (code << ((i & 1) * 4)) as u8;

            if fea != 0 {
                push_extra(a);
            }
            if feb != 0 {
                push_extra(b);
            }
            if fec != 0 {
                push_extra(c);
            }

            push_edge_fifo8(&mut edgefifo, c, b, &mut edgefifooffset);
            push_edge_fifo8(&mut edgefifo, a, c, &mut edgefifooffset);
        }
    }

    extra_size
}

fn encode_vertices(ctrl: &mut [u8], data: &mut [u8], vertices: &[u32], vertex_count: usize) -> usize {
    // grouped varint, 2 bit per value to indicate 0/1/2/3 byte deltas, with per-group 4-byte fallback
    ctrl[..vertex_count.div_ceil(4)].fill(0);

    let mut data_size = 0;

    let mut last = !0u32;

    for i in (0..vertex_count).step_by(4) {
        let mut gv = [0u32; 4];

        for (k, g) in gv.iter_mut().enumerate().take(vertex_count - i) {
            let d = vertices[i + k].wrapping_sub(last).wrapping_sub(1);
            let v = (d << 1) ^ ((d as i32) >> 31) as u32;

            *g = v;
            last = vertices[i + k];
        }

        // if any value needs 4 bytes, or if *all* values need 3 bytes, we use 4 bytes for all values
        // this allows us to encode most 3-byte deltas with 3 bytes which saves space overall
        let use4 = (gv[0] | gv[1] | gv[2] | gv[3]) > 0xffffff || gv.iter().all(|&g| g > 0xffff);

        for (k, &v) in gv.iter().enumerate() {
            // 0/1/2/3 bytes per value, or all 4 values use 4 bytes
            let code = if use4 {
                3
            } else if v == 0 {
                0
            } else if v < 256 {
                1
            } else if v < 65536 {
                2
            } else {
                3
            };

            let len = if use4 { 4 } else { code };
            data[data_size..data_size + len].copy_from_slice(&v.to_le_bytes()[..len]);
            data_size += len;

            // split low and high bits into two nibbles for better packing
            ctrl[i / 4] |= (((code & 1) << k) | ((code >> 1) << (k + 4))) as u8;
        }
    }

    data_size
}

fn decode_triangles<T: MeshletTriangleType>(
    triangles: &mut [T],
    codes: &[u8],
    buffer: &[u8],
    mut extra: usize,
    bound: usize,
    triangle_count: usize,
) -> Option<usize> {
    let mut next = 0u32;
    let mut fifo = [0u32; 3]; // two edge fifo entries in one uint: 0xcbac

    // branchlessly read next or extra vertex and advance pointers
    let mut read = |extra: &mut usize, ec: bool| -> u32 {
        let e = u32::from(buffer[*extra]);
        let v = if ec { e } else { next };
        *extra += usize::from(ec);
        next += u32::from(!ec);
        v
    };

    for i in 0..triangle_count {
        if extra > bound {
            return None;
        }

        let code = u32::from(codes[i / 2] >> ((i & 1) * 4)) & 0xf;

        let tri = if code < 12 {
            // reuse
            let mut edge = fifo[(code / 4) as usize];
            edge >>= (code << 3) & 16; // shift by 16 if bit 1 is set (odd edge for each triangle)

            // 0-1 extra vertices
            let c = read(&mut extra, code & 1 != 0);

            // repack triangle into edge format (0xcbac)
            ((edge & 0xff) << 16) | (edge & 0xff00) | c | (c << 24)
        } else {
            // restart; 0-3 extra vertices
            let a = read(&mut extra, code > 12);
            let b = read(&mut extra, code > 13);
            let c = read(&mut extra, code > 14);

            // repack triangle into edge format (0xcbac)
            c | (a << 8) | (b << 16) | (c << 24)
        };

        T::write(triangles, i, tri);

        fifo[2] = fifo[1];
        fifo[1] = fifo[0];
        fifo[0] = tri;
    }

    Some(extra)
}

fn decode_vertices<V: IndexType>(
    vertices: &mut [V],
    ctrl: &[u8],
    buffer: &[u8],
    mut data: usize,
    bound: usize,
    vertex_count: usize,
) -> Option<usize> {
    let mut last = !0u32;

    for i in (0..vertex_count).step_by(4) {
        if data > bound {
            return None;
        }

        let code4 = ctrl[i / 4];

        for k in 0..4 {
            let code = ((code4 >> k) & 1) | ((code4 >> (k + 3)) & 2);
            let length = if code4 == 0xff { 4 } else { usize::from(code) };

            // branchlessly read up to 4 bytes
            let mask = if length == 4 { !0u32 } else { (1u32 << (8 * length)) - 1 };
            let v = u32::from_le_bytes([buffer[data], buffer[data + 1], buffer[data + 2], buffer[data + 3]]) & mask;

            // unzigzag + 1
            let d = (v >> 1) ^ (-((v & 1) as i32)) as u32;
            let r = last.wrapping_add(d).wrapping_add(1);

            if i + k < vertex_count {
                vertices[i + k] = V::from_u32(r);
            }

            data += length;
            last = r;
        }
    }

    Some(data)
}

/// Worst case size of [`encode_meshlet`] output.
pub fn encode_meshlet_bound(max_vertices: usize, max_triangles: usize) -> usize {
    let codes_size = max_triangles.div_ceil(2);
    let extra_size = max_triangles * 3;

    let ctrl_size = max_vertices.div_ceil(4);
    let data_size = max_vertices.div_ceil(4) * 16; // worst case: 16 bytes per vertex group

    let gap_size = 16usize.saturating_sub(codes_size + ctrl_size);

    codes_size + extra_size + ctrl_size + data_size + gap_size
}

/// Meshlet encoder.
/// Encodes meshlet data into an array of bytes that is generally smaller and compresses better compared to original.
/// Returns encoded data size on success, 0 on error; the only error condition is if `buffer` doesn't have enough space.
/// `vertices` may be empty, in which case `vertex_count` must be 0; `vertex_count` and `triangle_count` must be <= 256.
pub fn encode_meshlet(
    buffer: &mut [u8],
    buffer_size: usize,
    vertices: &[u32],
    vertex_count: usize,
    triangles: &[u8],
    triangle_count: usize,
) -> Result<usize> {
    ensure(
        triangle_count <= 256 && vertex_count <= 256,
        Error::InvalidArgument("vertex_count and triangle_count must be <= 256"),
    )?;
    let buffer = buffer.get_mut(..buffer_size).ok_or(Error::BufferTooSmall("buffer"))?;
    let vertices = vertices.get(..vertex_count).ok_or(Error::BufferTooSmall("vertices"))?;
    let triangles = triangles.get(..triangle_count * 3).ok_or(Error::BufferTooSmall("triangles"))?;

    // 4 bits per triangle + up to three bytes of extra data
    let mut codes = [0u8; 256 / 2];
    let mut extra = [0u8; 256 * 3];
    let codes_size = triangle_count.div_ceil(2);
    let extra_size = encode_triangles(&mut codes, &mut extra, triangles, triangle_count);

    // 2 bits per vertex + up to 4 bytes of actual data
    let mut ctrl = [0u8; 256 / 4];
    let mut data = [0u8; 256 * 4];
    let ctrl_size = vertex_count.div_ceil(4);
    let data_size = encode_vertices(&mut ctrl, &mut data, vertices, vertex_count);

    // we need to ensure that up to 16 bytes after extra+data are available for SIMD decoding
    // to minimize overhead, we place fixed-size codes+control at the end of the buffer
    let gap_size = 16usize.saturating_sub(codes_size + ctrl_size);

    let result = codes_size + extra_size + ctrl_size + data_size + gap_size;

    if result > buffer_size {
        return Ok(0);
    }

    let mut offset = 0;
    let mut write = |src: &[u8]| {
        buffer[offset..offset + src.len()].copy_from_slice(src);
        offset += src.len();
    };

    // variable-size data first
    write(&data[..data_size]);
    write(&extra[..extra_size]);

    // gap (for accelerated decoding) separates variable-size and fixed-size data
    write(&[0u8; 16][..gap_size]);

    // fixed-size data last; it can be located from buffer end during decoding
    write(&ctrl[..ctrl_size]);
    write(&codes[..codes_size]);

    Ok(result)
}

/// Shared layout parsing; returns `(codes, ctrl, bound)` offsets.
fn meshlet_layout(vertex_count: usize, triangle_count: usize, buffer_size: usize) -> Result<(usize, usize, usize)> {
    ensure(
        triangle_count <= 256 && vertex_count <= 256,
        Error::InvalidArgument("vertex_count and triangle_count must be <= 256"),
    )?;

    // layout must match encoding
    let codes_size = triangle_count.div_ceil(2);
    let ctrl_size = vertex_count.div_ceil(4);
    let gap_size = 16usize.saturating_sub(codes_size + ctrl_size);

    if buffer_size < codes_size + ctrl_size + gap_size {
        return Err(Error::Decode(-2));
    }

    let codes = buffer_size - codes_size;
    let ctrl = codes - ctrl_size;

    // gap ensures we have at least 16 bytes available after bound; this allows SIMD decoders to over-read safely
    let bound = ctrl - gap_size;

    Ok((codes, ctrl, bound))
}

fn decode_meshlet_impl<V: IndexType, T: MeshletTriangleType>(
    vertices: &mut [V],
    vertex_count: usize,
    triangles: &mut [T],
    triangle_count: usize,
    buffer: &[u8],
    buffer_size: usize,
) -> Result<()> {
    let buffer = buffer.get(..buffer_size).ok_or(Error::BufferTooSmall("buffer"))?;
    let (codes, ctrl, bound) = meshlet_layout(vertex_count, triangle_count, buffer_size)?;
    let vertices = vertices.get_mut(..vertex_count).ok_or(Error::BufferTooSmall("vertices"))?;
    let triangles = triangles
        .get_mut(..triangle_count * T::PER_TRIANGLE)
        .ok_or(Error::BufferTooSmall("triangles"))?;

    let data = decode_vertices(vertices, &buffer[ctrl..codes], buffer, 0, bound, vertex_count).ok_or(Error::Decode(-2))?;
    let data = decode_triangles(triangles, &buffer[codes..], buffer, data, bound, triangle_count).ok_or(Error::Decode(-2))?;

    if data == bound { Ok(()) } else { Err(Error::Decode(-3)) }
}

/// Meshlet decoder.
/// Decodes meshlet data from an array of bytes generated by [`encode_meshlet`]; returns `Err(Decode(code))` on failure.
/// `V` selects 16/32-bit vertex references (C `vertex_size`), `T` selects 8-bit triangle indices or 32-bit packed triangles (C `triangle_size`).
/// `vertex_count`, `triangle_count` must match those used during encoding exactly; `buffer_size` must equal the encoded size.
/// The decoder is safe to use for untrusted input, but it may produce garbage data.
pub fn decode_meshlet<V: IndexType, T: MeshletTriangleType>(
    vertices: &mut [V],
    vertex_count: usize,
    triangles: &mut [T],
    triangle_count: usize,
    buffer: &[u8],
    buffer_size: usize,
) -> Result<()> {
    decode_meshlet_impl(vertices, vertex_count, triangles, triangle_count, buffer, buffer_size)
}

/// "Raw" meshlet decoder with 32-bit vertex references and 32-bit packed triangles; see [`decode_meshlet`].
pub fn decode_meshlet_raw(
    vertices: &mut [u32],
    vertex_count: usize,
    triangles: &mut [u32],
    triangle_count: usize,
    buffer: &[u8],
    buffer_size: usize,
) -> Result<()> {
    decode_meshlet_impl(vertices, vertex_count, triangles, triangle_count, buffer, buffer_size)
}
