// This work is based on:
// Fabian Giesen. Simple lossless index buffer compression & follow-up. 2013
// Conor Stokes. Vertex Cache Optimised Index Buffer Compression. 2014

use core::sync::atomic::{AtomicI32, Ordering};

use crate::error::{Error, Result, ensure};
use crate::index::IndexType;
use crate::validate;

const INDEX_HEADER: u8 = 0xe0;
const SEQUENCE_HEADER: u8 = 0xd0;

static ENCODE_INDEX_VERSION: AtomicI32 = AtomicI32::new(1);
const DECODE_INDEX_VERSION: i32 = 1;

type VertexFifo = [u32; 16];
type EdgeFifo = [[u32; 2]; 16];

const CODE_AUX_ENCODING_TABLE: [u8; 16] = [
    0x00, 0x76, 0x87, 0x56, 0x67, 0x78, 0xa9, 0x86, 0x65, 0x89, 0x68, 0x98, 0x01, 0x69, //
    0, 0, // last two entries aren't used for encoding
];

fn rotate_triangle(_a: u32, b: u32, c: u32, next: u32) -> usize {
    if b == next {
        1
    } else if c == next {
        2
    } else {
        0
    }
}

fn get_edge_fifo(fifo: &EdgeFifo, a: u32, b: u32, c: u32, offset: usize) -> i32 {
    for i in 0..16 {
        let index = offset.wrapping_sub(1 + i) & 15;

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

fn push_edge_fifo(fifo: &mut EdgeFifo, a: u32, b: u32, offset: &mut usize) {
    fifo[*offset] = [a, b];
    *offset = (*offset + 1) & 15;
}

fn get_vertex_fifo(fifo: &VertexFifo, v: u32, offset: usize) -> i32 {
    for i in 0..16 {
        let index = offset.wrapping_sub(1 + i) & 15;

        if fifo[index] == v {
            return i as i32;
        }
    }

    -1
}

fn push_vertex_fifo(fifo: &mut VertexFifo, v: u32, offset: &mut usize, cond: bool) {
    fifo[*offset] = v;
    *offset = (*offset + usize::from(cond)) & 15;
}

/// Writes `v` as up to 5 7-bit groups at `data[*pos..]`; caller guarantees space.
pub(crate) fn encode_vbyte(data: &mut [u8], pos: &mut usize, mut v: u32) {
    // encode 32-bit value in up to 5 7-bit groups
    loop {
        data[*pos] = ((v & 127) | if v > 127 { 128 } else { 0 }) as u8;
        *pos += 1;
        v >>= 7;

        if v == 0 {
            break;
        }
    }
}

/// Reads up to 5 bytes from `data[*pos..]`; caller guarantees they are in bounds.
pub(crate) fn decode_vbyte(data: &[u8], pos: &mut usize) -> u32 {
    let lead = data[*pos];
    *pos += 1;

    // fast path: single byte
    if lead < 128 {
        return u32::from(lead);
    }

    // slow path: up to 4 extra bytes
    // note that this loop always terminates, which is important for malformed data
    let mut result = u32::from(lead & 127);
    let mut shift = 7;

    for _ in 0..4 {
        let group = data[*pos];
        *pos += 1;
        result |= u32::from(group & 127).wrapping_shl(shift);
        shift += 7;

        if group < 128 {
            break;
        }
    }

    result
}

fn encode_index(data: &mut [u8], pos: &mut usize, index: u32, last: u32) {
    let d = index.wrapping_sub(last);
    let v = (d << 1) ^ ((d as i32) >> 31) as u32;

    encode_vbyte(data, pos, v);
}

fn decode_index(data: &[u8], pos: &mut usize, last: u32) -> u32 {
    let v = decode_vbyte(data, pos);
    let d = (v >> 1) ^ (-((v & 1) as i32)) as u32;

    last.wrapping_add(d)
}

fn get_code_aux_index(v: u8, table: &[u8; 16]) -> i32 {
    table.iter().position(|&t| t == v).map_or(-1, |i| i as i32)
}

fn write_triangle<T: IndexType>(destination: &mut [T], offset: &mut usize, a: u32, b: u32, c: u32) {
    destination[*offset] = T::from_u32(a);
    destination[*offset + 1] = T::from_u32(b);
    destination[*offset + 2] = T::from_u32(c);
    *offset += 3;
}

/// Encodes index data into an array of bytes that is generally much smaller (<1.5 bytes/triangle) and compresses better (<1 bytes/triangle) compared to original.
/// Input index buffer must represent a triangle list.
/// Returns encoded data size on success, 0 on error; the only error condition is if `buffer` doesn't have enough space.
/// For maximum efficiency the index buffer being encoded has to be optimized for vertex cache first.
pub fn encode_index_buffer<T: IndexType>(buffer: &mut [u8], buffer_size: usize, indices: &[T], index_count: usize) -> Result<usize> {
    validate::triangles(index_count)?;
    let buffer = buffer.get_mut(..buffer_size).ok_or(Error::BufferTooSmall("buffer"))?;
    let indices = indices.get(..index_count).ok_or(Error::BufferTooSmall("indices"))?;

    // the minimum valid encoding is header, 1 byte per triangle and a 16-byte codeaux table
    if buffer_size < 1 + index_count / 3 + 16 {
        return Ok(0);
    }

    let version = ENCODE_INDEX_VERSION.load(Ordering::Relaxed);

    buffer[0] = INDEX_HEADER | version as u8;

    let mut edgefifo: EdgeFifo = [[!0u32; 2]; 16];
    let mut vertexfifo: VertexFifo = [!0u32; 16];

    let mut edgefifooffset = 0usize;
    let mut vertexfifooffset = 0usize;

    let mut next = 0u32;
    let mut last = 0u32;

    let mut code = 1usize;
    let mut data = code + index_count / 3;
    let data_safe_end = buffer_size - 16;

    let fecmax = if version >= 1 { 13 } else { 15 };

    const ROTATIONS: [usize; 5] = [0, 1, 2, 0, 1];

    // use static encoding table; it's possible to pack the result and then build an optimal table and repack
    // for now we keep it simple and use the table that has been generated based on symbol frequency on a training mesh set
    let codeaux_table = &CODE_AUX_ENCODING_TABLE;

    for tri in indices.as_chunks::<3>().0 {
        // make sure we have enough space to write a triangle
        // each triangle writes at most 16 bytes: 1b for codeaux and 5b for each free index
        // after this we can be sure we can write without extra bounds checks
        if data > data_safe_end {
            return Ok(0);
        }

        let tri = [tri[0].to_u32(), tri[1].to_u32(), tri[2].to_u32()];

        let fer = get_edge_fifo(&edgefifo, tri[0], tri[1], tri[2], edgefifooffset);

        if fer >= 0 && (fer >> 2) < 15 {
            // note: getEdgeFifo implicitly rotates triangles by matching a/b to existing edge
            let order = &ROTATIONS[(fer & 3) as usize..];

            let (a, b, c) = (tri[order[0]], tri[order[1]], tri[order[2]]);

            // encode edge index and vertex fifo index, next or free index
            let fe = fer >> 2;
            let fc = get_vertex_fifo(&vertexfifo, c, vertexfifooffset);

            let mut fec = if fc >= 1 && fc < fecmax {
                fc
            } else if c == next {
                next = next.wrapping_add(1);
                0
            } else {
                15
            };

            if fec == 15 && version >= 1 {
                // encode last-1 and last+1 to optimize strip-like sequences
                if c.wrapping_add(1) == last {
                    fec = 13;
                    last = c;
                }
                if c == last.wrapping_add(1) {
                    fec = 14;
                    last = c;
                }
            }

            buffer[code] = ((fe << 4) | fec) as u8;
            code += 1;

            // note that we need to update the last index since free indices are delta-encoded
            if fec == 15 {
                encode_index(buffer, &mut data, c, last);
                last = c;
            }

            // we only need to push third vertex since first two are likely already in the vertex fifo
            if fec == 0 || fec >= fecmax {
                push_vertex_fifo(&mut vertexfifo, c, &mut vertexfifooffset, true);
            }

            // we only need to push two new edges to edge fifo since the third one is already there
            push_edge_fifo(&mut edgefifo, c, b, &mut edgefifooffset);
            push_edge_fifo(&mut edgefifo, a, c, &mut edgefifooffset);
        } else {
            let rotation = rotate_triangle(tri[0], tri[1], tri[2], next);
            let order = &ROTATIONS[rotation..];

            let (a, b, c) = (tri[order[0]], tri[order[1]], tri[order[2]]);

            // if a/b/c are 0/1/2, we emit a reset code
            let mut reset = false;

            if a == 0 && b == 1 && c == 2 && next > 0 && version >= 1 {
                reset = true;
                next = 0;

                // reset vertex fifo to make sure we don't accidentally reference vertices from that in the future
                // this makes sure next continues to get incremented instead of being stuck
                vertexfifo = [!0u32; 16];
            }

            let fb = get_vertex_fifo(&vertexfifo, b, vertexfifooffset);
            let fc = get_vertex_fifo(&vertexfifo, c, vertexfifooffset);

            let mut take_next = |v: u32| {
                if v == next {
                    next = next.wrapping_add(1);
                    0
                } else {
                    15
                }
            };

            // after rotation, a is almost always equal to next, so we don't waste bits on FIFO encoding for a
            // note: decoder implicitly assumes that if feb=fec=0, then fea=0 (reset code); this is enforced by rotation
            let fea = take_next(a);
            let feb = if (0..14).contains(&fb) { fb + 1 } else { take_next(b) };
            let fec = if (0..14).contains(&fc) { fc + 1 } else { take_next(c) };

            // we encode feb & fec in 4 bits using a table if possible, and as a full byte otherwise
            let codeaux = ((feb << 4) | fec) as u8;
            let codeauxindex = get_code_aux_index(codeaux, codeaux_table);

            // <14 encodes an index into codeaux table, 14 encodes fea=0, 15 encodes fea=15
            if fea == 0 && (0..14).contains(&codeauxindex) && !reset {
                buffer[code] = ((15 << 4) | codeauxindex) as u8;
                code += 1;
            } else {
                buffer[code] = ((15 << 4) | 14 | fea) as u8;
                code += 1;
                buffer[data] = codeaux;
                data += 1;
            }

            // note that we need to update the last index since free indices are delta-encoded
            if fea == 15 {
                encode_index(buffer, &mut data, a, last);
                last = a;
            }

            if feb == 15 {
                encode_index(buffer, &mut data, b, last);
                last = b;
            }

            if fec == 15 {
                encode_index(buffer, &mut data, c, last);
                last = c;
            }

            // only push vertices that weren't already in fifo
            if fea == 0 || fea == 15 {
                push_vertex_fifo(&mut vertexfifo, a, &mut vertexfifooffset, true);
            }

            if feb == 0 || feb == 15 {
                push_vertex_fifo(&mut vertexfifo, b, &mut vertexfifooffset, true);
            }

            if fec == 0 || fec == 15 {
                push_vertex_fifo(&mut vertexfifo, c, &mut vertexfifooffset, true);
            }

            // all three edges aren't in the fifo; pushing all of them is important so that we can match them for later triangles
            push_edge_fifo(&mut edgefifo, b, a, &mut edgefifooffset);
            push_edge_fifo(&mut edgefifo, c, b, &mut edgefifooffset);
            push_edge_fifo(&mut edgefifo, a, c, &mut edgefifooffset);
        }
    }

    // make sure we have enough space to write codeaux table
    if data > data_safe_end {
        return Ok(0);
    }

    // add codeaux encoding table to the end of the stream; this is used for decoding codeaux *and* as padding
    // we need padding for decoding to be able to assume that each triangle is encoded as <= 16 bytes of extra data
    // this is enough space for aux byte + 5 bytes per varint index which is the absolute worst case for any input
    buffer[data..data + 16].copy_from_slice(codeaux_table);
    data += 16;

    Ok(data)
}

/// Worst case size of [`encode_index_buffer`] output.
pub fn encode_index_buffer_bound(index_count: usize, vertex_count: usize) -> Result<usize> {
    validate::triangles(index_count)?;

    // compute number of bits required for each index
    let mut vertex_bits = 1u32;

    while vertex_bits < 32 && vertex_count as u64 > 1u64 << vertex_bits {
        vertex_bits += 1;
    }

    // worst-case encoding is 2 header bytes + 3 varint-7 encoded index deltas
    let vertex_groups = (vertex_bits + 1).div_ceil(7);

    Ok(1 + (index_count / 3) * (2 + 3 * vertex_groups as usize) + 16)
}

/// Set index encoder format version (0 or 1, default 1).
pub fn encode_index_version(version: i32) -> Result<()> {
    ensure(
        (0..=DECODE_INDEX_VERSION).contains(&version),
        Error::InvalidArgument("unsupported index encoding version"),
    )?;

    ENCODE_INDEX_VERSION.store(version, Ordering::Relaxed);
    Ok(())
}

/// Decodes encoded index buffer/sequence version; `Err(Decode(-1))` if the buffer header is invalid.
/// Note that a non-negative result doesn't guarantee that decoding will succeed.
pub fn decode_index_version(buffer: &[u8], buffer_size: usize) -> Result<i32> {
    let buffer = buffer.get(..buffer_size).ok_or(Error::BufferTooSmall("buffer"))?;

    let header = *buffer.first().ok_or(Error::Decode(-1))?;

    if (header & 0xf0) != INDEX_HEADER && (header & 0xf0) != SEQUENCE_HEADER {
        return Err(Error::Decode(-1));
    }

    let version = i32::from(header & 0x0f);
    if version > DECODE_INDEX_VERSION {
        return Err(Error::Decode(-1));
    }

    Ok(version)
}

/// Decodes index data from an array of bytes generated by [`encode_index_buffer`].
/// Returns `Err(Decode(code))` with negative C status codes if decoding failed. `index_size` is `T::SIZE`.
/// The decoder is safe to use for untrusted input, but it may produce garbage data (e.g. out of range indices).
pub fn decode_index_buffer<T: IndexType>(destination: &mut [T], index_count: usize, buffer: &[u8], buffer_size: usize) -> Result<()> {
    validate::triangles(index_count)?;
    let buffer = buffer.get(..buffer_size).ok_or(Error::BufferTooSmall("buffer"))?;
    let destination = validate::output(destination, index_count)?;

    // the minimum valid encoding is header, 1 byte per triangle and a 16-byte codeaux table
    if buffer_size < 1 + index_count / 3 + 16 {
        return Err(Error::Decode(-2));
    }

    if (buffer[0] & 0xf0) != INDEX_HEADER {
        return Err(Error::Decode(-1));
    }

    let version = i32::from(buffer[0] & 0x0f);
    if version > DECODE_INDEX_VERSION {
        return Err(Error::Decode(-1));
    }

    let mut edgefifo: EdgeFifo = [[!0u32; 2]; 16];
    let mut vertexfifo: VertexFifo = [!0u32; 16];

    let mut edgefifooffset = 0usize;
    let mut vertexfifooffset = 0usize;

    let mut next = 0u32;
    let mut last = 0u32;

    let fecmax = if version >= 1 { 13 } else { 15 };

    // since we store 16-byte codeaux table at the end, triangle data has to begin before data_safe_end
    let code_end = 1 + index_count / 3;
    let mut data = code_end;

    // each triangle reads at most 16 bytes of data: 1b for codeaux and 5b for each free index
    let data_safe_end = buffer_size - 16;

    let codeaux_table = &buffer[data_safe_end..];

    let mut out = 0usize;

    for &codetri in &buffer[1..code_end] {
        if codetri < 0xf0 {
            let fe = usize::from(codetri >> 4);

            // fifo reads are wrapped around 16 entry buffer
            let [a, b] = edgefifo[edgefifooffset.wrapping_sub(1 + fe) & 15];
            let c;

            let fec = codetri & 15;

            // note: this is the most common path in the entire decoder
            // inside this if we try to stay branchless (by using cmov/etc.) since these aren't predictable
            if fec < fecmax {
                // fifo reads are wrapped around 16 entry buffer
                let cf = vertexfifo[vertexfifooffset.wrapping_sub(1 + usize::from(fec)) & 15];

                c = if fec == 0 { next } else { cf };

                let fec0 = fec == 0;
                next = next.wrapping_add(u32::from(fec0));

                // push vertex fifo must match the encoding step *exactly* otherwise the data will not be decoded correctly
                push_vertex_fifo(&mut vertexfifo, c, &mut vertexfifooffset, fec0);
            } else {
                // make sure we have enough data to read for a triangle; this check covers worst case advance
                if data > data_safe_end {
                    return Err(Error::Decode(-2));
                }

                // fec * 2 - 27 decodes 13, 14 into -1, 1
                // note that we need to update the last index since free indices are delta-encoded
                c = if fec != 15 {
                    last.wrapping_add((i32::from(fec) * 2 - 27) as u32)
                } else {
                    decode_index(buffer, &mut data, last)
                };
                last = c;

                // push vertex/edge fifo must match the encoding step *exactly* otherwise the data will not be decoded correctly
                push_vertex_fifo(&mut vertexfifo, c, &mut vertexfifooffset, true);
            }

            // push edge fifo must match the encoding step *exactly* otherwise the data will not be decoded correctly
            push_edge_fifo(&mut edgefifo, c, b, &mut edgefifooffset);
            push_edge_fifo(&mut edgefifo, a, c, &mut edgefifooffset);

            // output triangle
            write_triangle(destination, &mut out, a, b, c);
        } else if codetri < 0xfe {
            // fast path: read codeaux from the table
            let codeaux = codeaux_table[usize::from(codetri & 15)];

            // note: table can't contain feb/fec=15
            let feb = usize::from(codeaux >> 4);
            let fec = usize::from(codeaux & 15);

            // fifo reads are wrapped around 16 entry buffer
            // also note that we increment next for all three vertices before decoding indices - this matches encoder behavior
            let a = next;
            next = next.wrapping_add(1);

            let bf = vertexfifo[vertexfifooffset.wrapping_sub(feb) & 15];
            let b = if feb == 0 { next } else { bf };

            let feb0 = feb == 0;
            next = next.wrapping_add(u32::from(feb0));

            let cf = vertexfifo[vertexfifooffset.wrapping_sub(fec) & 15];
            let c = if fec == 0 { next } else { cf };

            let fec0 = fec == 0;
            next = next.wrapping_add(u32::from(fec0));

            // output triangle
            write_triangle(destination, &mut out, a, b, c);

            // push vertex/edge fifo must match the encoding step *exactly* otherwise the data will not be decoded correctly
            push_vertex_fifo(&mut vertexfifo, a, &mut vertexfifooffset, true);
            push_vertex_fifo(&mut vertexfifo, b, &mut vertexfifooffset, feb0);
            push_vertex_fifo(&mut vertexfifo, c, &mut vertexfifooffset, fec0);

            push_edge_fifo(&mut edgefifo, b, a, &mut edgefifooffset);
            push_edge_fifo(&mut edgefifo, c, b, &mut edgefifooffset);
            push_edge_fifo(&mut edgefifo, a, c, &mut edgefifooffset);
        } else {
            // make sure we have enough data to read for a triangle; this check covers worst case advance
            if data > data_safe_end {
                return Err(Error::Decode(-2));
            }

            // slow path: read a full byte for codeaux instead of using a table lookup
            let codeaux = buffer[data];
            data += 1;

            let fea = if codetri == 0xfe { 0 } else { 15 };
            let feb = usize::from(codeaux >> 4);
            let fec = usize::from(codeaux & 15);

            // reset: codeaux is 0 but encoded as not-a-table
            if codeaux == 0 {
                next = 0;
            }

            let mut take_next = || {
                let v = next;
                next = next.wrapping_add(1);
                v
            };

            // fifo reads are wrapped around 16 entry buffer
            // also note that we increment next for all three vertices before decoding indices - this matches encoder behavior
            let mut a = if fea == 0 { take_next() } else { 0 };
            let mut b = if feb == 0 {
                take_next()
            } else {
                vertexfifo[vertexfifooffset.wrapping_sub(feb) & 15]
            };
            let mut c = if fec == 0 {
                take_next()
            } else {
                vertexfifo[vertexfifooffset.wrapping_sub(fec) & 15]
            };

            // note that we need to update the last index since free indices are delta-encoded
            if fea == 15 {
                a = decode_index(buffer, &mut data, last);
                last = a;
            }

            if feb == 15 {
                b = decode_index(buffer, &mut data, last);
                last = b;
            }

            if fec == 15 {
                c = decode_index(buffer, &mut data, last);
                last = c;
            }

            // output triangle
            write_triangle(destination, &mut out, a, b, c);

            // push vertex/edge fifo must match the encoding step *exactly* otherwise the data will not be decoded correctly
            push_vertex_fifo(&mut vertexfifo, a, &mut vertexfifooffset, true);
            push_vertex_fifo(&mut vertexfifo, b, &mut vertexfifooffset, feb == 0 || feb == 15);
            push_vertex_fifo(&mut vertexfifo, c, &mut vertexfifooffset, fec == 0 || fec == 15);

            push_edge_fifo(&mut edgefifo, b, a, &mut edgefifooffset);
            push_edge_fifo(&mut edgefifo, c, b, &mut edgefifooffset);
            push_edge_fifo(&mut edgefifo, a, c, &mut edgefifooffset);
        }
    }

    // we should've read all data bytes and stopped at the boundary between data and codeaux table
    if data != data_safe_end {
        return Err(Error::Decode(-3));
    }

    Ok(())
}

/// Encodes index sequence into an array of bytes that is generally smaller and compresses better compared to original.
/// Input index sequence can represent arbitrary topology; for triangle lists [`encode_index_buffer`] is likely to be better.
/// Returns encoded data size on success, 0 on error; the only error condition is if `buffer` doesn't have enough space.
pub fn encode_index_sequence<T: IndexType>(buffer: &mut [u8], buffer_size: usize, indices: &[T], index_count: usize) -> Result<usize> {
    let buffer = buffer.get_mut(..buffer_size).ok_or(Error::BufferTooSmall("buffer"))?;
    let indices = indices.get(..index_count).ok_or(Error::BufferTooSmall("indices"))?;

    // the minimum valid encoding is header, 1 byte per index and a 4-byte tail
    if buffer_size < 1 + index_count + 4 {
        return Ok(0);
    }

    let version = ENCODE_INDEX_VERSION.load(Ordering::Relaxed);

    buffer[0] = SEQUENCE_HEADER | version as u8;

    let mut last = [0u32; 2];
    let mut current = 0usize;

    let mut data = 1usize;
    let data_safe_end = buffer_size - 4;

    for &index in indices {
        // make sure we have enough data to write
        // each index writes at most 5 bytes of data; there's a 4 byte tail after data_safe_end
        // after this we can be sure we can write without extra bounds checks
        if data >= data_safe_end {
            return Ok(0);
        }

        let index = index.to_u32();

        // this is a heuristic that switches between baselines when the delta grows too large
        // we want the encoded delta to fit into one byte (7 bits), but 2 bits are used for sign and baseline index
        // for now we immediately switch the baseline when delta grows too large - this can be adjusted arbitrarily
        let cd = index.wrapping_sub(last[current]) as i32;
        current ^= usize::from(cd.wrapping_abs() >= 30);

        // encode delta from the last index
        let d = index.wrapping_sub(last[current]);
        let v = (d << 1) ^ ((d as i32) >> 31) as u32;

        // note: low bit encodes the index of the last baseline which will be used for reconstruction
        encode_vbyte(buffer, &mut data, (v << 1) | current as u32);

        // update last for the next iteration that uses it
        last[current] = index;
    }

    // make sure we have enough space to write tail
    if data > data_safe_end {
        return Ok(0);
    }

    buffer[data..data + 4].fill(0);
    data += 4;

    Ok(data)
}

/// Worst case size of [`encode_index_sequence`] output.
pub fn encode_index_sequence_bound(index_count: usize, vertex_count: usize) -> usize {
    // compute number of bits required for each index
    let mut vertex_bits = 1u32;

    while vertex_bits < 32 && vertex_count as u64 > 1u64 << vertex_bits {
        vertex_bits += 1;
    }

    // worst-case encoding is 1 varint-7 encoded index delta for a K bit value and an extra bit
    let vertex_groups = (vertex_bits + 1 + 1).div_ceil(7);

    1 + index_count * vertex_groups as usize + 4
}

/// Decodes index data from an array of bytes generated by [`encode_index_sequence`].
/// Returns `Err(Decode(code))` with negative C status codes if decoding failed. `index_size` is `T::SIZE`.
pub fn decode_index_sequence<T: IndexType>(destination: &mut [T], index_count: usize, buffer: &[u8], buffer_size: usize) -> Result<()> {
    let buffer = buffer.get(..buffer_size).ok_or(Error::BufferTooSmall("buffer"))?;
    let destination = validate::output(destination, index_count)?;

    // the minimum valid encoding is header, 1 byte per index and a 4-byte tail
    if buffer_size < 1 + index_count + 4 {
        return Err(Error::Decode(-2));
    }

    if (buffer[0] & 0xf0) != SEQUENCE_HEADER {
        return Err(Error::Decode(-1));
    }

    let version = i32::from(buffer[0] & 0x0f);
    if version > DECODE_INDEX_VERSION {
        return Err(Error::Decode(-1));
    }

    let mut data = 1usize;
    let data_safe_end = buffer_size - 4;

    let mut last = [0u32; 2];

    for d in destination.iter_mut() {
        // make sure we have enough data to read
        // each index reads at most 5 bytes of data; there's a 4 byte tail after data_safe_end
        // after this we can be sure we can read without extra bounds checks
        if data >= data_safe_end {
            return Err(Error::Decode(-2));
        }

        let mut v = decode_vbyte(buffer, &mut data);

        // decode the index of the last baseline
        let current = (v & 1) as usize;
        v >>= 1;

        // reconstruct index as a delta
        let delta = (v >> 1) ^ (-((v & 1) as i32)) as u32;
        let index = last[current].wrapping_add(delta);

        // update last for the next iteration that uses it
        last[current] = index;

        *d = T::from_u32(index);
    }

    // we should've read all data bytes and stopped at the boundary between data and tail
    if data != data_safe_end {
        return Err(Error::Decode(-3));
    }

    Ok(())
}
