//! Vertex buffer codec; only the scalar code paths of the C++ implementation are ported (output is bit-identical).

use core::sync::atomic::{AtomicI32, Ordering};

use crate::error::{Error, Result, ensure};
use crate::validate;

const VERTEX_HEADER: u8 = 0xa0;

static ENCODE_VERTEX_VERSION: AtomicI32 = AtomicI32::new(1);
const DECODE_VERTEX_VERSION: i32 = 1;

const VERTEX_BLOCK_SIZE_BYTES: usize = 8192;
const VERTEX_BLOCK_MAX_SIZE: usize = 256;
const BYTE_GROUP_SIZE: usize = 16;
const BYTE_GROUP_DECODE_LIMIT: usize = 24;
const TAIL_MIN_SIZE_V0: usize = 32;
const TAIL_MIN_SIZE_V1: usize = 24;

const BITS_V0: [i32; 4] = [0, 2, 4, 8];
const BITS_V1: [i32; 5] = [0, 1, 2, 4, 8];

const ENCODE_DEFAULT_LEVEL: i32 = 2;

fn get_vertex_block_size(vertex_size: usize) -> usize {
    // make sure the entire block fits into the scratch buffer and is aligned to byte group size
    // note: the block size is implicitly part of the format, so we can't change it without breaking compatibility
    let result = (VERTEX_BLOCK_SIZE_BYTES / vertex_size) & !(BYTE_GROUP_SIZE - 1);

    result.min(VERTEX_BLOCK_MAX_SIZE)
}

fn zigzag(v: u32, bits: u32) -> u32 {
    (0u32.wrapping_sub(v >> (bits - 1)) ^ (v << 1)) & mask(bits)
}

fn unzigzag(v: u32, bits: u32) -> u32 {
    (0u32.wrapping_sub(v & 1) ^ (v >> 1)) & mask(bits)
}

fn mask(bits: u32) -> u32 {
    if bits == 32 { !0 } else { (1 << bits) - 1 }
}

fn read_le(data: &[u8], width: usize) -> u32 {
    data[..width].iter().rev().fold(0, |v, &b| (v << 8) | u32::from(b))
}

fn group(buffer: &[u8]) -> [u64; 2] {
    let mut lo = [0u8; 8];
    let mut hi = [0u8; 8];
    lo.copy_from_slice(&buffer[..8]);
    hi.copy_from_slice(&buffer[8..16]);
    [u64::from_le_bytes(lo), u64::from_le_bytes(hi)]
}

fn encode_bytes_group_zero(buffer: &[u8]) -> bool {
    let [v0, v1] = group(buffer);

    (v0 | v1) == 0
}

fn encode_bytes_group_measure(buffer: &[u8], bits: i32) -> usize {
    if bits == 0 {
        return if encode_bytes_group_zero(buffer) { 0 } else { usize::MAX };
    }

    if bits == 8 {
        return BYTE_GROUP_SIZE;
    }

    let [v0, v1] = group(buffer);

    let sent = ((1u64 << bits) - 1).wrapping_mul(0x0101010101010101);
    let mask = 0x8080808080808080u64;

    // each byte gets high bit 1 iff x >= sentinel; subtraction borrows high bit, so we need to reset high bit to 1 for all bytes that already had it as 1
    let m0 = (((v0 | mask).wrapping_sub(sent)) | v0) & mask;
    let m1 = (((v1 | mask).wrapping_sub(sent)) | v1) & mask;

    // adds all bytes together to get the number of bytes that were >= sentinel
    let rest = (((m0 >> 7) + (m1 >> 7)).wrapping_mul(0x0101010101010101) >> 56) as usize;

    BYTE_GROUP_SIZE / 8 * bits as usize + rest
}

fn reverse_bits_byte(byte: u8) -> u8 {
    ((((u64::from(byte)).wrapping_mul(0x80200802) & 0x0884422110).wrapping_mul(0x0101010101)) >> 32) as u8
}

fn encode_bytes_group(data: &mut [u8], mut pos: usize, buffer: &[u8], bits: i32) -> usize {
    if bits == 0 {
        return pos;
    }

    if bits == 8 {
        data[pos..pos + BYTE_GROUP_SIZE].copy_from_slice(&buffer[..BYTE_GROUP_SIZE]);
        return pos + BYTE_GROUP_SIZE;
    }

    let byte_size = 8usize >> (bits >> 1); // 8 / bits for bits 1/2/4

    // fixed portion: bits bits for each value
    // variable portion: full byte for each out-of-range value (using 1...1 as sentinel)
    let sentinel = ((1u32 << bits) - 1) as u8;

    for chunk in buffer[..BYTE_GROUP_SIZE].chunks_exact(byte_size) {
        let mut byte = 0u8;

        for &b in chunk {
            let enc = if b >= sentinel { sentinel } else { b };

            byte = byte.wrapping_shl(bits as u32);
            byte |= enc;
        }

        // encode 1-bit groups in reverse bit order
        // this makes them faster to decode alongside other groups
        if bits == 1 {
            byte = reverse_bits_byte(byte);
        }

        data[pos] = byte;
        pos += 1;
    }

    for &v in &buffer[..BYTE_GROUP_SIZE] {
        // append of out-of-range values
        if v >= sentinel {
            data[pos] = v;
            pos += 1;
        }
    }

    pos
}

/// Returns the new write position, or `None` if `data_end` is reached.
fn encode_bytes(data: &mut [u8], mut pos: usize, data_end: usize, buffer: &[u8], buffer_size: usize, bits: &[i32]) -> Option<usize> {
    let header = pos;

    // round number of groups to 4 to get number of header bytes
    let header_size = (buffer_size / BYTE_GROUP_SIZE).div_ceil(4);

    if data_end - pos < header_size {
        return None;
    }

    pos += header_size;

    data[header..header + header_size].fill(0);

    let mut last_bits = -1;

    for i in (0..buffer_size).step_by(BYTE_GROUP_SIZE) {
        if data_end - pos < BYTE_GROUP_DECODE_LIMIT {
            return None;
        }

        let group = &buffer[i..i + BYTE_GROUP_SIZE];

        let mut best_bitk = 3;
        let mut best_size = encode_bytes_group_measure(group, bits[best_bitk]);

        for bitk in 0..3 {
            let size = encode_bytes_group_measure(group, bits[bitk]);

            // favor consistent bit selection across groups, but never replace literals
            if size < best_size || (size == best_size && bits[bitk] == last_bits && bits[best_bitk] != 8) {
                best_bitk = bitk;
                best_size = size;
            }
        }

        let header_offset = i / BYTE_GROUP_SIZE;
        data[header + header_offset / 4] |= (best_bitk << ((header_offset % 4) * 2)) as u8;

        let best_bits = bits[best_bitk];
        pos = encode_bytes_group(data, pos, group, best_bits);

        last_bits = best_bits;
    }

    Some(pos)
}

/// Encodes deltas for 4 bytes of each vertex starting at byte `k` into `buffer[j * VERTEX_BLOCK_MAX_SIZE + i]`.
/// `width` is the delta element size (1, 2 or 4 bytes); 4-byte deltas use rotated xor.
fn encode_deltas4(
    buffer: &mut [u8],
    vertex_data: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &[u8],
    k: usize,
    width: usize,
    xor: bool,
    rot: u32,
) {
    let bits = (width * 8) as u32;

    for i in 0..vertex_count {
        let vertex = &vertex_data[i * vertex_size + k..];
        let last = if i == 0 {
            &last_vertex[k..]
        } else {
            &vertex_data[(i - 1) * vertex_size + k..]
        };

        for kk in (0..4).step_by(width) {
            let p = read_le(&last[kk..], width);
            let v = read_le(&vertex[kk..], width);

            let d = if xor {
                (v ^ p).rotate_left(rot)
            } else {
                zigzag(v.wrapping_sub(p) & mask(bits), bits)
            };

            for j in 0..width {
                buffer[(kk + j) * VERTEX_BLOCK_MAX_SIZE + i] = (d >> (j * 8)) as u8;
            }
        }
    }
}

fn encode_deltas(
    buffer: &mut [u8],
    vertex_data: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &[u8],
    k: usize,
    channel: i32,
) {
    // encode deltas for a single channel (4 bytes per vertex) into [buffer + 0..3 * kVertexBlockMaxSize]
    match channel & 3 {
        0 => encode_deltas4(buffer, vertex_data, vertex_count, vertex_size, last_vertex, k, 1, false, 0),
        1 => encode_deltas4(buffer, vertex_data, vertex_count, vertex_size, last_vertex, k, 2, false, 0),
        _ => encode_deltas4(
            buffer,
            vertex_data,
            vertex_count,
            vertex_size,
            last_vertex,
            k,
            4,
            true,
            (channel >> 4) as u32,
        ),
    }
}

fn estimate_bits(v: u8) -> usize {
    if v <= 15 {
        if v <= 3 { if v == 0 { 0 } else { 2 } } else { 4 }
    } else {
        8
    }
}

fn estimate_rotate(vertex_data: &[u8], vertex_count: usize, vertex_size: usize, k: usize, group_size: usize) -> i32 {
    let mut sizes = [0usize; 8];

    let mut offset = k;
    let mut last = read_le(&vertex_data[offset..], 4);

    for i in (0..vertex_count).step_by(group_size) {
        let mut bitg = 0u32;

        // calculate bit consistency mask for the group
        for _ in 0..group_size.min(vertex_count - i) {
            let v = read_le(&vertex_data[offset..], 4);
            let d = v ^ last;

            bitg |= d;
            last = v;
            offset += vertex_size;
        }

        for (j, size) in sizes.iter_mut().enumerate() {
            let bitr = bitg.rotate_left(j as u32);

            *size += estimate_bits(bitr as u8) + estimate_bits((bitr >> 8) as u8);
            *size += estimate_bits((bitr >> 16) as u8) + estimate_bits((bitr >> 24) as u8);
        }
    }

    let mut best_rot = 0;
    for rot in 1..8 {
        best_rot = if sizes[rot] < sizes[best_rot] { rot } else { best_rot };
    }

    best_rot as i32
}

fn estimate_channel(
    vertex_data: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    k: usize,
    vertex_block_size: usize,
    block_skip: usize,
    max_channel: i32,
    xor_rot: i32,
) -> i32 {
    let mut blocks = [0u8; VERTEX_BLOCK_MAX_SIZE * 4];

    let mut last_vertex = [0u8; 256];

    let mut sizes = [0usize; 3];

    for i in (0..vertex_count).step_by(vertex_block_size * block_skip) {
        let block_size = vertex_block_size.min(vertex_count - i);
        let block_size_aligned = block_size.next_multiple_of(BYTE_GROUP_SIZE);

        let src = (if i == 0 { 0 } else { i - 1 }) * vertex_size;
        last_vertex[..vertex_size].copy_from_slice(&vertex_data[src..src + vertex_size]);

        // we sometimes encode elements we didn't fill when rounding to kByteGroupSize
        if block_size < block_size_aligned {
            for j in 0..4 {
                blocks[j * VERTEX_BLOCK_MAX_SIZE + block_size..j * VERTEX_BLOCK_MAX_SIZE + block_size_aligned].fill(0);
            }
        }

        for (channel, size) in sizes.iter_mut().enumerate().take(max_channel as usize) {
            // fill blocks with deltas for 4 bytes per vertex (current channel) at once
            encode_deltas(
                &mut blocks,
                &vertex_data[i * vertex_size..],
                block_size,
                vertex_size,
                &last_vertex,
                k,
                channel as i32 | (xor_rot << 4),
            );

            for j in 0..4 {
                let block = &blocks[j * VERTEX_BLOCK_MAX_SIZE..];

                for ig in (0..block_size).step_by(BYTE_GROUP_SIZE) {
                    let group = &block[ig..ig + BYTE_GROUP_SIZE];

                    // to maximize encoding performance we only evaluate 1/2/4/8 bit groups
                    let size1 = encode_bytes_group_measure(group, 1);
                    let size2 = encode_bytes_group_measure(group, 2);
                    let size4 = encode_bytes_group_measure(group, 4);
                    let size8 = encode_bytes_group_measure(group, 8);

                    *size += size1.min(size2).min(size4).min(size8);
                }
            }
        }
    }

    let mut best_channel = 0;
    for channel in 1..max_channel as usize {
        best_channel = if sizes[channel] < sizes[best_channel] {
            channel
        } else {
            best_channel
        };
    }

    if best_channel == 2 {
        best_channel as i32 | (xor_rot << 4)
    } else {
        best_channel as i32
    }
}

fn estimate_control_zero(buffer: &[u8], vertex_count_aligned: usize) -> bool {
    buffer[..vertex_count_aligned]
        .as_chunks::<BYTE_GROUP_SIZE>()
        .0
        .iter()
        .all(|g| encode_bytes_group_zero(g))
}

fn estimate_control(buffer: &[u8], vertex_count: usize, vertex_count_aligned: usize, level: i32) -> usize {
    if estimate_control_zero(buffer, vertex_count_aligned) {
        return 2; // zero encoding
    }

    if level == 0 {
        return 1; // 1248 encoding in level 0 for encoding speed
    }

    // round number of groups to 4 to get number of header bytes
    let header_size = (vertex_count_aligned / BYTE_GROUP_SIZE).div_ceil(4);

    let mut est_bytes0 = header_size;
    let mut est_bytes1 = header_size;

    for group in buffer[..vertex_count_aligned].as_chunks::<BYTE_GROUP_SIZE>().0 {
        // assumes kBitsV1[] = {0, 1, 2, 4, 8} for performance
        let size0 = encode_bytes_group_measure(group, 0);
        let size1 = encode_bytes_group_measure(group, 1);
        let size2 = encode_bytes_group_measure(group, 2);
        let size4 = encode_bytes_group_measure(group, 4);
        let size8 = encode_bytes_group_measure(group, 8);

        // both control modes have access to 1/2/4 bit encoding
        let size124 = size1.min(size2).min(size4);

        // each control mode has access to 0/8 bit encoding respectively
        est_bytes0 += size124.min(size0);
        est_bytes1 += size124.min(size8);
    }

    // pick shortest control entry but prefer literal encoding
    if est_bytes0 < vertex_count || est_bytes1 < vertex_count {
        if est_bytes0 < est_bytes1 { 0 } else { 1 }
    } else {
        3 // literal encoding
    }
}

fn encode_vertex_block(
    data: &mut [u8],
    mut pos: usize,
    data_end: usize,
    vertex_data: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &mut [u8; 256],
    channels: &[u8; 64],
    version: i32,
    level: i32,
) -> Option<usize> {
    // we sometimes encode elements we didn't fill when rounding to kByteGroupSize
    let mut buffers = [0u8; VERTEX_BLOCK_MAX_SIZE * 4];

    let vertex_count_aligned = vertex_count.next_multiple_of(BYTE_GROUP_SIZE);

    let control_size = if version == 0 { 0 } else { vertex_size / 4 };
    if data_end - pos < control_size {
        return None;
    }

    let control = pos;
    pos += control_size;

    data[control..control + control_size].fill(0);

    for k in 0..vertex_size {
        // fill buffers with deltas for 4 bytes per vertex (current channel) at once
        if k % 4 == 0 {
            let channel = if version == 0 { 0 } else { i32::from(channels[k / 4]) };
            encode_deltas(&mut buffers, vertex_data, vertex_count, vertex_size, last_vertex, k, channel);
        }

        let buffer = &buffers[(k % 4) * VERTEX_BLOCK_MAX_SIZE..(k % 4 + 1) * VERTEX_BLOCK_MAX_SIZE];

        let mut ctrl = 0;

        if version != 0 {
            ctrl = estimate_control(buffer, vertex_count, vertex_count_aligned, level);

            data[control + k / 4] |= (ctrl << ((k % 4) * 2)) as u8;
        }

        if ctrl == 3 {
            // literal encoding
            if data_end - pos < vertex_count {
                return None;
            }

            data[pos..pos + vertex_count].copy_from_slice(&buffer[..vertex_count]);
            pos += vertex_count;
        } else if ctrl != 2 {
            // non-zero encoding
            let bits: &[i32] = if version == 0 { &BITS_V0 } else { &BITS_V1[ctrl..] };
            pos = encode_bytes(data, pos, data_end, buffer, vertex_count_aligned, bits)?;
        }
    }

    let last = vertex_size * (vertex_count - 1);
    last_vertex[..vertex_size].copy_from_slice(&vertex_data[last..last + vertex_size]);

    Some(pos)
}

/// Decodes one byte group; reads at most `BYTE_GROUP_DECODE_LIMIT` bytes from `data[pos..]`.
/// One byte group with `BITS` (1, 2 or 4) bits per value; values equal to the sentinel `(1 << BITS) - 1` are read from
/// the variable-length tail. `window` holds the group's [`BYTE_GROUP_DECODE_LIMIT`] bytes; returns bytes consumed.
#[inline(always)]
fn decode_bytes_group_bits<const BITS: u32>(window: &[u8; BYTE_GROUP_DECODE_LIMIT], buffer: &mut [u8; BYTE_GROUP_SIZE]) -> usize {
    let per_byte = (8 / BITS) as usize;
    let fixed = BYTE_GROUP_SIZE / per_byte;
    let sentinel = ((1u32 << BITS) - 1) as u8;

    let mut data_var = fixed;

    for (b, out) in buffer.as_chunks_mut::<1>().0.chunks_exact_mut(per_byte).enumerate() {
        let mut byte = window[b];

        // 1-bit groups are stored in reverse bit order
        if BITS == 1 {
            byte = reverse_bits_byte(byte);
        }

        for [o] in out.iter_mut() {
            let enc = byte >> (8 - BITS);
            byte = byte.wrapping_shl(BITS);

            // fixed + 16 sentinels never exceeds the window
            let encv = window[data_var % BYTE_GROUP_DECODE_LIMIT];
            *o = if enc == sentinel { encv } else { enc };
            data_var += usize::from(enc == sentinel);
        }
    }

    data_var
}

fn decode_bytes_group(data: &[u8], pos: usize, buffer: &mut [u8; BYTE_GROUP_SIZE], bits: i32) -> Option<usize> {
    let window = data.get(pos..)?.first_chunk::<BYTE_GROUP_DECODE_LIMIT>()?;

    let size = match bits {
        0 => {
            buffer.fill(0);
            0
        }
        1 => decode_bytes_group_bits::<1>(window, buffer),
        2 => decode_bytes_group_bits::<2>(window, buffer),
        4 => decode_bytes_group_bits::<4>(window, buffer),
        _ => {
            buffer.copy_from_slice(&window[..BYTE_GROUP_SIZE]);
            BYTE_GROUP_SIZE
        }
    };

    Some(pos + size)
}

fn decode_bytes(data: &[u8], mut pos: usize, data_end: usize, buffer: &mut [u8], buffer_size: usize, bits: &[i32]) -> Option<usize> {
    // round number of groups to 4 to get number of header bytes
    let header_size = (buffer_size / BYTE_GROUP_SIZE).div_ceil(4);
    if data_end - pos < header_size {
        return None;
    }

    let header = pos;
    pos += header_size;

    for (header_offset, group) in buffer[..buffer_size].as_chunks_mut::<BYTE_GROUP_SIZE>().0.iter_mut().enumerate() {
        if data_end - pos < BYTE_GROUP_DECODE_LIMIT {
            return None;
        }

        let bitsk = (data[header + header_offset / 4] >> ((header_offset % 4) * 2)) & 3;

        pos = decode_bytes_group(data, pos, group, bits[usize::from(bitsk)])?;
    }

    Some(pos)
}

/// `W`-byte deltas (1, 2 or 4) of one 4-byte channel; `XOR` selects rotated xor deltas over zigzag deltas.
fn decode_deltas4<const W: usize, const XOR: bool>(
    buffer: &[u8],
    transposed: &mut [u8],
    k: usize,
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &[u8],
    rot: u32,
) {
    let bits = (W * 8) as u32;

    for kk in (0..4).step_by(W) {
        let lanes = &buffer[vertex_count * kk..vertex_count * (kk + W)];
        let mut vertex_offset = k + kk;

        let mut p = read_le(&last_vertex[k + kk..], W);

        for i in 0..vertex_count {
            let mut v = 0u32;
            for j in 0..W {
                v |= u32::from(lanes[i + vertex_count * j]) << (8 * j);
            }

            v = if XOR {
                v.rotate_left(rot) ^ p
            } else {
                unzigzag(v, bits).wrapping_add(p) & mask(bits)
            };

            for j in 0..W {
                transposed[vertex_offset + j] = (v >> (j * 8)) as u8;
            }

            p = v;

            vertex_offset += vertex_size;
        }
    }
}

/// Per-call decode buffers, allocated once instead of zeroed for every block (C++: uninitialized stack arrays).
struct DecodeScratch {
    buffer: [u8; VERTEX_BLOCK_MAX_SIZE * 4],
    transposed: [u8; VERTEX_BLOCK_SIZE_BYTES],
}

fn decode_vertex_block(
    scratch: &mut DecodeScratch,
    data: &[u8],
    mut pos: usize,
    data_end: usize,
    vertex_data: &mut [u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &mut [u8; 256],
    channels: Option<&[u8]>,
    version: i32,
) -> Option<usize> {
    let DecodeScratch { buffer, transposed } = scratch;

    let vertex_count_aligned = vertex_count.next_multiple_of(BYTE_GROUP_SIZE);

    let control_size = if version == 0 { 0 } else { vertex_size / 4 };
    if data_end - pos < control_size {
        return None;
    }

    let control = pos;
    pos += control_size;

    for k in (0..vertex_size).step_by(4) {
        let ctrl_byte = if version == 0 { 0 } else { data[control + k / 4] };

        for j in 0..4 {
            let ctrl = usize::from((ctrl_byte >> (j * 2)) & 3);
            let target = &mut buffer[j * vertex_count..];

            if ctrl == 3 {
                // literal encoding
                if data_end - pos < vertex_count {
                    return None;
                }

                target[..vertex_count].copy_from_slice(&data[pos..pos + vertex_count]);
                pos += vertex_count;
            } else if ctrl == 2 {
                // zero encoding
                target[..vertex_count].fill(0);
            } else {
                let bits: &[i32] = if version == 0 { &BITS_V0 } else { &BITS_V1[ctrl..] };
                pos = decode_bytes(data, pos, data_end, target, vertex_count_aligned, bits)?;
            }
        }

        let channel = channels.map_or(0, |c| c[k / 4]);

        match channel & 3 {
            0 => decode_deltas4::<1, false>(buffer, transposed, k, vertex_count, vertex_size, last_vertex, 0),
            1 => decode_deltas4::<2, false>(buffer, transposed, k, vertex_count, vertex_size, last_vertex, 0),
            2 => {
                let rot = (32 - u32::from(channel >> 4)) & 31;
                decode_deltas4::<4, true>(buffer, transposed, k, vertex_count, vertex_size, last_vertex, rot)
            }
            _ => return None, // invalid channel type
        }
    }

    vertex_data[..vertex_count * vertex_size].copy_from_slice(&transposed[..vertex_count * vertex_size]);

    let last = vertex_size * (vertex_count - 1);
    last_vertex[..vertex_size].copy_from_slice(&transposed[last..last + vertex_size]);

    Some(pos)
}

fn validate_vertex_size(vertex_size: usize) -> Result<()> {
    ensure(
        vertex_size > 0 && vertex_size <= 256,
        Error::InvalidArgument("vertex_size must be in 1..=256"),
    )?;
    ensure(
        vertex_size.is_multiple_of(4),
        Error::InvalidArgument("vertex_size must be a multiple of 4"),
    )
}

/// Encodes vertex data into an array of bytes that is generally smaller and compresses better compared to original.
/// Returns encoded data size on success, 0 on error; the only error condition is if `buffer` doesn't have enough space.
/// `level` (0..=9) controls compression level; `version` is the format version (0/1) or -1 for the global default.
pub fn encode_vertex_buffer_level(
    buffer: &mut [u8],
    buffer_size: usize,
    vertices: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    level: i32,
    version: i32,
) -> Result<usize> {
    validate_vertex_size(vertex_size)?;
    ensure((0..=9).contains(&level), Error::InvalidArgument("level must be in 0..=9"))?;
    ensure(
        version <= DECODE_VERTEX_VERSION,
        Error::InvalidArgument("unsupported vertex encoding version"),
    )?;
    validate::bytes(vertices, vertex_count, vertex_size, vertex_size)?;
    let data = buffer.get_mut(..buffer_size).ok_or(Error::BufferTooSmall("buffer"))?;

    let version = if version < 0 {
        ENCODE_VERTEX_VERSION.load(Ordering::Relaxed)
    } else {
        version
    };

    let vertex_data = vertices;

    let data_end = buffer_size;
    let mut pos = 0usize;

    if data_end < 1 {
        return Ok(0);
    }

    data[pos] = VERTEX_HEADER | version as u8;
    pos += 1;

    let mut first_vertex = [0u8; 256];
    if vertex_count > 0 {
        first_vertex[..vertex_size].copy_from_slice(&vertex_data[..vertex_size]);
    }

    let mut last_vertex = first_vertex;

    let vertex_block_size = get_vertex_block_size(vertex_size);

    let mut channels = [0u8; 64];
    if version != 0 && level > 1 && vertex_count > 1 {
        for k in (0..vertex_size).step_by(4) {
            let rot = if level >= 3 {
                estimate_rotate(vertex_data, vertex_count, vertex_size, k, 16)
            } else {
                0
            };
            let channel = estimate_channel(
                vertex_data,
                vertex_count,
                vertex_size,
                k,
                vertex_block_size,
                3,
                if level >= 3 { 3 } else { 2 },
                rot,
            );

            channels[k / 4] = channel as u8;
        }
    }

    let mut vertex_offset = 0;

    while vertex_offset < vertex_count {
        let block_size = vertex_block_size.min(vertex_count - vertex_offset);

        let block = &vertex_data[vertex_offset * vertex_size..];
        match encode_vertex_block(
            data,
            pos,
            data_end,
            block,
            block_size,
            vertex_size,
            &mut last_vertex,
            &channels,
            version,
            level,
        ) {
            Some(p) => pos = p,
            None => return Ok(0),
        }

        vertex_offset += block_size;
    }

    let tail_size = vertex_size + if version == 0 { 0 } else { vertex_size / 4 };
    let tail_size_min = if version == 0 { TAIL_MIN_SIZE_V0 } else { TAIL_MIN_SIZE_V1 };
    let tail_size_pad = tail_size.max(tail_size_min);

    if data_end - pos < tail_size_pad {
        return Ok(0);
    }

    if tail_size < tail_size_pad {
        data[pos..pos + tail_size_pad - tail_size].fill(0);
        pos += tail_size_pad - tail_size;
    }

    data[pos..pos + vertex_size].copy_from_slice(&first_vertex[..vertex_size]);
    pos += vertex_size;

    if version != 0 {
        data[pos..pos + vertex_size / 4].copy_from_slice(&channels[..vertex_size / 4]);
        pos += vertex_size / 4;
    }

    Ok(pos)
}

/// Encodes vertex data with the default level (2) and the global format version; see [`encode_vertex_buffer_level`].
pub fn encode_vertex_buffer(
    buffer: &mut [u8],
    buffer_size: usize,
    vertices: &[u8],
    vertex_count: usize,
    vertex_size: usize,
) -> Result<usize> {
    encode_vertex_buffer_level(
        buffer,
        buffer_size,
        vertices,
        vertex_count,
        vertex_size,
        ENCODE_DEFAULT_LEVEL,
        ENCODE_VERTEX_VERSION.load(Ordering::Relaxed),
    )
}

/// Worst case size of [`encode_vertex_buffer`] output.
pub fn encode_vertex_buffer_bound(vertex_count: usize, vertex_size: usize) -> Result<usize> {
    validate_vertex_size(vertex_size)?;

    let vertex_block_size = get_vertex_block_size(vertex_size);
    let vertex_block_count = vertex_count.div_ceil(vertex_block_size);

    let vertex_block_control_size = vertex_size / 4;
    let vertex_byte_header_size = (vertex_block_size / BYTE_GROUP_SIZE).div_ceil(4);
    let vertex_byte_data_size = vertex_block_size;

    let tail_size = vertex_size + (vertex_size / 4);
    let tail_size_min = TAIL_MIN_SIZE_V0.max(TAIL_MIN_SIZE_V1);
    let tail_size_pad = tail_size.max(tail_size_min);

    Ok(
        1 + vertex_block_count * (vertex_block_control_size + vertex_size * (vertex_byte_header_size + vertex_byte_data_size))
            + tail_size_pad,
    )
}

/// Set vertex encoder format version (0 or 1, default 1).
pub fn encode_vertex_version(version: i32) -> Result<()> {
    ensure(
        (0..=DECODE_VERTEX_VERSION).contains(&version),
        Error::InvalidArgument("unsupported vertex encoding version"),
    )?;

    ENCODE_VERTEX_VERSION.store(version, Ordering::Relaxed);
    Ok(())
}

/// Decodes encoded vertex buffer version; `Err(Decode(-1))` if the buffer header is invalid.
pub fn decode_vertex_version(buffer: &[u8], buffer_size: usize) -> Result<i32> {
    let buffer = buffer.get(..buffer_size).ok_or(Error::BufferTooSmall("buffer"))?;

    let header = *buffer.first().ok_or(Error::Decode(-1))?;

    if (header & 0xf0) != VERTEX_HEADER {
        return Err(Error::Decode(-1));
    }

    let version = i32::from(header & 0x0f);
    if version > DECODE_VERTEX_VERSION {
        return Err(Error::Decode(-1));
    }

    Ok(version)
}

/// Decodes vertex data from an array of bytes generated by [`encode_vertex_buffer`].
/// Returns `Err(Decode(code))` with negative C status codes if decoding failed.
/// The decoder is safe to use for untrusted input, but it may produce garbage data.
pub fn decode_vertex_buffer(
    destination: &mut [u8],
    vertex_count: usize,
    vertex_size: usize,
    buffer: &[u8],
    buffer_size: usize,
) -> Result<()> {
    validate_vertex_size(vertex_size)?;
    let data = buffer.get(..buffer_size).ok_or(Error::BufferTooSmall("buffer"))?;
    let vertex_data = validate::output(destination, vertex_count * vertex_size)?;

    let data_end = buffer_size;
    let mut pos = 0usize;

    if data_end < 1 {
        return Err(Error::Decode(-2));
    }

    let data_header = data[pos];
    pos += 1;

    if (data_header & 0xf0) != VERTEX_HEADER {
        return Err(Error::Decode(-1));
    }

    let version = i32::from(data_header & 0x0f);
    if version > DECODE_VERTEX_VERSION {
        return Err(Error::Decode(-1));
    }

    let tail_size = vertex_size + if version == 0 { 0 } else { vertex_size / 4 };
    let tail_size_min = if version == 0 { TAIL_MIN_SIZE_V0 } else { TAIL_MIN_SIZE_V1 };
    let tail_size_pad = tail_size.max(tail_size_min);

    if data_end - pos < tail_size_pad {
        return Err(Error::Decode(-2));
    }

    let tail = data_end - tail_size;

    let mut last_vertex = [0u8; 256];
    last_vertex[..vertex_size].copy_from_slice(&data[tail..tail + vertex_size]);

    let channels = if version == 0 { None } else { Some(&data[tail + vertex_size..]) };

    let vertex_block_size = get_vertex_block_size(vertex_size);

    let mut vertex_offset = 0;
    let mut scratch = DecodeScratch {
        buffer: [0; VERTEX_BLOCK_MAX_SIZE * 4],
        transposed: [0; VERTEX_BLOCK_SIZE_BYTES],
    };

    while vertex_offset < vertex_count {
        let block_size = vertex_block_size.min(vertex_count - vertex_offset);

        let block = &mut vertex_data[vertex_offset * vertex_size..];
        pos = decode_vertex_block(
            &mut scratch,
            data,
            pos,
            data_end,
            block,
            block_size,
            vertex_size,
            &mut last_vertex,
            channels,
            version,
        )
        .ok_or(Error::Decode(-2))?;

        vertex_offset += block_size;
    }

    if data_end - pos != tail_size_pad {
        return Err(Error::Decode(-3));
    }

    Ok(())
}
