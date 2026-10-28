//! Wasm SIMD paths (`SIMD_WASM`, compile-time `simd128` like C++ `__wasm_simd128__`): vertex block decoding
//! (`decodeVertexBlockSimd`) and vertex filter decoding (`decodeFilter*Simd`). C++ has no Wasm meshlet decoder.

use std::arch::wasm32::*;

use super::common::{GROUP_COUNT, GROUP_SHUFFLE, blocks8, dispatch, window, window_mut};
use crate::index::IndexType;
use crate::meshletcodec::MeshletTriangleType;
use crate::vertexcodec::{BYTE_GROUP_DECODE_LIMIT, BYTE_GROUP_SIZE, DecodeScratch, VERTEX_BLOCK_MAX_SIZE};

/// 16 bytes into a register (`wasm_v128_load`).
#[inline]
#[target_feature(enable = "simd128")]
fn load16(b: &[u8; 16]) -> v128 {
    let (c, _) = b.as_chunks::<8>();
    u64x2(u64::from_le_bytes(c[0]), u64::from_le_bytes(c[1]))
}

/// Register into 16 bytes (`wasm_v128_store`).
#[inline]
#[target_feature(enable = "simd128")]
fn store16(o: &mut [u8; 16], v: v128) {
    let (c, _) = o.as_chunks_mut::<8>();
    c[0] = u64x2_extract_lane::<0>(v).to_le_bytes();
    c[1] = u64x2_extract_lane::<1>(v).to_le_bytes();
}

/// `wasmx_unpacklo_v8x16`
#[inline]
#[target_feature(enable = "simd128")]
fn unpacklo8(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a, b)
}

/// `wasmx_unpackhi_v8x16`
#[inline]
#[target_feature(enable = "simd128")]
fn unpackhi8(a: v128, b: v128) -> v128 {
    i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(a, b)
}

/// `wasmx_unpacklo_v16x8`
#[inline]
#[target_feature(enable = "simd128")]
fn unpacklo16(a: v128, b: v128) -> v128 {
    i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(a, b)
}

/// `wasmx_unpackhi_v16x8`
#[inline]
#[target_feature(enable = "simd128")]
fn unpackhi16(a: v128, b: v128) -> v128 {
    i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(a, b)
}

/// `wasmx_unziplo_v32x4`
#[inline]
#[target_feature(enable = "simd128")]
fn unziplo32(a: v128, b: v128) -> v128 {
    i32x4_shuffle::<0, 2, 4, 6>(a, b)
}

/// `wasmx_unziphi_v32x4`
#[inline]
#[target_feature(enable = "simd128")]
fn unziphi32(a: v128, b: v128) -> v128 {
    i32x4_shuffle::<1, 3, 5, 7>(a, b)
}

/// `decodeShuffleMask`: both 8-byte halves of the expansion shuffle; the second half skips the bytes of the first.
#[inline]
#[target_feature(enable = "simd128")]
fn decode_shuffle_mask(mask0: u8, mask1: u8) -> v128 {
    let sm0 = u64::from_le_bytes(GROUP_SHUFFLE[usize::from(mask0)]);
    let sm1 = u64::from_le_bytes(GROUP_SHUFFLE[usize::from(mask1)]);

    let sm1off = u8x16_splat(GROUP_COUNT[usize::from(mask0)]);
    let sm1r = i8x16_add(u64x2(sm1, 0), sm1off);

    i64x2_shuffle::<0, 2>(u64x2(sm0, 0), sm1r)
}

/// `wasmMoveMask`
#[inline]
#[target_feature(enable = "simd128")]
fn move_mask(mask: v128) -> (u8, u8) {
    // magic constant found using z3 SMT assuming mask has 8 groups of 0xff or 0x00
    const MAGIC: u64 = 0x000103070f1f3f80;

    let mask0 = (u64x2_extract_lane::<0>(mask).wrapping_mul(MAGIC) >> 56) as u8;
    let mask1 = (u64x2_extract_lane::<1>(mask).wrapping_mul(MAGIC) >> 56) as u8;
    (mask0, mask1)
}

/// `decodeBytesGroupSimd`: decodes one 16-byte group from `w`, returns the number of bytes consumed.
#[target_feature(enable = "simd128")]
fn decode_bytes_group(w: &[u8; BYTE_GROUP_DECODE_LIMIT], buffer: &mut [u8; BYTE_GROUP_SIZE], hbits: usize) -> Option<usize> {
    let at16 = |at: usize| window::<16>(w, at);
    let counts = |mask0: u8, mask1: u8| usize::from(GROUP_COUNT[usize::from(mask0)]) + usize::from(GROUP_COUNT[usize::from(mask1)]);

    match hbits {
        0 | 4 => {
            store16(buffer, i8x16_splat(0));
            Some(0)
        }
        1 | 6 => {
            let sel2 = load16(at16(0)?);
            let rest = load16(at16(4)?);

            let sel22 = unpacklo8(i16x8_shr(sel2, 4), sel2);
            let sel2222 = unpacklo8(i16x8_shr(sel22, 2), sel22);
            let sel = v128_and(sel2222, i8x16_splat(3));

            let mask = i8x16_eq(sel, i8x16_splat(3));
            let (mask0, mask1) = move_mask(mask);

            let shuf = decode_shuffle_mask(mask0, mask1);
            let result = v128_bitselect(i8x16_swizzle(rest, shuf), sel, mask);

            store16(buffer, result);

            Some(4 + counts(mask0, mask1))
        }
        2 | 7 => {
            let sel4 = load16(at16(0)?);
            let rest = load16(at16(8)?);

            let sel44 = unpacklo8(i16x8_shr(sel4, 4), sel4);
            let sel = v128_and(sel44, i8x16_splat(15));

            let mask = i8x16_eq(sel, i8x16_splat(15));
            let (mask0, mask1) = move_mask(mask);

            let shuf = decode_shuffle_mask(mask0, mask1);
            let result = v128_bitselect(i8x16_swizzle(rest, shuf), sel, mask);

            store16(buffer, result);

            Some(8 + counts(mask0, mask1))
        }
        3 | 8 => {
            store16(buffer, load16(at16(0)?));
            Some(16)
        }
        _ => {
            // 1-bit groups (v1 only): the two header bytes are the sentinel masks themselves
            let rest = load16(at16(2)?);

            let (mask0, mask1) = (w[0], w[1]);

            let shuf = decode_shuffle_mask(mask0, mask1);
            store16(buffer, i8x16_swizzle(rest, shuf));

            Some(2 + counts(mask0, mask1))
        }
    }
}

/// `decodeBytesSimd`: `buffer_size` (a multiple of 16) bytes into `buffer`; returns the new data position.
#[target_feature(enable = "simd128")]
fn decode_bytes(data: &[u8], mut pos: usize, buffer: &mut [u8], buffer_size: usize, hshift: usize) -> Option<usize> {
    let data_end = data.len();

    // round number of groups to 4 to get number of header bytes
    let header_size = (buffer_size / BYTE_GROUP_SIZE).div_ceil(4);
    if data_end - pos < header_size {
        return None;
    }

    let header = pos;
    pos += header_size;

    let groups = buffer.get_mut(..buffer_size)?.as_chunks_mut::<BYTE_GROUP_SIZE>().0;
    let mut i = 0;

    // fast-path: process 4 groups at a time, do a shared bounds check
    while i + 4 <= groups.len() && data_end - pos >= BYTE_GROUP_DECODE_LIMIT * 4 {
        let header_byte = data[header + i / 4];

        for (g, group) in groups[i..i + 4].iter_mut().enumerate() {
            let w = window::<BYTE_GROUP_DECODE_LIMIT>(data, pos)?;
            pos += decode_bytes_group(w, group, hshift + usize::from((header_byte >> (g * 2)) & 3))?;
        }

        i += 4;
    }

    // slow-path: process remaining groups
    while i < groups.len() {
        if data_end - pos < BYTE_GROUP_DECODE_LIMIT {
            return None;
        }

        let header_byte = data[header + i / 4];
        let w = window::<BYTE_GROUP_DECODE_LIMIT>(data, pos)?;
        pos += decode_bytes_group(w, &mut groups[i], hshift + usize::from((header_byte >> ((i % 4) * 2)) & 3))?;

        i += 1;
    }

    Some(pos)
}

#[inline]
#[target_feature(enable = "simd128")]
fn transpose8(x: &mut [v128; 4]) {
    let t0 = unpacklo8(x[0], x[1]);
    let t1 = unpackhi8(x[0], x[1]);
    let t2 = unpacklo8(x[2], x[3]);
    let t3 = unpackhi8(x[2], x[3]);

    x[0] = unpacklo16(t0, t2);
    x[1] = unpackhi16(t0, t2);
    x[2] = unpacklo16(t1, t3);
    x[3] = unpackhi16(t1, t3);
}

#[inline]
#[target_feature(enable = "simd128")]
fn unzigzag8(v: v128) -> v128 {
    let xl = i8x16_neg(v128_and(v, i8x16_splat(1)));
    let xr = u8x16_shr(v, 1);

    v128_xor(xl, xr)
}

#[inline]
#[target_feature(enable = "simd128")]
fn unzigzag16(v: v128) -> v128 {
    let xl = i16x8_neg(v128_and(v, i16x8_splat(1)));
    let xr = u16x8_shr(v, 1);

    v128_xor(xl, xr)
}

#[inline]
#[target_feature(enable = "simd128")]
fn rotate32(v: v128, r: u32) -> v128 {
    // wasm shift counts are modulo 32, so r = 0 gives v | v
    v128_or(i32x4_shl(v, r), u32x4_shr(v, 32 - r))
}

/// `decodeDeltas4Simd<Channel>`: 16 vertices at a time; 4 transposed byte lanes, then a running delta per vertex.
#[target_feature(enable = "simd128")]
fn decode_deltas4<const CHANNEL: u8>(
    buffer: &[u8],
    transposed: &mut [u8],
    k: usize,
    vertex_count_aligned: usize,
    vertex_size: usize,
    last_vertex: &[u8; 4],
    rot: u32,
) -> Option<()> {
    // C++ loads 16 bytes of last_vertex; only lane 0 reaches the output, and lane arithmetic never crosses lanes
    let mut pi = u32x4(u32::from_le_bytes(*last_vertex), 0, 0, 0);

    let mut savep = k;

    for j in (0..vertex_count_aligned).step_by(16) {
        let mut r = [i8x16_splat(0); 4];
        for (i, r) in r.iter_mut().enumerate() {
            *r = load16(window::<16>(buffer, j + i * vertex_count_aligned)?);
        }

        transpose8(&mut r);

        for r in r {
            let r = match CHANNEL {
                0 => unzigzag8(r),
                1 => unzigzag16(r),
                _ => rotate32(r, rot),
            };

            let t = [
                r,
                i32x4_shuffle::<1, 1, 1, 1>(r, r),
                i32x4_shuffle::<2, 2, 2, 2>(r, r),
                i32x4_shuffle::<3, 3, 3, 3>(r, r),
            ];

            for t in t {
                pi = match CHANNEL {
                    0 => i8x16_add(pi, t),
                    1 => i16x8_add(pi, t),
                    _ => v128_xor(pi, t),
                };

                *window_mut::<4>(transposed, savep)? = u32x4_extract_lane::<0>(pi).to_le_bytes();
                savep += vertex_size;
            }
        }
    }

    Some(())
}

/// `decodeVertexBlockSimd`
#[target_feature(enable = "simd128")]
fn decode_vertex_block(
    scratch: &mut DecodeScratch,
    data: &[u8],
    mut pos: usize,
    vertex_data: &mut [u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &mut [u8; 256],
    channels: Option<&[u8]>,
    version: i32,
) -> Option<usize> {
    debug_assert!(vertex_count > 0 && vertex_count <= VERTEX_BLOCK_MAX_SIZE);

    let DecodeScratch { buffer, transposed } = scratch;
    let data_end = data.len();

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
            let target = buffer.get_mut(j * vertex_count_aligned..(j + 1) * vertex_count_aligned)?;

            if ctrl == 3 {
                // literal encoding; safe to over-copy due to tail
                if data_end - pos < vertex_count_aligned {
                    return None;
                }

                target.copy_from_slice(&data[pos..pos + vertex_count_aligned]);
                pos += vertex_count;
            } else if ctrl == 2 {
                // zero encoding
                target.fill(0);
            } else {
                // for v0, headers are mapped to 0..3; for v1, headers are mapped to 4..8
                let hshift = if version == 0 { 0 } else { 4 + ctrl };

                pos = decode_bytes(data, pos, target, vertex_count_aligned, hshift)?;
            }
        }

        let channel = channels.map_or(0, |c| c[k / 4]);
        let last = window::<4>(last_vertex, k)?;

        match channel & 3 {
            0 => decode_deltas4::<0>(buffer, transposed, k, vertex_count_aligned, vertex_size, last, 0)?,
            1 => decode_deltas4::<1>(buffer, transposed, k, vertex_count_aligned, vertex_size, last, 0)?,
            2 => decode_deltas4::<2>(
                buffer,
                transposed,
                k,
                vertex_count_aligned,
                vertex_size,
                last,
                u32::from((32 - (channel >> 4)) & 31),
            )?,
            _ => return None, // invalid channel type
        }
    }

    vertex_data[..vertex_count * vertex_size].copy_from_slice(&transposed[..vertex_count * vertex_size]);

    let last = vertex_size * (vertex_count - 1);
    last_vertex[..vertex_size].copy_from_slice(&transposed[last..last + vertex_size]);

    Some(pos)
}

/// Wasm SIMD is selected at compile time (`target_feature = "simd128"`); there is no runtime detection, and calling
/// `#[target_feature]` functions is safe on Wasm (an engine without SIMD rejects the module), so no `unsafe` is needed.
#[derive(Clone, Copy)]
pub(crate) struct VertexDecoder(());

impl VertexDecoder {
    pub(crate) fn detect() -> Option<Self> {
        Some(VertexDecoder(()))
    }

    pub(crate) fn decode_vertex_block(
        self,
        scratch: &mut DecodeScratch,
        data: &[u8],
        pos: usize,
        vertex_data: &mut [u8],
        vertex_count: usize,
        vertex_size: usize,
        last_vertex: &mut [u8; 256],
        channels: Option<&[u8]>,
        version: i32,
    ) -> Option<usize> {
        decode_vertex_block(
            scratch,
            data,
            pos,
            vertex_data,
            vertex_count,
            vertex_size,
            last_vertex,
            channels,
            version,
        )
    }
}

/// fast rounded signed float->int: addition triggers renormalization after which mantissa stores the integer value
/// note: the result is offset by 0x4B40_0000, but only the low 8/16 bits are used so the subtraction is omitted
const FSNAP: f32 = (3 << 22) as f32;

/// `decodeFilterOctSimd8`
#[target_feature(enable = "simd128")]
fn decode_oct8(data: &mut [u8], count: usize) {
    let sign = f32x4_splat(-0.0);

    for block in data[..count * 4].as_chunks_mut::<16>().0 {
        let n4 = load16(block);

        // sign-extends each of x,y in [x y ? ?] with arithmetic shifts
        let xf = i32x4_shr(i32x4_shl(n4, 24), 24);
        let yf = i32x4_shr(i32x4_shl(n4, 16), 24);

        // unpack z; note that z is unsigned so we technically don't need to sign extend it
        let zf = i32x4_shr(i32x4_shl(n4, 8), 24);

        // convert x and y to floats and reconstruct z; this assumes zf encodes 1.f at the same bit count
        let mut x = f32x4_convert_i32x4(xf);
        let mut y = f32x4_convert_i32x4(yf);
        let z = f32x4_sub(f32x4_convert_i32x4(zf), f32x4_add(f32x4_abs(x), f32x4_abs(y)));

        // fixup octahedral coordinates for z<0
        // note: i32x4_min with 0 is equivalent to f32x4_min
        let t = i32x4_min(z, i32x4_splat(0));

        x = f32x4_add(x, v128_xor(t, v128_and(x, sign)));
        y = f32x4_add(y, v128_xor(t, v128_and(y, sign)));

        // compute normal length & scale
        let ll = f32x4_add(f32x4_mul(x, x), f32x4_add(f32x4_mul(y, y), f32x4_mul(z, z)));
        let s = f32x4_div(f32x4_splat(127.0), f32x4_sqrt(ll));

        let fsnap = f32x4_splat(FSNAP);

        let xr = f32x4_add(f32x4_mul(x, s), fsnap);
        let yr = f32x4_add(f32x4_mul(y, s), fsnap);
        let zr = f32x4_add(f32x4_mul(z, s), fsnap);

        // combine xr/yr/zr into final value
        let mut res = v128_and(n4, u32x4_splat(0xff000000));
        res = v128_or(res, v128_and(xr, i32x4_splat(0xff)));
        res = v128_or(res, i32x4_shl(v128_and(yr, i32x4_splat(0xff)), 8));
        res = v128_or(res, i32x4_shl(v128_and(zr, i32x4_splat(0xff)), 16));

        store16(block, res);
    }
}

/// `decodeFilterOctSimd16`
#[target_feature(enable = "simd128")]
fn decode_oct16(data: &mut [u8], count: usize) {
    let sign = f32x4_splat(-0.0);
    let zmask = i32x4_splat(0x7fff);

    for [b0, b1] in blocks8(data, count) {
        let n4_0 = load16(b0);
        let n4_1 = load16(b1);

        // gather both x/y 16-bit pairs in each 32-bit lane
        let n4 = unziplo32(n4_0, n4_1);

        // sign-extends each of x,y in [x y] with arithmetic shifts
        let xf = i32x4_shr(i32x4_shl(n4, 16), 16);
        let yf = i32x4_shr(n4, 16);

        // unpack z; note that z is unsigned so we don't need to sign extend it
        let z4 = unziphi32(n4_0, n4_1);
        let zf = v128_and(z4, zmask);

        // convert x and y to floats and reconstruct z; this assumes zf encodes 1.f at the same bit count
        let mut x = f32x4_convert_i32x4(xf);
        let mut y = f32x4_convert_i32x4(yf);
        let z = f32x4_sub(f32x4_convert_i32x4(zf), f32x4_add(f32x4_abs(x), f32x4_abs(y)));

        // fixup octahedral coordinates for z<0
        // note: i32x4_min with 0 is equivalent to f32x4_min
        let t = i32x4_min(z, i32x4_splat(0));

        x = f32x4_add(x, v128_xor(t, v128_and(x, sign)));
        y = f32x4_add(y, v128_xor(t, v128_and(y, sign)));

        // compute normal length & scale
        let ll = f32x4_add(f32x4_mul(x, x), f32x4_add(f32x4_mul(y, y), f32x4_mul(z, z)));
        let s = f32x4_div(f32x4_splat(32767.0), f32x4_sqrt(ll));

        let fsnap = f32x4_splat(FSNAP);

        let xr = f32x4_add(f32x4_mul(x, s), fsnap);
        let yr = f32x4_add(f32x4_mul(y, s), fsnap);
        let zr = f32x4_add(f32x4_mul(z, s), fsnap);

        // mix x/z and y/0 to make 16-bit unpack easier
        let xzr = v128_or(v128_and(xr, i32x4_splat(0xffff)), i32x4_shl(zr, 16));
        let y0r = v128_and(yr, i32x4_splat(0xffff));

        // pack x/y/z using 16-bit unpacks; note that this has 0 where we should have .w
        let mut res_0 = unpacklo16(xzr, y0r);
        let mut res_1 = unpackhi16(xzr, y0r);

        // patch in .w
        res_0 = v128_or(res_0, v128_and(n4_0, u64x2_splat(0xffff000000000000)));
        res_1 = v128_or(res_1, v128_and(n4_1, u64x2_splat(0xffff000000000000)));

        store16(b0, res_0);
        store16(b1, res_1);
    }
}

/// `decodeFilterQuatSimd`
#[target_feature(enable = "simd128")]
fn decode_quat(data: &mut [u8], count: usize) {
    let scale = 32767.0 / 2.0f32.sqrt();

    for [b0, b1] in blocks8(data, count) {
        let q4_0 = load16(b0);
        let q4_1 = load16(b1);

        // gather both x/y 16-bit pairs in each 32-bit lane
        let q4_xy = unziplo32(q4_0, q4_1);
        let q4_zc = unziphi32(q4_0, q4_1);

        // sign-extends each of x,y in [x y] with arithmetic shifts
        let xf = i32x4_shr(i32x4_shl(q4_xy, 16), 16);
        let yf = i32x4_shr(q4_xy, 16);
        let zf = i32x4_shr(i32x4_shl(q4_zc, 16), 16);
        let cf = i32x4_shr(q4_zc, 16);

        // get a floating-point scaler using zc with bottom 2 bits set to 1 (which represents 1.f)
        let sf = v128_or(cf, i32x4_splat(3));
        let s = f32x4_convert_i32x4(sf);

        // convert x/y/z to floating point (unscaled! implied scale of 1/sqrt(2.f) * 1/sf)
        let x = f32x4_convert_i32x4(xf);
        let y = f32x4_convert_i32x4(yf);
        let z = f32x4_convert_i32x4(zf);

        // reconstruct w as a square root (unscaled); we clamp to 0.f to avoid NaN due to precision errors
        // note: i32x4_max with 0 is equivalent to f32x4_max
        let ws = f32x4_mul(s, s);
        let ww = f32x4_sub(
            f32x4_add(ws, ws),
            f32x4_add(f32x4_mul(x, x), f32x4_add(f32x4_mul(y, y), f32x4_mul(z, z))),
        );
        let w = f32x4_sqrt(i32x4_max(ww, i32x4_splat(0)));

        // compute final scale; note that all computations above are unscaled
        // we need to divide by sf to get out of fixed point, divide by sqrt(2) to renormalize and multiply by 32767 to get to int16 range
        let ss = f32x4_div(f32x4_splat(scale), s);

        let fsnap = f32x4_splat(FSNAP);

        let xr = f32x4_add(f32x4_mul(x, ss), fsnap);
        let yr = f32x4_add(f32x4_mul(y, ss), fsnap);
        let zr = f32x4_add(f32x4_mul(z, ss), fsnap);
        let wr = f32x4_add(f32x4_mul(w, ss), fsnap);

        // mix x/z and w/y to make 16-bit unpack easier
        let xzr = v128_or(v128_and(xr, i32x4_splat(0xffff)), i32x4_shl(zr, 16));
        let wyr = v128_or(v128_and(wr, i32x4_splat(0xffff)), i32x4_shl(yr, 16));

        // pack x/y/z/w using 16-bit unpacks; we pack wxyz by default (for qc=0)
        let res_0 = unpacklo16(wyr, xzr);
        let res_1 = unpackhi16(wyr, xzr);

        // compute component index shifted left by 4 (and moved into i32x4 slot)
        let cm = i32x4_shl(cf, 4);

        // rotate and store
        let rotate = |v: u64, r: i32| v.rotate_left(r as u32).to_le_bytes();
        let (o0, _) = b0.as_chunks_mut::<8>();
        o0[0] = rotate(u64x2_extract_lane::<0>(res_0), i32x4_extract_lane::<0>(cm));
        o0[1] = rotate(u64x2_extract_lane::<1>(res_0), i32x4_extract_lane::<1>(cm));
        let (o1, _) = b1.as_chunks_mut::<8>();
        o1[0] = rotate(u64x2_extract_lane::<0>(res_1), i32x4_extract_lane::<2>(cm));
        o1[1] = rotate(u64x2_extract_lane::<1>(res_1), i32x4_extract_lane::<3>(cm));
    }
}

/// `decodeFilterExpSimd`
#[target_feature(enable = "simd128")]
fn decode_exp(data: &mut [u8], count: usize) {
    for block in data[..count * 4].as_chunks_mut::<16>().0 {
        let v = load16(block);

        // decode exponent into 2^x directly
        let ef = i32x4_shr(v, 24);
        let es = i32x4_shl(i32x4_add(ef, i32x4_splat(127)), 23);

        // decode 24-bit mantissa into floating-point value
        let mf = i32x4_shr(i32x4_shl(v, 8), 8);
        let m = f32x4_convert_i32x4(mf);

        let r = f32x4_mul(es, m);

        store16(block, r);
    }
}

/// `decodeFilterColorSimd8`
#[target_feature(enable = "simd128")]
fn decode_color8(data: &mut [u8], count: usize) {
    for block in data[..count * 4].as_chunks_mut::<16>().0 {
        let c4 = load16(block);

        // unpack y/co/cg/a (co/cg are sign extended with arithmetic shifts)
        let yf = v128_and(c4, i32x4_splat(0xff));
        let cof = i32x4_shr(i32x4_shl(c4, 16), 24);
        let cgf = i32x4_shr(i32x4_shl(c4, 8), 24);
        let mut af = u32x4_shr(c4, 24);

        // recover scale from alpha high bit
        let mut as_ = af;
        as_ = v128_or(as_, i32x4_shr(as_, 1));
        as_ = v128_or(as_, i32x4_shr(as_, 2));
        as_ = v128_or(as_, i32x4_shr(as_, 4));

        // expand alpha by one bit to match other components
        af = v128_or(v128_and(i32x4_shl(af, 1), as_), v128_and(af, i32x4_splat(1)));

        // compute scaling factor
        let ss = f32x4_div(f32x4_splat(255.0), f32x4_convert_i32x4(as_));

        // convert to RGB in fixed point
        let rf = i32x4_add(yf, i32x4_sub(cof, cgf));
        let gf = i32x4_add(yf, cgf);
        let bf = i32x4_sub(yf, i32x4_add(cof, cgf));

        let fsnap = f32x4_splat(FSNAP);

        let rr = f32x4_add(f32x4_mul(f32x4_convert_i32x4(rf), ss), fsnap);
        let gr = f32x4_add(f32x4_mul(f32x4_convert_i32x4(gf), ss), fsnap);
        let br = f32x4_add(f32x4_mul(f32x4_convert_i32x4(bf), ss), fsnap);
        let ar = f32x4_add(f32x4_mul(f32x4_convert_i32x4(af), ss), fsnap);

        // repack rgba into final value
        let mut res = v128_and(rr, i32x4_splat(0xff));
        res = v128_or(res, i32x4_shl(v128_and(gr, i32x4_splat(0xff)), 8));
        res = v128_or(res, i32x4_shl(v128_and(br, i32x4_splat(0xff)), 16));
        res = v128_or(res, i32x4_shl(ar, 24));

        store16(block, res);
    }
}

/// `decodeFilterColorSimd16`
#[target_feature(enable = "simd128")]
fn decode_color16(data: &mut [u8], count: usize) {
    for [b0, b1] in blocks8(data, count) {
        let c4_0 = load16(b0);
        let c4_1 = load16(b1);

        // gather both y/co 16-bit pairs in each 32-bit lane
        let c4_yco = unziplo32(c4_0, c4_1);
        let c4_cga = unziphi32(c4_0, c4_1);

        // unpack y/co/cg/a components (co/cg are sign extended with arithmetic shifts)
        let yf = v128_and(c4_yco, i32x4_splat(0xffff));
        let cof = i32x4_shr(c4_yco, 16);
        let cgf = i32x4_shr(i32x4_shl(c4_cga, 16), 16);
        let mut af = u32x4_shr(c4_cga, 16);

        // recover scale from alpha high bit
        let mut as_ = af;
        as_ = v128_or(as_, i32x4_shr(as_, 1));
        as_ = v128_or(as_, i32x4_shr(as_, 2));
        as_ = v128_or(as_, i32x4_shr(as_, 4));
        as_ = v128_or(as_, i32x4_shr(as_, 8));

        // expand alpha by one bit to match other components
        af = v128_or(v128_and(i32x4_shl(af, 1), as_), v128_and(af, i32x4_splat(1)));

        // compute scaling factor
        let ss = f32x4_div(f32x4_splat(65535.0), f32x4_convert_i32x4(as_));

        // convert to RGB in fixed point
        let rf = i32x4_add(yf, i32x4_sub(cof, cgf));
        let gf = i32x4_add(yf, cgf);
        let bf = i32x4_sub(yf, i32x4_add(cof, cgf));

        let fsnap = f32x4_splat(FSNAP);

        let rr = f32x4_add(f32x4_mul(f32x4_convert_i32x4(rf), ss), fsnap);
        let gr = f32x4_add(f32x4_mul(f32x4_convert_i32x4(gf), ss), fsnap);
        let br = f32x4_add(f32x4_mul(f32x4_convert_i32x4(bf), ss), fsnap);
        let ar = f32x4_add(f32x4_mul(f32x4_convert_i32x4(af), ss), fsnap);

        // mix r/b and g/a to make 16-bit unpack easier
        let rbr = v128_or(v128_and(rr, i32x4_splat(0xffff)), i32x4_shl(br, 16));
        let gar = v128_or(v128_and(gr, i32x4_splat(0xffff)), i32x4_shl(ar, 16));

        // pack r/g/b/a using 16-bit unpacks
        store16(b0, unpacklo16(rbr, gar));
        store16(b1, unpackhi16(rbr, gar));
    }
}

/// Wasm SIMD is selected at compile time; see [`VertexDecoder`].
#[derive(Clone, Copy)]
pub(crate) struct FilterDecoder(());

impl FilterDecoder {
    pub(crate) fn detect() -> Option<Self> {
        Some(FilterDecoder(()))
    }

    /// `stride` is 4 (8-bit) or 8 (16-bit components).
    pub(crate) fn decode_oct(self, data: &mut [u8], count: usize, stride: usize) {
        if stride == 4 {
            dispatch::<4>(data, count, |d, c| decode_oct8(d, c));
        } else {
            dispatch::<8>(data, count, |d, c| decode_oct16(d, c));
        }
    }

    pub(crate) fn decode_quat(self, data: &mut [u8], count: usize) {
        dispatch::<8>(data, count, |d, c| decode_quat(d, c));
    }

    /// `count` 32-bit values.
    pub(crate) fn decode_exp(self, data: &mut [u8], count: usize) {
        dispatch::<4>(data, count, |d, c| decode_exp(d, c));
    }

    pub(crate) fn decode_color(self, data: &mut [u8], count: usize, stride: usize) {
        if stride == 4 {
            dispatch::<4>(data, count, |d, c| decode_color8(d, c));
        } else {
            dispatch::<8>(data, count, |d, c| decode_color16(d, c));
        }
    }
}

/// C++ has no Wasm meshlet decoder: uninhabited, so meshlets always use the scalar path.
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
