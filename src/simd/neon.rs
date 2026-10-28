//! AArch64 NEON paths (`SIMD_NEON`): vertex block decoding (`decodeVertexBlockSimd` with `SIMD_LATENCYOPT`), vertex
//! filter decoding (`decodeFilter*Simd`) and meshlet decoding (`decodeMeshletSimd`).

use std::arch::aarch64::*;

use super::common::{GROUP_COUNT, GROUP_SHUFFLE, TRIANGLE_TABLES, VERTEX_TABLES, blocks8, dispatch, window, window_mut};
use crate::index::IndexType;
use crate::meshletcodec::MeshletTriangleType;
use crate::vertexcodec::{BYTE_GROUP_DECODE_LIMIT, BYTE_GROUP_SIZE, DecodeScratch, VERTEX_BLOCK_MAX_SIZE};

/// 8 bytes into a register (`vld1_u8`).
#[inline]
#[target_feature(enable = "neon")]
fn load8(b: &[u8; 8]) -> uint8x8_t {
    vcreate_u8(u64::from_le_bytes(*b))
}

/// 16 bytes into a register (`vld1q_u8`).
#[inline]
#[target_feature(enable = "neon")]
fn load16(b: &[u8; 16]) -> uint8x16_t {
    let (c, _) = b.as_chunks::<8>();
    vcombine_u8(vcreate_u8(u64::from_le_bytes(c[0])), vcreate_u8(u64::from_le_bytes(c[1])))
}

/// Register into 16 bytes (`vst1q_u8`).
#[inline]
#[target_feature(enable = "neon")]
fn store16(o: &mut [u8; 16], v: uint8x16_t) {
    let v = vreinterpretq_u64_u8(v);
    let (c, _) = o.as_chunks_mut::<8>();
    c[0] = vgetq_lane_u64::<0>(v).to_le_bytes();
    c[1] = vgetq_lane_u64::<1>(v).to_le_bytes();
}

#[inline]
#[target_feature(enable = "neon")]
fn load16s(b: &[u8; 16]) -> int32x4_t {
    vreinterpretq_s32_u8(load16(b))
}

#[inline]
#[target_feature(enable = "neon")]
fn store16s(o: &mut [u8; 16], v: int32x4_t) {
    store16(o, vreinterpretq_u8_s32(v));
}

/// `shuffleBytes`
#[inline]
#[target_feature(enable = "neon")]
fn shuffle_bytes(mask0: u8, mask1: u8, rest0: uint8x8_t, rest1: uint8x8_t) -> uint8x16_t {
    let sm0 = load8(&GROUP_SHUFFLE[usize::from(mask0)]);
    let sm1 = load8(&GROUP_SHUFFLE[usize::from(mask1)]);

    let r0 = vtbl1_u8(rest0, sm0);
    let r1 = vtbl1_u8(rest1, sm1);

    vcombine_u8(r0, r1)
}

/// `neonMoveMask`
#[inline]
#[target_feature(enable = "neon")]
fn move_mask(mask: uint8x16_t) -> (u8, u8) {
    // magic constant found using z3 SMT assuming mask has 8 groups of 0xff or 0x00
    const MAGIC: u64 = 0x000103070f1f3f80;

    let mask2 = vreinterpretq_u64_u8(mask);

    let mask0 = (vgetq_lane_u64::<0>(mask2).wrapping_mul(MAGIC) >> 56) as u8;
    let mask1 = (vgetq_lane_u64::<1>(mask2).wrapping_mul(MAGIC) >> 56) as u8;
    (mask0, mask1)
}

/// `decodeBytesGroupSimd`: decodes one 16-byte group from `w`, returns the number of bytes consumed.
#[target_feature(enable = "neon")]
fn decode_bytes_group(w: &[u8; BYTE_GROUP_DECODE_LIMIT], buffer: &mut [u8; BYTE_GROUP_SIZE], hbits: usize) -> Option<usize> {
    let at8 = |at: usize| window::<8>(w, at);

    match hbits {
        0 | 4 => {
            store16(buffer, vdupq_n_u8(0));
            Some(0)
        }
        1 | 6 => {
            let (head, _) = w.as_chunks::<4>();
            let mut data32 = u32::from_le_bytes(head[0]);
            data32 &= data32 >> 1;

            // arrange bits such that low bits of nibbles of data64 contain all 2-bit elements of data32
            let data64 = (u64::from(data32) << 30) | u64::from(data32);

            // adds all 1-bit nibbles together; the sum fits in 4 bits because datacnt=16 would have used a cheaper mode
            let datacnt = ((data64 & 0x1111111111111111).wrapping_mul(0x1111111111111111) >> 60) as usize;

            let sel2 = load8(at8(0)?);
            let sel22 = vzip_u8(vshr_n_u8::<4>(sel2), sel2).0;
            let sel2222 = vzip_u8(vshr_n_u8::<2>(sel22), sel22);
            let sel = vandq_u8(vcombine_u8(sel2222.0, sel2222.1), vdupq_n_u8(3));

            let mask = vceqq_u8(sel, vdupq_n_u8(3));
            let (mask0, mask1) = move_mask(mask);

            let rest0 = load8(at8(4)?);
            let rest1 = load8(at8(4 + usize::from(GROUP_COUNT[usize::from(mask0)]))?);

            let result = vbslq_u8(mask, shuffle_bytes(mask0, mask1, rest0, rest1), sel);

            store16(buffer, result);

            Some(4 + datacnt)
        }
        2 | 7 => {
            let mut data64 = u64::from_le_bytes(*at8(0)?);
            data64 &= data64 >> 1;
            data64 &= data64 >> 2;

            // adds all 1-bit nibbles together; the sum fits in 4 bits because datacnt=16 would have used a cheaper mode
            let datacnt = ((data64 & 0x1111111111111111).wrapping_mul(0x1111111111111111) >> 60) as usize;

            let sel4 = load8(at8(0)?);
            let sel44 = vzip_u8(vshr_n_u8::<4>(sel4), vand_u8(sel4, vdup_n_u8(15)));
            let sel = vcombine_u8(sel44.0, sel44.1);

            let mask = vceqq_u8(sel, vdupq_n_u8(15));
            let (mask0, mask1) = move_mask(mask);

            let rest0 = load8(at8(8)?);
            let rest1 = load8(at8(8 + usize::from(GROUP_COUNT[usize::from(mask0)]))?);

            let result = vbslq_u8(mask, shuffle_bytes(mask0, mask1, rest0, rest1), sel);

            store16(buffer, result);

            Some(8 + datacnt)
        }
        3 | 8 => {
            store16(buffer, load16(window::<16>(w, 0)?));
            Some(16)
        }
        _ => {
            // 1-bit groups (v1 only): the two header bytes are the sentinel masks themselves
            let (mask0, mask1) = (w[0], w[1]);
            let count0 = usize::from(GROUP_COUNT[usize::from(mask0)]);

            let rest0 = load8(at8(2)?);
            let rest1 = load8(at8(2 + count0)?);

            store16(buffer, shuffle_bytes(mask0, mask1, rest0, rest1));

            Some(2 + count0 + usize::from(GROUP_COUNT[usize::from(mask1)]))
        }
    }
}

/// `decodeBytesSimd`: `buffer_size` (a multiple of 16) bytes into `buffer`; returns the new data position.
#[target_feature(enable = "neon")]
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
#[target_feature(enable = "neon")]
fn transpose8(x: &mut [uint8x16_t; 4]) {
    let t01 = vzipq_u8(x[0], x[1]);
    let t23 = vzipq_u8(x[2], x[3]);

    let x01 = vzipq_u16(vreinterpretq_u16_u8(t01.0), vreinterpretq_u16_u8(t23.0));
    let x23 = vzipq_u16(vreinterpretq_u16_u8(t01.1), vreinterpretq_u16_u8(t23.1));

    x[0] = vreinterpretq_u8_u16(x01.0);
    x[1] = vreinterpretq_u8_u16(x01.1);
    x[2] = vreinterpretq_u8_u16(x23.0);
    x[3] = vreinterpretq_u8_u16(x23.1);
}

#[inline]
#[target_feature(enable = "neon")]
fn unzigzag8(v: uint8x16_t) -> uint8x16_t {
    let xl = vreinterpretq_u8_s8(vnegq_s8(vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(1)))));
    let xr = vshrq_n_u8::<1>(v);

    veorq_u8(xl, xr)
}

#[inline]
#[target_feature(enable = "neon")]
fn unzigzag16(v: uint8x16_t) -> uint8x16_t {
    let vv = vreinterpretq_u16_u8(v);
    let xl = vreinterpretq_u8_s16(vnegq_s16(vreinterpretq_s16_u16(vandq_u16(vv, vdupq_n_u16(1)))));
    let xr = vreinterpretq_u8_u16(vshrq_n_u16::<1>(vv));

    veorq_u8(xl, xr)
}

#[inline]
#[target_feature(enable = "neon")]
fn rotate32(v: uint8x16_t, r: i32) -> uint8x16_t {
    let v32 = vreinterpretq_u32_u8(v);
    vreinterpretq_u8_u32(vorrq_u32(vshlq_u32(v32, vdupq_n_s32(r)), vshlq_u32(v32, vdupq_n_s32(r - 32))))
}

/// `decodeDeltas4Simd<Channel>`: 16 vertices at a time; 4 transposed byte lanes, then a running delta per vertex.
#[target_feature(enable = "neon")]
fn decode_deltas4<const CHANNEL: u8>(
    buffer: &[u8],
    transposed: &mut [u8],
    k: usize,
    vertex_count_aligned: usize,
    vertex_size: usize,
    last_vertex: &[u8; 4],
    rot: i32,
) -> Option<()> {
    let mut pi = vcreate_u8(u64::from(u32::from_le_bytes(*last_vertex)));

    let mut savep = k;

    for j in (0..vertex_count_aligned).step_by(16) {
        let mut r = [vdupq_n_u8(0); 4];
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

            let lo = vget_low_u8(r);
            let hi = vget_high_u8(r);
            let dup1 = |v: uint8x8_t| vreinterpret_u8_u32(vdup_lane_u32::<1>(vreinterpret_u32_u8(v)));
            let t = [lo, dup1(lo), hi, dup1(hi)];

            for t in t {
                pi = match CHANNEL {
                    0 => vadd_u8(pi, t),
                    1 => vreinterpret_u8_u16(vadd_u16(vreinterpret_u16_u8(pi), vreinterpret_u16_u8(t))),
                    _ => veor_u8(pi, t),
                };

                *window_mut::<4>(transposed, savep)? = vget_lane_u32::<0>(vreinterpret_u32_u8(pi)).to_le_bytes();
                savep += vertex_size;
            }
        }
    }

    Some(())
}

/// `decodeVertexBlockSimd`
#[target_feature(enable = "neon")]
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
                i32::from((32 - (channel >> 4)) & 31),
            )?,
            _ => return None, // invalid channel type
        }
    }

    vertex_data[..vertex_count * vertex_size].copy_from_slice(&transposed[..vertex_count * vertex_size]);

    let last = vertex_size * (vertex_count - 1);
    last_vertex[..vertex_size].copy_from_slice(&transposed[last..last + vertex_size]);

    Some(pos)
}

/// Proof that the CPU supports NEON (always true on AArch64; C++ enables these paths unconditionally).
#[derive(Clone, Copy)]
pub(crate) struct VertexDecoder(());

impl VertexDecoder {
    pub(crate) fn detect() -> Option<Self> {
        std::arch::is_aarch64_feature_detected!("neon").then_some(VertexDecoder(()))
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
        // SAFETY: `self` exists only if `detect` found neon, the feature the callee is compiled for
        #[allow(unsafe_code)]
        unsafe {
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
}

const SIGN: i32 = i32::MIN;

/// fast rounded signed float->int: addition triggers renormalization after which mantissa stores the integer value
/// note: the result is offset by 0x4B40_0000, but only the low 8/16 bits are used so the subtraction is omitted
const FSNAP: f32 = (3 << 22) as f32;

/// `decodeFilterOctSimd8`
#[target_feature(enable = "neon")]
fn decode_oct8(data: &mut [u8], count: usize) {
    let sign = vdupq_n_s32(SIGN);

    for block in data[..count * 4].as_chunks_mut::<16>().0 {
        let n4 = load16s(block);

        // sign-extends each of x,y in [x y ? ?] with arithmetic shifts
        let xf = vshrq_n_s32::<24>(vshlq_n_s32::<24>(n4));
        let yf = vshrq_n_s32::<24>(vshlq_n_s32::<16>(n4));

        // unpack z; note that z is unsigned so we technically don't need to sign extend it
        let zf = vshrq_n_s32::<24>(vshlq_n_s32::<8>(n4));

        // convert x and y to floats and reconstruct z; this assumes zf encodes 1.f at the same bit count
        let mut x = vcvtq_f32_s32(xf);
        let mut y = vcvtq_f32_s32(yf);
        let z = vsubq_f32(vcvtq_f32_s32(zf), vaddq_f32(vabsq_f32(x), vabsq_f32(y)));

        // fixup octahedral coordinates for z<0
        let t = vminq_f32(z, vdupq_n_f32(0.0));

        x = vaddq_f32(
            x,
            vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(t), vandq_s32(vreinterpretq_s32_f32(x), sign))),
        );
        y = vaddq_f32(
            y,
            vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(t), vandq_s32(vreinterpretq_s32_f32(y), sign))),
        );

        // compute normal length & scale
        let ll = vfmaq_f32(vfmaq_f32(vmulq_f32(x, x), y, y), z, z);
        let rl = vrsqrteq_f32(ll);
        let s = vmulq_f32(vdupq_n_f32(127.0), rl);

        let fsnap = vdupq_n_f32(FSNAP);

        let xr = vreinterpretq_s32_f32(vfmaq_f32(fsnap, x, s));
        let yr = vreinterpretq_s32_f32(vfmaq_f32(fsnap, y, s));
        let zr = vreinterpretq_s32_f32(vfmaq_f32(fsnap, z, s));

        // combine xr/yr/zr into final value
        let mut res = vsliq_n_s32::<8>(xr, vsliq_n_s32::<8>(yr, zr));
        res = vbslq_s32(vdupq_n_u32(0xff000000), n4, res);

        store16s(block, res);
    }
}

/// `decodeFilterOctSimd16`
#[target_feature(enable = "neon")]
fn decode_oct16(data: &mut [u8], count: usize) {
    let sign = vdupq_n_s32(SIGN);

    for [b0, b1] in blocks8(data, count) {
        let n4_0 = load16s(b0);
        let n4_1 = load16s(b1);

        // gather both x/y 16-bit pairs in each 32-bit lane
        let n4 = vuzpq_s32(n4_0, n4_1).0;

        // sign-extends each of x,y in [x y] with arithmetic shifts
        let xf = vshrq_n_s32::<16>(vshlq_n_s32::<16>(n4));
        let yf = vshrq_n_s32::<16>(n4);

        // unpack z; note that z is unsigned so we don't need to sign extend it
        let z4 = vuzpq_s32(n4_0, n4_1).1;
        let zf = vandq_s32(z4, vdupq_n_s32(0x7fff));

        // convert x and y to floats and reconstruct z; this assumes zf encodes 1.f at the same bit count
        let mut x = vcvtq_f32_s32(xf);
        let mut y = vcvtq_f32_s32(yf);
        let z = vsubq_f32(vcvtq_f32_s32(zf), vaddq_f32(vabsq_f32(x), vabsq_f32(y)));

        // fixup octahedral coordinates for z<0
        let t = vminq_f32(z, vdupq_n_f32(0.0));

        x = vaddq_f32(
            x,
            vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(t), vandq_s32(vreinterpretq_s32_f32(x), sign))),
        );
        y = vaddq_f32(
            y,
            vreinterpretq_f32_s32(veorq_s32(vreinterpretq_s32_f32(t), vandq_s32(vreinterpretq_s32_f32(y), sign))),
        );

        // compute normal length & scale
        let ll = vfmaq_f32(vfmaq_f32(vmulq_f32(x, x), y, y), z, z);
        let s = vdivq_f32(vdupq_n_f32(32767.0), vsqrtq_f32(ll));

        let fsnap = vdupq_n_f32(FSNAP);

        let xr = vreinterpretq_s32_f32(vfmaq_f32(fsnap, x, s));
        let yr = vreinterpretq_s32_f32(vfmaq_f32(fsnap, y, s));
        let zr = vreinterpretq_s32_f32(vfmaq_f32(fsnap, z, s));

        // mix x/z and y/0 to make 16-bit unpack easier
        let xzr = vsliq_n_s32::<16>(xr, zr);
        let y0r = vandq_s32(yr, vdupq_n_s32(0xffff));

        // pack x/y/z using 16-bit unpacks; note that this has 0 where we should have .w
        let zip = vzipq_s16(vreinterpretq_s16_s32(xzr), vreinterpretq_s16_s32(y0r));
        let mut res_0 = vreinterpretq_s32_s16(zip.0);
        let mut res_1 = vreinterpretq_s32_s16(zip.1);

        // patch in .w
        let maskw = vreinterpretq_u32_u64(vdupq_n_u64(0xffff000000000000));
        res_0 = vbslq_s32(maskw, n4_0, res_0);
        res_1 = vbslq_s32(maskw, n4_1, res_1);

        store16s(b0, res_0);
        store16s(b1, res_1);
    }
}

/// `decodeFilterQuatSimd`
#[target_feature(enable = "neon")]
fn decode_quat(data: &mut [u8], count: usize) {
    let scale = 32767.0 / 2.0f32.sqrt();

    for [b0, b1] in blocks8(data, count) {
        // the component indices (low 2 bits of each .w) select the output rotation
        let w = |b: &[u8; 16], at: usize| i32::from(i16::from_le_bytes([b[at], b[at + 1]]));
        let qc = [w(b0, 6), w(b0, 14), w(b1, 6), w(b1, 14)];

        let q4_0 = load16s(b0);
        let q4_1 = load16s(b1);

        // gather both x/y 16-bit pairs in each 32-bit lane
        let q4_xy = vuzpq_s32(q4_0, q4_1).0;
        let q4_zc = vuzpq_s32(q4_0, q4_1).1;

        // sign-extends each of x,y in [x y] with arithmetic shifts
        let xf = vshrq_n_s32::<16>(vshlq_n_s32::<16>(q4_xy));
        let yf = vshrq_n_s32::<16>(q4_xy);
        let zf = vshrq_n_s32::<16>(vshlq_n_s32::<16>(q4_zc));
        let cf = vshrq_n_s32::<16>(q4_zc);

        // get a floating-point scaler using zc with bottom 2 bits set to 1 (which represents 1.f)
        let sf = vorrq_s32(cf, vdupq_n_s32(3));
        let s = vcvtq_f32_s32(sf);

        // convert x/y/z to floating point (unscaled! implied scale of 1/sqrt(2.f) * 1/sf)
        let x = vcvtq_f32_s32(xf);
        let y = vcvtq_f32_s32(yf);
        let z = vcvtq_f32_s32(zf);

        // reconstruct w as a square root (unscaled); we clamp to 0.f to avoid NaN due to precision errors
        let ws = vmulq_f32(s, s);
        let ww = vsubq_f32(vaddq_f32(ws, ws), vfmaq_f32(vfmaq_f32(vmulq_f32(x, x), y, y), z, z));
        let w = vsqrtq_f32(vmaxq_f32(ww, vdupq_n_f32(0.0)));

        // compute final scale; note that all computations above are unscaled
        // we need to divide by sf to get out of fixed point, divide by sqrt(2) to renormalize and multiply by 32767 to get to int16 range
        let ss = vdivq_f32(vdupq_n_f32(scale), s);

        let fsnap = vdupq_n_f32(FSNAP);

        let xr = vreinterpretq_s32_f32(vfmaq_f32(fsnap, x, ss));
        let yr = vreinterpretq_s32_f32(vfmaq_f32(fsnap, y, ss));
        let zr = vreinterpretq_s32_f32(vfmaq_f32(fsnap, z, ss));
        let wr = vreinterpretq_s32_f32(vfmaq_f32(fsnap, w, ss));

        // mix x/z and w/y to make 16-bit unpack easier
        let xzr = vsliq_n_s32::<16>(xr, zr);
        let wyr = vsliq_n_s32::<16>(wr, yr);

        // pack x/y/z/w using 16-bit unpacks; we pack wxyz by default (for qc=0)
        let zip = vzipq_s16(vreinterpretq_s16_s32(wyr), vreinterpretq_s16_s32(xzr));

        let mut res = [[0u8; 16]; 2];
        store16(&mut res[0], vreinterpretq_u8_s16(zip.0));
        store16(&mut res[1], vreinterpretq_u8_s16(zip.1));

        // rotate and store
        let outs = b0.as_chunks_mut::<8>().0.iter_mut().chain(b1.as_chunks_mut::<8>().0);
        for ((out, v), qc) in outs.zip(res.as_flattened().as_chunks::<8>().0).zip(qc) {
            *out = u64::from_le_bytes(*v).rotate_left((qc << 4) as u32).to_le_bytes();
        }
    }
}

/// `decodeFilterExpSimd`
#[target_feature(enable = "neon")]
fn decode_exp(data: &mut [u8], count: usize) {
    for block in data[..count * 4].as_chunks_mut::<16>().0 {
        let v = load16s(block);

        // decode exponent into 2^x directly
        let ef = vshrq_n_s32::<24>(v);
        let es = vshlq_n_s32::<23>(vaddq_s32(ef, vdupq_n_s32(127)));

        // decode 24-bit mantissa into floating-point value
        let mf = vshrq_n_s32::<8>(vshlq_n_s32::<8>(v));
        let m = vcvtq_f32_s32(mf);

        let r = vmulq_f32(vreinterpretq_f32_s32(es), m);

        store16s(block, vreinterpretq_s32_f32(r));
    }
}

/// `decodeFilterColorSimd8`
#[target_feature(enable = "neon")]
fn decode_color8(data: &mut [u8], count: usize) {
    for block in data[..count * 4].as_chunks_mut::<16>().0 {
        let c4 = load16s(block);

        // unpack y/co/cg/a (co/cg are sign extended with arithmetic shifts)
        let yf = vandq_s32(c4, vdupq_n_s32(0xff));
        let cof = vshrq_n_s32::<24>(vshlq_n_s32::<16>(c4));
        let cgf = vshrq_n_s32::<24>(vshlq_n_s32::<8>(c4));
        let mut af = vreinterpretq_s32_u32(vshrq_n_u32::<24>(vreinterpretq_u32_s32(c4)));

        // recover scale from alpha high bit
        let mut as_ = af;
        as_ = vorrq_s32(as_, vshrq_n_s32::<1>(as_));
        as_ = vorrq_s32(as_, vshrq_n_s32::<2>(as_));
        as_ = vorrq_s32(as_, vshrq_n_s32::<4>(as_));

        // expand alpha by one bit to match other components
        af = vorrq_s32(vandq_s32(vshlq_n_s32::<1>(af), as_), vandq_s32(af, vdupq_n_s32(1)));

        // compute scaling factor
        let ss = vmulq_f32(vdupq_n_f32(255.0), vrecpeq_f32(vcvtq_f32_s32(as_)));

        // convert to RGB in fixed point
        let rf = vaddq_s32(yf, vsubq_s32(cof, cgf));
        let gf = vaddq_s32(yf, cgf);
        let bf = vsubq_s32(yf, vaddq_s32(cof, cgf));

        let fsnap = vdupq_n_f32(FSNAP);

        let rr = vreinterpretq_s32_f32(vfmaq_f32(fsnap, vcvtq_f32_s32(rf), ss));
        let gr = vreinterpretq_s32_f32(vfmaq_f32(fsnap, vcvtq_f32_s32(gf), ss));
        let br = vreinterpretq_s32_f32(vfmaq_f32(fsnap, vcvtq_f32_s32(bf), ss));
        let ar = vreinterpretq_s32_f32(vfmaq_f32(fsnap, vcvtq_f32_s32(af), ss));

        // repack rgba into final value
        let res = vsliq_n_s32::<8>(rr, vsliq_n_s32::<8>(gr, vsliq_n_s32::<8>(br, ar)));

        store16s(block, res);
    }
}

/// `decodeFilterColorSimd16`
#[target_feature(enable = "neon")]
fn decode_color16(data: &mut [u8], count: usize) {
    for [b0, b1] in blocks8(data, count) {
        let c4_0 = load16s(b0);
        let c4_1 = load16s(b1);

        // gather both y/co 16-bit pairs in each 32-bit lane
        let c4_yco = vuzpq_s32(c4_0, c4_1).0;
        let c4_cga = vuzpq_s32(c4_0, c4_1).1;

        // unpack y/co/cg/a components (co/cg are sign extended with arithmetic shifts)
        let yf = vandq_s32(c4_yco, vdupq_n_s32(0xffff));
        let cof = vshrq_n_s32::<16>(c4_yco);
        let cgf = vshrq_n_s32::<16>(vshlq_n_s32::<16>(c4_cga));
        let mut af = vreinterpretq_s32_u32(vshrq_n_u32::<16>(vreinterpretq_u32_s32(c4_cga)));

        // recover scale from alpha high bit
        let mut as_ = af;
        as_ = vorrq_s32(as_, vshrq_n_s32::<1>(as_));
        as_ = vorrq_s32(as_, vshrq_n_s32::<2>(as_));
        as_ = vorrq_s32(as_, vshrq_n_s32::<4>(as_));
        as_ = vorrq_s32(as_, vshrq_n_s32::<8>(as_));

        // expand alpha by one bit to match other components
        af = vorrq_s32(vandq_s32(vshlq_n_s32::<1>(af), as_), vandq_s32(af, vdupq_n_s32(1)));

        // compute scaling factor
        let ss = vdivq_f32(vdupq_n_f32(65535.0), vcvtq_f32_s32(as_));

        // convert to RGB in fixed point
        let rf = vaddq_s32(yf, vsubq_s32(cof, cgf));
        let gf = vaddq_s32(yf, cgf);
        let bf = vsubq_s32(yf, vaddq_s32(cof, cgf));

        let fsnap = vdupq_n_f32(FSNAP);

        let rr = vreinterpretq_s32_f32(vfmaq_f32(fsnap, vcvtq_f32_s32(rf), ss));
        let gr = vreinterpretq_s32_f32(vfmaq_f32(fsnap, vcvtq_f32_s32(gf), ss));
        let br = vreinterpretq_s32_f32(vfmaq_f32(fsnap, vcvtq_f32_s32(bf), ss));
        let ar = vreinterpretq_s32_f32(vfmaq_f32(fsnap, vcvtq_f32_s32(af), ss));

        // mix r/b and g/a to make 16-bit unpack easier
        let rbr = vsliq_n_s32::<16>(rr, br);
        let gar = vsliq_n_s32::<16>(gr, ar);

        // pack r/g/b/a using 16-bit unpacks
        let zip = vzipq_s16(vreinterpretq_s16_s32(rbr), vreinterpretq_s16_s32(gar));

        store16s(b0, vreinterpretq_s32_s16(zip.0));
        store16s(b1, vreinterpretq_s32_s16(zip.1));
    }
}

/// Proof that the CPU supports NEON (always true on AArch64; C++ enables these paths unconditionally).
#[derive(Clone, Copy)]
pub(crate) struct FilterDecoder(());

#[allow(unsafe_code)]
impl FilterDecoder {
    pub(crate) fn detect() -> Option<Self> {
        std::arch::is_aarch64_feature_detected!("neon").then_some(FilterDecoder(()))
    }

    /// `stride` is 4 (8-bit) or 8 (16-bit components).
    pub(crate) fn decode_oct(self, data: &mut [u8], count: usize, stride: usize) {
        // SAFETY (all methods): `self` exists only if `detect` found neon, the feature the callees are compiled for
        if stride == 4 {
            dispatch::<4>(data, count, |d, c| unsafe { decode_oct8(d, c) });
        } else {
            dispatch::<8>(data, count, |d, c| unsafe { decode_oct16(d, c) });
        }
    }

    pub(crate) fn decode_quat(self, data: &mut [u8], count: usize) {
        dispatch::<8>(data, count, |d, c| unsafe { decode_quat(d, c) });
    }

    /// `count` 32-bit values.
    pub(crate) fn decode_exp(self, data: &mut [u8], count: usize) {
        dispatch::<4>(data, count, |d, c| unsafe { decode_exp(d, c) });
    }

    pub(crate) fn decode_color(self, data: &mut [u8], count: usize, stride: usize) {
        if stride == 4 {
            dispatch::<4>(data, count, |d, c| unsafe { decode_color8(d, c) });
        } else {
            dispatch::<8>(data, count, |d, c| unsafe { decode_color16(d, c) });
        }
    }
}

/// `decodeTriangleGroup`
#[inline]
#[target_feature(enable = "neon")]
fn decode_triangle_group(state: uint8x16_t, code: u8, extra: &[u8; 8]) -> uint8x16_t {
    let shuf = load16(&TRIANGLE_TABLES.0[usize::from(code)]);
    let next = vextq_u8::<6>(vdupq_n_u8(0), shuf);

    // patch first 6 bytes with current extra and roll state forward
    let ext = vcombine_u8(load8(extra), vdup_n_u8(0));
    let state = vbslq_u8(vcombine_u8(vcreate_u8(0xffffffffffff), vdup_n_u8(0)), ext, state);
    vaddq_u8(vqtbl1q_u8(state, shuf), next)
}

/// `decodeVertexGroup`
#[inline]
#[target_feature(enable = "neon")]
fn decode_vertex_group(last: uint32x4_t, code: u8, data: &[u8; 16]) -> uint32x4_t {
    let word = load16(data);
    let shuf = load16(&VERTEX_TABLES.0[usize::from(code)]);

    let v = vreinterpretq_u32_u8(vqtbl1q_u8(word, shuf));

    // unzigzag+1
    let xl = vsubq_u32(vdupq_n_u32(0), vandq_u32(v, vdupq_n_u32(1)));
    let xr = vshrq_n_u32::<1>(v);
    let mut x = vaddq_u32(veorq_u32(xl, xr), vdupq_n_u32(1));

    // prefix sum
    x = vaddq_u32(x, vextq_u32::<2>(vdupq_n_u32(0), x));
    x = vaddq_u32(x, vextq_u32::<3>(vdupq_n_u32(0), x));
    vaddq_u32(x, vdupq_n_u32(vgetq_lane_u32::<3>(last)))
}

/// `decodeTrianglesSimd`; tails are written exactly (see the x86 version).
#[target_feature(enable = "neon")]
fn decode_meshlet_triangles<T: MeshletTriangleType>(
    triangles: &mut [T],
    codes: &[u8],
    buffer: &[u8],
    mut extra: usize,
    bound: usize,
    triangle_count: usize,
) -> Option<usize> {
    let mut state = vdupq_n_u8(0);

    for (i, &code) in codes[..triangle_count.div_ceil(2)].iter().enumerate() {
        // each triangle pair reads <=6 bytes from extra; the gap guarantees 16 bytes of overread past bound
        if extra > bound {
            return None;
        }

        state = decode_triangle_group(state, code, window::<8>(buffer, extra)?);
        extra += usize::from(TRIANGLE_TABLES.1[usize::from(code)]);

        // state bytes 9..11 and 12..14 hold the two decoded triangles; the last pair may hold a single one
        let tris = vgetq_lane_u64::<0>(vreinterpretq_u64_u8(vextq_u8::<9>(state, vdupq_n_u8(0))));
        T::write_packed(triangles, i * 2, (triangle_count - i * 2).min(2), tris);
    }

    Some(extra)
}

/// `decodeVerticesSimd`; tails are written exactly (see the x86 version).
#[target_feature(enable = "neon")]
fn decode_meshlet_vertices<V: IndexType>(
    vertices: &mut [V],
    ctrl: &[u8],
    buffer: &[u8],
    mut data: usize,
    bound: usize,
    vertex_count: usize,
) -> Option<usize> {
    let mut last = vdupq_n_u32(!0);

    let (groups, tail) = vertices[..vertex_count].as_chunks_mut::<4>();
    let tail = (!tail.is_empty()).then_some(tail);

    for (i, out) in groups.iter_mut().map(|g| g.as_mut_slice()).chain(tail).enumerate() {
        let code = ctrl[i];
        if data > bound {
            return None;
        }

        last = decode_vertex_group(last, code, window::<16>(buffer, data)?);
        data += usize::from(VERTEX_TABLES.1[usize::from(code)]);

        let mut v = [0u8; 16];
        store16(&mut v, vreinterpretq_u8_u32(last));

        for (o, v) in out.iter_mut().zip(v.as_chunks::<4>().0) {
            *o = V::from_u32(u32::from_le_bytes(*v));
        }
    }

    Some(data)
}

/// Proof that the CPU supports NEON.
#[derive(Clone, Copy)]
pub(crate) struct MeshletDecoder(());

#[allow(unsafe_code)]
impl MeshletDecoder {
    pub(crate) fn detect() -> Option<Self> {
        std::arch::is_aarch64_feature_detected!("neon").then_some(MeshletDecoder(()))
    }

    pub(crate) fn decode_vertices<V: IndexType>(
        self,
        vertices: &mut [V],
        ctrl: &[u8],
        buffer: &[u8],
        data: usize,
        bound: usize,
        vertex_count: usize,
    ) -> Option<usize> {
        // SAFETY: `self` exists only if `detect` found neon, the feature the callee is compiled for
        unsafe { decode_meshlet_vertices(vertices, ctrl, buffer, data, bound, vertex_count) }
    }

    pub(crate) fn decode_triangles<T: MeshletTriangleType>(
        self,
        triangles: &mut [T],
        codes: &[u8],
        buffer: &[u8],
        extra: usize,
        bound: usize,
        triangle_count: usize,
    ) -> Option<usize> {
        // SAFETY: as above
        unsafe { decode_meshlet_triangles(triangles, codes, buffer, extra, bound, triangle_count) }
    }
}
