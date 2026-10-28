//! x86-64 SIMD paths (`SIMD_SSE`): vertex block decoding with SSSE3 + POPCNT (`decodeVertexBlockSimd`, with
//! `SIMD_LATENCYOPT`), vertex filter decoding with SSE2 (`decodeFilter*Simd`) and meshlet decoding with SSE4.1
//! (`decodeMeshletSimd`).

use std::arch::x86_64::*;

use crate::index::IndexType;
use crate::meshletcodec::MeshletTriangleType;
use crate::vertexcodec::{BYTE_GROUP_DECODE_LIMIT, BYTE_GROUP_SIZE, DecodeScratch, VERTEX_BLOCK_MAX_SIZE};

/// 16 bytes into a register; compiles to `movdqu`.
#[inline]
#[target_feature(enable = "sse2")]
fn load16(b: &[u8; 16]) -> __m128i {
    let (c, _) = b.as_chunks::<8>();
    _mm_unpacklo_epi64(
        _mm_cvtsi64_si128(i64::from_le_bytes(c[0])),
        _mm_cvtsi64_si128(i64::from_le_bytes(c[1])),
    )
}

/// Register into 16 bytes; compiles to `movdqu`.
#[inline]
#[target_feature(enable = "sse2")]
fn store16(o: &mut [u8; 16], v: __m128i) {
    let (c, _) = o.as_chunks_mut::<8>();
    c[0] = _mm_cvtsi128_si64(v).to_le_bytes();
    c[1] = _mm_cvtsi128_si64(_mm_unpackhi_epi64(v, v)).to_le_bytes();
}

/// `slice[at..at + N]` as an array; `None` when out of bounds.
#[inline(always)]
fn window<const N: usize>(slice: &[u8], at: usize) -> Option<&[u8; N]> {
    slice.get(at..)?.first_chunk::<N>()
}

#[inline(always)]
fn window_mut<const N: usize>(slice: &mut [u8], at: usize) -> Option<&mut [u8; N]> {
    slice.get_mut(at..)?.first_chunk_mut::<N>()
}

/// `kDecodeBytesGroupShuffle`: for each 8-bit sentinel mask, the shuffle that expands consecutive bytes into the
/// masked lanes (0x80 zeroes the others).
const GROUP_SHUFFLE: [[u8; 8]; 256] = {
    let mut table = [[0u8; 8]; 256];
    let mut mask = 0;
    while mask < 256 {
        let mut count = 0;
        let mut i = 0;
        while i < 8 {
            let maski = (mask >> i) & 1;
            table[mask][i] = if maski != 0 { count } else { 0x80 };
            count += maski as u8;
            i += 1;
        }
        mask += 1;
    }
    table
};

/// `kDecodeBytesGroupConfig`: sentinel mask, replicating shuffle, and two multipliers (even/odd) for multishift
/// emulation, indexed by `hbits` (v0: 0..3 for 0/2/4/8 bits; v1: 4..8 for 0/1/2/4/8 bits).
const GROUP_CONFIG: [[[u8; 16]; 4]; 9] = {
    const Z: [u8; 16] = [0; 16];
    const NONE: [u8; 16] = [128; 16];
    const B2: [[u8; 16]; 4] = [
        [3; 16],
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        [4, 0, 64, 0, 4, 0, 64, 0, 4, 0, 64, 0, 4, 0, 64, 0],
        [16, 0, 0, 1, 16, 0, 0, 1, 16, 0, 0, 1, 16, 0, 0, 1],
    ];
    const B4: [[u8; 16]; 4] = [
        [15; 16],
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
        [16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0, 16, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ];
    [
        [[1; 16], NONE, Z, Z],
        B2,
        B4,
        [Z, NONE, Z, Z],
        [[1; 16], NONE, Z, Z],
        [
            [1; 16],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 64, 0, 16, 0, 4, 0, 0, 1, 64, 0, 16, 0, 4, 0],
            [128, 0, 32, 0, 8, 0, 2, 0, 128, 0, 32, 0, 8, 0, 2, 0],
        ],
        B2,
        B4,
        [Z, NONE, Z, Z],
    ]
};

/// `decodeBytesGroupSimd`: decodes one 16-byte group from `window`, returns the number of bytes consumed
/// (`skip + 16` never exceeds the window, so this is always `Some`).
#[target_feature(enable = "ssse3,popcnt")]
fn decode_bytes_group(window: &[u8; BYTE_GROUP_DECODE_LIMIT], buffer: &mut [u8; BYTE_GROUP_SIZE], hbits: usize) -> Option<usize> {
    // 0 for 1-bit, 1 for 2-bit, 2 for 4-bit, 3 for 8-bit, and 4 for 0-bit as it makes some of the uses easier
    const HBTN: [u32; 9] = [4, 1, 2, 3, 4, 0, 1, 2, 3];
    // one bit per group that is set if all group bits were 1
    const LANES: [u64; 9] = [0, 0x55555555, 0x1111111111111111, 0, 0, 0xffff, 0x55555555, 0x1111111111111111, 0];

    let n = HBTN[hbits];
    let config = &GROUP_CONFIG[hbits];

    let (head, _) = window.as_chunks::<8>();
    let data64 = u64::from_le_bytes(head[0]);

    // SIMD_LATENCYOPT: count sentinels from the bit fields directly so the next group doesn't wait for the shuffle
    let mut sentinels = data64 & (data64 >> n);
    sentinels &= sentinels >> (n >> 1);
    let datacnt = (sentinels & LANES[hbits]).count_ones() as usize;

    // for 8-bit groups, instead of loading the bytes through 'data', we load them through 'skip' as they are easier to preserve
    // for 0-bit groups, the load results get discarded because mask is always 0; in both cases the shift wraps to zero
    let skip = ((2 << n) & 15) as usize;
    let (_, rest) = window.split_at(skip);
    let rest = load16(rest.first_chunk::<16>()?);

    let selb = _mm_cvtsi64_si128(data64 as i64);

    // unpack 1, 2 or 4-bit values: shuffle replicates each source byte into both halves of a 16-bit lane
    // mulhi extracts even and odd fields into the low byte; the results are interleaved back with shift/or
    let selw = _mm_shuffle_epi8(selb, load16(&config[1]));
    let sel0 = _mm_mulhi_epu16(selw, load16(&config[2]));
    let sel1 = _mm_mulhi_epu16(selw, load16(&config[3]));
    let seli = _mm_or_si128(sel0, _mm_slli_epi16::<8>(sel1));

    // the interleaved fields are masked by the bit count (special handling: for 0/8-bit values, mul produces 0)
    let sent = load16(&config[0]);
    let sel = _mm_and_si128(seli, sent);

    // compare sel to sentinel; returns 0 for 0-bit (mul produces 0, sent is 1), 1 for 8-bit (mul produces 0, sent is 0)
    let mask = _mm_cmpeq_epi8(sel, sent);
    let mask16 = _mm_movemask_epi8(mask) as u32;
    let mask0 = (mask16 & 255) as usize;
    let mask1 = (mask16 >> 8) as usize;

    // decode shuffle mask from two halves; second half needs to be shifted by popcount(mask0)
    let sm0 = _mm_cvtsi64_si128(i64::from_le_bytes(GROUP_SHUFFLE[mask0]));
    let sm1 = _mm_cvtsi64_si128(i64::from_le_bytes(GROUP_SHUFFLE[mask1]));

    // each lane of mask is 0x00 or 0xff; sad yields 255*popcount(mask0) in low word => low byte is -popcount(mask0)
    let npops = _mm_sad_epu8(mask, _mm_setzero_si128());
    let sm1r = _mm_sub_epi8(sm1, _mm_shuffle_epi8(npops, _mm_setzero_si128()));
    let shuf = _mm_unpacklo_epi64(sm0, sm1r);

    // expand rest via shuffle mask and combine with sel; shuffle mask zeroes out bytes that are replaced by sel
    let result = _mm_or_si128(_mm_shuffle_epi8(rest, shuf), _mm_andnot_si128(mask, sel));

    store16(buffer, result);

    // datacnt is 0 for 8-bit groups so we can't use skip to advance; 0-bit groups wrap the shift to zero
    Some(((2 << n) & 31) as usize + datacnt)
}

/// `decodeBytesSimd`: `buffer_size` (a multiple of 16) bytes into `buffer`; returns the new data position.
#[target_feature(enable = "ssse3,popcnt")]
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
        let quad = &mut groups[i..i + 4];

        // very-fast-path: for consecutive 4 groups that are all 0-bit (v0/0, v1/0/0000) or 8-bit (v0/3333, v1/1/3333),
        // the branchless decoders are slower than branching over the decoding of 4 groups and issuing a few load/store ops
        if hshift != 5 && header_byte == 0 {
            quad.as_flattened_mut().fill(0);
        } else if hshift != 4 && header_byte == 255 {
            quad.as_flattened_mut().copy_from_slice(&data[pos..pos + BYTE_GROUP_SIZE * 4]);
            pos += BYTE_GROUP_SIZE * 4;
        } else {
            for (g, group) in quad.iter_mut().enumerate() {
                let w = window::<BYTE_GROUP_DECODE_LIMIT>(data, pos)?;
                pos += decode_bytes_group(w, group, hshift + usize::from((header_byte >> (g * 2)) & 3))?;
            }
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
#[target_feature(enable = "ssse3,popcnt")]
fn transpose8(x: &mut [__m128i; 4]) {
    let t0 = _mm_unpacklo_epi8(x[0], x[1]);
    let t1 = _mm_unpackhi_epi8(x[0], x[1]);
    let t2 = _mm_unpacklo_epi8(x[2], x[3]);
    let t3 = _mm_unpackhi_epi8(x[2], x[3]);

    x[0] = _mm_unpacklo_epi16(t0, t2);
    x[1] = _mm_unpackhi_epi16(t0, t2);
    x[2] = _mm_unpacklo_epi16(t1, t3);
    x[3] = _mm_unpackhi_epi16(t1, t3);
}

#[inline]
#[target_feature(enable = "ssse3,popcnt")]
fn unzigzag8(v: __m128i) -> __m128i {
    let xl = _mm_sub_epi8(_mm_setzero_si128(), _mm_and_si128(v, _mm_set1_epi8(1)));
    let xr = _mm_and_si128(_mm_srli_epi16::<1>(v), _mm_set1_epi8(127));

    _mm_xor_si128(xl, xr)
}

#[inline]
#[target_feature(enable = "ssse3,popcnt")]
fn unzigzag16(v: __m128i) -> __m128i {
    let xl = _mm_sub_epi16(_mm_setzero_si128(), _mm_and_si128(v, _mm_set1_epi16(1)));
    let xr = _mm_srli_epi16::<1>(v);

    _mm_xor_si128(xl, xr)
}

#[inline]
#[target_feature(enable = "ssse3,popcnt")]
fn rotate32(v: __m128i, r: i32) -> __m128i {
    // shifts by a register count: 32 - 0 shifts everything out, like _mm_srli_epi32(v, 32)
    _mm_or_si128(_mm_sll_epi32(v, _mm_cvtsi32_si128(r)), _mm_srl_epi32(v, _mm_cvtsi32_si128(32 - r)))
}

/// `decodeDeltas4Simd<Channel>`: 16 vertices at a time; 4 transposed byte lanes, then a running delta per vertex.
#[target_feature(enable = "ssse3,popcnt")]
fn decode_deltas4<const CHANNEL: u8>(
    buffer: &[u8],
    transposed: &mut [u8],
    k: usize,
    vertex_count_aligned: usize,
    vertex_size: usize,
    last_vertex: &[u8; 4],
    rot: i32,
) -> Option<()> {
    let mut pi = _mm_cvtsi32_si128(i32::from_le_bytes(*last_vertex));

    let mut savep = k;

    for j in (0..vertex_count_aligned).step_by(16) {
        let mut r = [_mm_setzero_si128(); 4];
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

            let t = [r, _mm_shuffle_epi32::<1>(r), _mm_shuffle_epi32::<2>(r), _mm_shuffle_epi32::<3>(r)];

            for t in t {
                pi = match CHANNEL {
                    0 => _mm_add_epi8(pi, t),
                    1 => _mm_add_epi16(pi, t),
                    _ => _mm_xor_si128(pi, t),
                };

                *window_mut::<4>(transposed, savep)? = _mm_cvtsi128_si32(pi).to_le_bytes();
                savep += vertex_size;
            }
        }
    }

    Some(())
}

/// `decodeVertexBlockSimd`
#[target_feature(enable = "ssse3,popcnt")]
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

/// Proof that the CPU supports SSSE3 and POPCNT (C++: `cpuid & cpumask`).
#[derive(Clone, Copy)]
pub(crate) struct VertexDecoder(());

impl VertexDecoder {
    pub(crate) fn detect() -> Option<Self> {
        (is_x86_feature_detected!("ssse3") && is_x86_feature_detected!("popcnt")).then_some(VertexDecoder(()))
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
        // SAFETY: `self` exists only if `detect` found ssse3 and popcnt, the features the callee is compiled for
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

/// `dispatchSimd`: runs `process` on whole groups of 4 elements, then on a zero-padded copy of the tail.
#[inline]
fn dispatch<const S: usize>(data: &mut [u8], count: usize, process: impl Fn(&mut [u8], usize)) {
    let count4 = count & !3;
    process(&mut data[..count4 * S], count4);

    if count4 < count {
        // max stride 4 x 32-bit, max count 4
        let mut tail = [0u8; 64];
        let tail_size = (count - count4) * S;

        tail[..tail_size].copy_from_slice(&data[count4 * S..count * S]);
        // C++ passes count - count4, but its loop always processes a whole group of 4
        process(&mut tail[..4 * S], 4);
        data[count4 * S..count * S].copy_from_slice(&tail[..tail_size]);
    }
}

#[inline]
#[target_feature(enable = "sse2")]
fn loadps(b: &[u8; 16]) -> __m128 {
    _mm_castsi128_ps(load16(b))
}

/// `count` elements of 8 bytes as blocks of 4 elements, each split into two 16-byte halves.
#[inline]
fn blocks8(data: &mut [u8], count: usize) -> &mut [[[u8; 16]; 2]] {
    data[..count * 8].as_chunks_mut::<16>().0.as_chunks_mut::<2>().0
}

/// `decodeFilterOctSimd8`
#[target_feature(enable = "sse2")]
fn decode_oct8(data: &mut [u8], count: usize) {
    let sign = _mm_set1_ps(-0.0);

    for block in data[..count * 4].as_chunks_mut::<16>().0 {
        let n4 = _mm_castps_si128(loadps(block));

        // sign-extends each of x,y in [x y ? ?] with arithmetic shifts
        let xf = _mm_srai_epi32::<24>(_mm_slli_epi32::<24>(n4));
        let yf = _mm_srai_epi32::<24>(_mm_slli_epi32::<16>(n4));

        // unpack z; note that z is unsigned so we technically don't need to sign extend it
        let zf = _mm_srai_epi32::<24>(_mm_slli_epi32::<8>(n4));

        // convert x and y to floats and reconstruct z; this assumes zf encodes 1.f at the same bit count
        let mut x = _mm_cvtepi32_ps(xf);
        let mut y = _mm_cvtepi32_ps(yf);
        let z = _mm_sub_ps(_mm_cvtepi32_ps(zf), _mm_add_ps(_mm_andnot_ps(sign, x), _mm_andnot_ps(sign, y)));

        // fixup octahedral coordinates for z<0
        let t = _mm_min_ps(z, _mm_setzero_ps());

        x = _mm_add_ps(x, _mm_xor_ps(t, _mm_and_ps(x, sign)));
        y = _mm_add_ps(y, _mm_xor_ps(t, _mm_and_ps(y, sign)));

        // compute normal length & scale
        let ll = _mm_add_ps(_mm_mul_ps(x, x), _mm_add_ps(_mm_mul_ps(y, y), _mm_mul_ps(z, z)));
        let s = _mm_mul_ps(_mm_set1_ps(127.0), _mm_rsqrt_ps(ll));

        // rounded signed float->int
        let xr = _mm_cvtps_epi32(_mm_mul_ps(x, s));
        let yr = _mm_cvtps_epi32(_mm_mul_ps(y, s));
        let zr = _mm_cvtps_epi32(_mm_mul_ps(z, s));

        // combine xr/yr/zr into final value
        let mut res = _mm_and_si128(n4, _mm_set1_epi32(0xff000000u32 as i32));
        res = _mm_or_si128(res, _mm_and_si128(xr, _mm_set1_epi32(0xff)));
        res = _mm_or_si128(res, _mm_slli_epi32::<8>(_mm_and_si128(yr, _mm_set1_epi32(0xff))));
        res = _mm_or_si128(res, _mm_slli_epi32::<16>(_mm_and_si128(zr, _mm_set1_epi32(0xff))));

        store16(block, res);
    }
}

/// `decodeFilterOctSimd16`
#[target_feature(enable = "sse2")]
fn decode_oct16(data: &mut [u8], count: usize) {
    let sign = _mm_set1_ps(-0.0);

    for [b0, b1] in blocks8(data, count) {
        let n4_0 = loadps(b0);
        let n4_1 = loadps(b1);

        // gather both x/y 16-bit pairs in each 32-bit lane
        let n4 = _mm_castps_si128(_mm_shuffle_ps::<0b10_00_10_00>(n4_0, n4_1));

        // sign-extends each of x,y in [x y] with arithmetic shifts
        let xf = _mm_srai_epi32::<16>(_mm_slli_epi32::<16>(n4));
        let yf = _mm_srai_epi32::<16>(n4);

        // unpack z; note that z is unsigned so we don't need to sign extend it
        let z4 = _mm_castps_si128(_mm_shuffle_ps::<0b11_01_11_01>(n4_0, n4_1));
        let zf = _mm_and_si128(z4, _mm_set1_epi32(0x7fff));

        // convert x and y to floats and reconstruct z; this assumes zf encodes 1.f at the same bit count
        let mut x = _mm_cvtepi32_ps(xf);
        let mut y = _mm_cvtepi32_ps(yf);
        let z = _mm_sub_ps(_mm_cvtepi32_ps(zf), _mm_add_ps(_mm_andnot_ps(sign, x), _mm_andnot_ps(sign, y)));

        // fixup octahedral coordinates for z<0
        let t = _mm_min_ps(z, _mm_setzero_ps());

        x = _mm_add_ps(x, _mm_xor_ps(t, _mm_and_ps(x, sign)));
        y = _mm_add_ps(y, _mm_xor_ps(t, _mm_and_ps(y, sign)));

        // compute normal length & scale
        let ll = _mm_add_ps(_mm_mul_ps(x, x), _mm_add_ps(_mm_mul_ps(y, y), _mm_mul_ps(z, z)));
        let s = _mm_div_ps(_mm_set1_ps(32767.0), _mm_sqrt_ps(ll));

        // rounded signed float->int
        let xr = _mm_cvtps_epi32(_mm_mul_ps(x, s));
        let yr = _mm_cvtps_epi32(_mm_mul_ps(y, s));
        let zr = _mm_cvtps_epi32(_mm_mul_ps(z, s));

        // mix x/z and y/0 to make 16-bit unpack easier
        let xzr = _mm_or_si128(_mm_and_si128(xr, _mm_set1_epi32(0xffff)), _mm_slli_epi32::<16>(zr));
        let y0r = _mm_and_si128(yr, _mm_set1_epi32(0xffff));

        // pack x/y/z using 16-bit unpacks; note that this has 0 where we should have .w
        let mut res_0 = _mm_unpacklo_epi16(xzr, y0r);
        let mut res_1 = _mm_unpackhi_epi16(xzr, y0r);

        // patch in .w
        let maskw = _mm_set_epi32(0xffff0000u32 as i32, 0, 0xffff0000u32 as i32, 0);
        res_0 = _mm_or_si128(res_0, _mm_and_si128(_mm_castps_si128(n4_0), maskw));
        res_1 = _mm_or_si128(res_1, _mm_and_si128(_mm_castps_si128(n4_1), maskw));

        store16(b0, res_0);
        store16(b1, res_1);
    }
}

/// `decodeFilterQuatSimd`
#[target_feature(enable = "sse2")]
fn decode_quat(data: &mut [u8], count: usize) {
    let scale = 32767.0 / 2.0f32.sqrt();

    for [b0, b1] in blocks8(data, count) {
        // the component indices (low 2 bits of each .w) select the output rotation
        let w = |b: &[u8; 16], at: usize| i32::from(i16::from_le_bytes([b[at], b[at + 1]]));
        let qc = [w(b0, 6), w(b0, 14), w(b1, 6), w(b1, 14)];

        let q4_0 = loadps(b0);
        let q4_1 = loadps(b1);

        // gather both x/y 16-bit pairs in each 32-bit lane
        let q4_xy = _mm_castps_si128(_mm_shuffle_ps::<0b10_00_10_00>(q4_0, q4_1));
        let q4_zc = _mm_castps_si128(_mm_shuffle_ps::<0b11_01_11_01>(q4_0, q4_1));

        // sign-extends each of x,y in [x y] with arithmetic shifts
        let xf = _mm_srai_epi32::<16>(_mm_slli_epi32::<16>(q4_xy));
        let yf = _mm_srai_epi32::<16>(q4_xy);
        let zf = _mm_srai_epi32::<16>(_mm_slli_epi32::<16>(q4_zc));
        let cf = _mm_srai_epi32::<16>(q4_zc);

        // get a floating-point scaler using zc with bottom 2 bits set to 1 (which represents 1.f)
        let sf = _mm_or_si128(cf, _mm_set1_epi32(3));
        let s = _mm_cvtepi32_ps(sf);

        // convert x/y/z to floating point (unscaled! implied scale of 1/sqrt(2.f) * 1/sf)
        let x = _mm_cvtepi32_ps(xf);
        let y = _mm_cvtepi32_ps(yf);
        let z = _mm_cvtepi32_ps(zf);

        // reconstruct w as a square root (unscaled); we clamp to 0.f to avoid NaN due to precision errors
        let ws = _mm_mul_ps(s, _mm_add_ps(s, s));
        let ww = _mm_sub_ps(ws, _mm_add_ps(_mm_mul_ps(x, x), _mm_add_ps(_mm_mul_ps(y, y), _mm_mul_ps(z, z))));
        let w = _mm_sqrt_ps(_mm_max_ps(ww, _mm_setzero_ps()));

        // compute final scale; note that all computations above are unscaled
        // we need to divide by sf to get out of fixed point, divide by sqrt(2) to renormalize and multiply by 32767 to get to int16 range
        let ss = _mm_div_ps(_mm_set1_ps(scale), s);

        // rounded signed float->int
        let xr = _mm_cvtps_epi32(_mm_mul_ps(x, ss));
        let yr = _mm_cvtps_epi32(_mm_mul_ps(y, ss));
        let zr = _mm_cvtps_epi32(_mm_mul_ps(z, ss));
        let wr = _mm_cvtps_epi32(_mm_mul_ps(w, ss));

        // mix x/z and w/y to make 16-bit unpack easier
        let xzr = _mm_or_si128(_mm_and_si128(xr, _mm_set1_epi32(0xffff)), _mm_slli_epi32::<16>(zr));
        let wyr = _mm_or_si128(_mm_and_si128(wr, _mm_set1_epi32(0xffff)), _mm_slli_epi32::<16>(yr));

        // pack x/y/z/w using 16-bit unpacks; we pack wxyz by default (for qc=0)
        let res_0 = _mm_unpacklo_epi16(wyr, xzr);
        let res_1 = _mm_unpackhi_epi16(wyr, xzr);

        let mut res = [[0u8; 16]; 2];
        store16(&mut res[0], res_0);
        store16(&mut res[1], res_1);

        // rotate and store
        let outs = b0.as_chunks_mut::<8>().0.iter_mut().chain(b1.as_chunks_mut::<8>().0);
        for ((out, v), qc) in outs.zip(res.as_flattened().as_chunks::<8>().0).zip(qc) {
            *out = u64::from_le_bytes(*v).rotate_left((qc << 4) as u32).to_le_bytes();
        }
    }
}

/// `decodeFilterExpSimd`
#[target_feature(enable = "sse2")]
fn decode_exp(data: &mut [u8], count: usize) {
    for block in data[..count * 4].as_chunks_mut::<16>().0 {
        let v = _mm_castps_si128(loadps(block));

        // decode exponent into 2^x directly
        let ef = _mm_srai_epi32::<24>(v);
        let es = _mm_slli_epi32::<23>(_mm_add_epi32(ef, _mm_set1_epi32(127)));

        // decode 24-bit mantissa into floating-point value
        let mf = _mm_srai_epi32::<8>(_mm_slli_epi32::<8>(v));
        let m = _mm_cvtepi32_ps(mf);

        let r = _mm_mul_ps(_mm_castsi128_ps(es), m);

        store16(block, _mm_castps_si128(r));
    }
}

/// `decodeFilterColorSimd8`
#[target_feature(enable = "sse2")]
fn decode_color8(data: &mut [u8], count: usize) {
    for block in data[..count * 4].as_chunks_mut::<16>().0 {
        let c4 = _mm_castps_si128(loadps(block));

        // unpack y/co/cg/a (co/cg are sign extended with arithmetic shifts)
        let yf = _mm_and_si128(c4, _mm_set1_epi32(0xff));
        let cof = _mm_srai_epi32::<24>(_mm_slli_epi32::<16>(c4));
        let cgf = _mm_srai_epi32::<24>(_mm_slli_epi32::<8>(c4));
        let mut af = _mm_srli_epi32::<24>(c4);

        // recover scale from alpha high bit
        let mut as_ = af;
        as_ = _mm_or_si128(as_, _mm_srli_epi32::<1>(as_));
        as_ = _mm_or_si128(as_, _mm_srli_epi32::<2>(as_));
        as_ = _mm_or_si128(as_, _mm_srli_epi32::<4>(as_));

        // expand alpha by one bit to match other components
        af = _mm_or_si128(_mm_and_si128(_mm_slli_epi32::<1>(af), as_), _mm_and_si128(af, _mm_set1_epi32(1)));

        // compute scaling factor
        let ss = _mm_mul_ps(_mm_set1_ps(255.0), _mm_rcp_ps(_mm_cvtepi32_ps(as_)));

        // convert to RGB in fixed point
        let rf = _mm_add_epi32(yf, _mm_sub_epi32(cof, cgf));
        let gf = _mm_add_epi32(yf, cgf);
        let bf = _mm_sub_epi32(yf, _mm_add_epi32(cof, cgf));

        // rounded signed float->int
        let rr = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(rf), ss));
        let gr = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(gf), ss));
        let br = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(bf), ss));
        let ar = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(af), ss));

        // repack rgba into final value
        let mut res = rr;
        res = _mm_or_si128(res, _mm_slli_epi32::<8>(gr));
        res = _mm_or_si128(res, _mm_slli_epi32::<16>(br));
        res = _mm_or_si128(res, _mm_slli_epi32::<24>(ar));

        store16(block, res);
    }
}

/// `decodeFilterColorSimd16`
#[target_feature(enable = "sse2")]
fn decode_color16(data: &mut [u8], count: usize) {
    for [b0, b1] in blocks8(data, count) {
        let c4_0 = loadps(b0);
        let c4_1 = loadps(b1);

        // gather both y/co 16-bit pairs in each 32-bit lane
        let c4_yco = _mm_castps_si128(_mm_shuffle_ps::<0b10_00_10_00>(c4_0, c4_1));
        let c4_cga = _mm_castps_si128(_mm_shuffle_ps::<0b11_01_11_01>(c4_0, c4_1));

        // unpack y/co/cg/a components (co/cg are sign extended with arithmetic shifts)
        let yf = _mm_and_si128(c4_yco, _mm_set1_epi32(0xffff));
        let cof = _mm_srai_epi32::<16>(c4_yco);
        let cgf = _mm_srai_epi32::<16>(_mm_slli_epi32::<16>(c4_cga));
        let mut af = _mm_srli_epi32::<16>(c4_cga);

        // recover scale from alpha high bit
        let mut as_ = af;
        as_ = _mm_or_si128(as_, _mm_srli_epi32::<1>(as_));
        as_ = _mm_or_si128(as_, _mm_srli_epi32::<2>(as_));
        as_ = _mm_or_si128(as_, _mm_srli_epi32::<4>(as_));
        as_ = _mm_or_si128(as_, _mm_srli_epi32::<8>(as_));

        // expand alpha by one bit to match other components
        af = _mm_or_si128(_mm_and_si128(_mm_slli_epi32::<1>(af), as_), _mm_and_si128(af, _mm_set1_epi32(1)));

        // compute scaling factor
        let ss = _mm_div_ps(_mm_set1_ps(65535.0), _mm_cvtepi32_ps(as_));

        // convert to RGB in fixed point
        let rf = _mm_add_epi32(yf, _mm_sub_epi32(cof, cgf));
        let gf = _mm_add_epi32(yf, cgf);
        let bf = _mm_sub_epi32(yf, _mm_add_epi32(cof, cgf));

        // rounded signed float->int
        let rr = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(rf), ss));
        let gr = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(gf), ss));
        let br = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(bf), ss));
        let ar = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(af), ss));

        // mix r/b and g/a to make 16-bit unpack easier
        let rbr = _mm_or_si128(_mm_and_si128(rr, _mm_set1_epi32(0xffff)), _mm_slli_epi32::<16>(br));
        let gar = _mm_or_si128(_mm_and_si128(gr, _mm_set1_epi32(0xffff)), _mm_slli_epi32::<16>(ar));

        // pack r/g/b/a using 16-bit unpacks
        store16(b0, _mm_unpacklo_epi16(rbr, gar));
        store16(b1, _mm_unpackhi_epi16(rbr, gar));
    }
}

/// Proof that the CPU supports SSE2 (always true on x86-64; C++ enables these paths unconditionally).
#[derive(Clone, Copy)]
pub(crate) struct FilterDecoder(());

#[allow(unsafe_code)]
impl FilterDecoder {
    pub(crate) fn detect() -> Option<Self> {
        is_x86_feature_detected!("sse2").then_some(FilterDecoder(()))
    }

    /// `stride` is 4 (8-bit) or 8 (16-bit components).
    pub(crate) fn decode_oct(self, data: &mut [u8], count: usize, stride: usize) {
        // SAFETY (all methods): `self` exists only if `detect` found sse2, the feature the callees are compiled for
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

/// `kDecodeTableMasks` / `kDecodeTableExtra`: for each pair of 4-bit triangle codes, the shuffle (bytes 6..16) and
/// per-byte `next` increments (bytes 0..6) that roll the 16-byte state forward by two triangles, and the extra bytes
/// consumed. State layout: 0..5 next extra bytes, 6..14 three triangles of index data, 15 the `next` index.
const TRIANGLE_TABLES: ([[u8; 16]; 256], [u8; 256]) = {
    let mut masks = [[0u8; 16]; 256];
    let mut extras = [0u8; 256];

    let mut code = 0;
    while code < 256 {
        let mut shuf = [0u8; 16];
        let mut next = [0u8; 16];
        let mut extra = 0u8;
        let mut nextoff = 0u8;

        // state 0..5 will be refilled every iteration, so we ignore that
        // state 6..8 will always contain the last decoded triangle because every triangle shifts fifo equally, so we can decode it independently
        shuf[6] = 12;
        shuf[7] = 13;
        shuf[8] = 14;

        // state 15 will contain next (potentially incremented a few times)
        shuf[15] = 15;

        // NEXT(var, ec): the vertex comes from extra (ec) or from next
        macro_rules! next_or_extra {
            ($var:expr, $ec:expr) => {
                let ec = $ec;
                shuf[$var] = if ec { extra } else { 15 };
                next[$var] = if ec { 0 } else { nextoff };
                extra += ec as u8;
                nextoff += 1 - ec as u8;
            };
        }

        // state 9..11 will contain the first decoded triangle (tri0), which can refer to extra/next and the original triangle history
        // state 12..14 will contain the second decoded triangle (tri1); when decoding edge reuse, we need to handle edge 0/1 specially as it was just decoded earlier
        let mut k = 0;
        while k < 2 {
            let tri = (code >> (k * 4)) & 0xf;

            if tri < 12 {
                if k == 1 && tri / 4 == 0 {
                    // we need to decode one of two edges from the triangle we just decoded earlier
                    // for that we simply need to copy shuf/next values for the two decoded indices
                    let a = 9 + if tri & 2 != 0 { 2 } else { 0 };
                    let b = 9 + if tri & 2 != 0 { 1 } else { 2 };
                    shuf[9 + k * 3] = shuf[a];
                    next[9 + k * 3] = next[a];
                    shuf[10 + k * 3] = shuf[b];
                    next[10 + k * 3] = next[b];
                } else {
                    // reuse: edge comes from the history based on edge index
                    // note: we reuse with an offset because last triangle in the original history was consumed by tri0
                    let trioff = (6 + k * 3 + (2 - tri / 4) * 3) as u8;

                    // edge cb or ac
                    shuf[9 + k * 3] = trioff + if tri & 2 != 0 { 2 } else { 0 };
                    shuf[10 + k * 3] = trioff + if tri & 2 != 0 { 1 } else { 2 };
                }

                // third vertex is either next or comes from extra
                next_or_extra!(11 + k * 3, tri & 1 != 0);
            } else {
                // restart: three vertices, each comes from next or extra
                next_or_extra!(9 + k * 3, tri > 12);
                next_or_extra!(10 + k * 3, tri > 13);
                next_or_extra!(11 + k * 3, tri > 14);
            }

            k += 1;
        }

        // next needs to advance
        next[15] = nextoff;

        // next[0..8] = 0 trivially (never written to); next[9] must also be 0 because nextoff is 0 initially
        // shuf[0..5] is not used, which allows us to pack next[10..15] + shuf[6..15] into a single 16-byte entry
        let mut i = 0;
        while i < 6 {
            masks[code][i] = next[10 + i];
            i += 1;
        }
        while i < 16 {
            masks[code][i] = shuf[i];
            i += 1;
        }
        extras[code] = extra;

        code += 1;
    }

    (masks, extras)
};

/// `kDecodeTableVerts` / `kDecodeTableLength`: for each control byte of 4 vertex references (2-bit lengths of 0..3
/// bytes, or 4 bytes each for 0xff), the shuffle that unpacks them into 32-bit lanes and the bytes consumed.
const VERTEX_TABLES: ([[u8; 16]; 256], [u8; 256]) = {
    let mut verts = [[0u8; 16]; 256];
    let mut lengths = [0u8; 256];

    let mut i = 0usize;
    while i < 256 {
        let mut offset = 0u8;
        let mut k = 0;
        while k < 4 {
            let code = ((i >> k) & 1) | ((i >> (k + 3)) & 2);
            let length = if i == 0xff { 4 } else { code as u8 };

            let mut j = 0;
            while j < 4 {
                verts[i][k * 4 + j] = if length > j as u8 { offset + j as u8 } else { 0x80 };
                j += 1;
            }

            offset += length;
            k += 1;
        }

        lengths[i] = offset;
        i += 1;
    }

    (verts, lengths)
};

/// `decodeTriangleGroup`
#[inline]
#[target_feature(enable = "sse4.1")]
fn decode_triangle_group(state: __m128i, code: u8, extra: &[u8; 8]) -> __m128i {
    let shuf = load16(&TRIANGLE_TABLES.0[usize::from(code)]);
    let next = _mm_slli_si128::<10>(shuf);

    // patch first 6 bytes with current extra and roll state forward
    let ext = _mm_cvtsi64_si128(i64::from_le_bytes(*extra));
    let state = _mm_blend_epi16::<7>(state, ext);
    _mm_add_epi8(_mm_shuffle_epi8(state, shuf), next)
}

/// `decodeVertexGroup`
#[inline]
#[target_feature(enable = "sse4.1")]
fn decode_vertex_group(last: __m128i, code: u8, data: &[u8; 16]) -> __m128i {
    let word = load16(data);
    let shuf = load16(&VERTEX_TABLES.0[usize::from(code)]);

    let v = _mm_shuffle_epi8(word, shuf);

    // unzigzag+1
    let xl = _mm_sub_epi32(_mm_setzero_si128(), _mm_and_si128(v, _mm_set1_epi32(1)));
    let xr = _mm_srli_epi32::<1>(v);
    let mut x = _mm_add_epi32(_mm_xor_si128(xl, xr), _mm_set1_epi32(1));

    // prefix sum
    x = _mm_add_epi32(x, _mm_slli_si128::<8>(x));
    x = _mm_add_epi32(x, _mm_slli_si128::<4>(x));
    _mm_add_epi32(x, _mm_shuffle_epi32::<0xff>(last))
}

/// `decodeTrianglesSimd`; tails are written exactly instead of rounding up to 4 bytes (the Rust API takes exactly
/// `triangle_count` triangles), which leaves the decoded values unchanged.
#[target_feature(enable = "sse4.1")]
fn decode_meshlet_triangles<T: MeshletTriangleType>(
    triangles: &mut [T],
    codes: &[u8],
    buffer: &[u8],
    mut extra: usize,
    bound: usize,
    triangle_count: usize,
) -> Option<usize> {
    let mut state = _mm_setzero_si128();

    for (i, &code) in codes[..triangle_count.div_ceil(2)].iter().enumerate() {
        // each triangle pair reads <=6 bytes from extra; the gap guarantees 16 bytes of overread past bound
        if extra > bound {
            return None;
        }

        state = decode_triangle_group(state, code, window::<8>(buffer, extra)?);
        extra += usize::from(TRIANGLE_TABLES.1[usize::from(code)]);

        // state bytes 9..11 and 12..14 hold the two decoded triangles; the last pair may hold a single one
        let tris = _mm_cvtsi128_si64(_mm_srli_si128::<9>(state)) as u64;
        T::write_packed(triangles, i * 2, (triangle_count - i * 2).min(2), tris);
    }

    Some(extra)
}

/// `decodeVerticesSimd`; tails are written exactly, see [`decode_meshlet_triangles`].
#[target_feature(enable = "sse4.1")]
fn decode_meshlet_vertices<V: IndexType>(
    vertices: &mut [V],
    ctrl: &[u8],
    buffer: &[u8],
    mut data: usize,
    bound: usize,
    vertex_count: usize,
) -> Option<usize> {
    let mut last = _mm_set1_epi32(-1);

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
        store16(&mut v, last);

        for (o, v) in out.iter_mut().zip(v.as_chunks::<4>().0) {
            *o = V::from_u32(u32::from_le_bytes(*v));
        }
    }

    Some(data)
}

/// Proof that the CPU supports SSE4.1 (C++: cpuid bit 19 checked in `decodeBuildTables`).
#[derive(Clone, Copy)]
pub(crate) struct MeshletDecoder(());

#[allow(unsafe_code)]
impl MeshletDecoder {
    pub(crate) fn detect() -> Option<Self> {
        is_x86_feature_detected!("sse4.1").then_some(MeshletDecoder(()))
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
        // SAFETY: `self` exists only if `detect` found sse4.1, the feature the callee is compiled for
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
