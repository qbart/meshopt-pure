//! x86-64 SIMD paths: vertex block decoding with SSSE3 + POPCNT (`decodeVertexBlockSimd`, `SIMD_SSE` with
//! `SIMD_LATENCYOPT`).

use std::arch::x86_64::*;

use crate::vertexcodec::{BYTE_GROUP_DECODE_LIMIT, BYTE_GROUP_SIZE, DecodeScratch, VERTEX_BLOCK_MAX_SIZE};

/// 16 bytes into a register; compiles to `movdqu`.
#[inline]
#[target_feature(enable = "ssse3,popcnt")]
fn load16(b: &[u8; 16]) -> __m128i {
    let (c, _) = b.as_chunks::<8>();
    _mm_unpacklo_epi64(
        _mm_cvtsi64_si128(i64::from_le_bytes(c[0])),
        _mm_cvtsi64_si128(i64::from_le_bytes(c[1])),
    )
}

/// Register into 16 bytes; compiles to `movdqu`.
#[inline]
#[target_feature(enable = "ssse3,popcnt")]
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
