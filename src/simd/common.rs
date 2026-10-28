//! ISA-independent parts of the SIMD decoders: lookup tables (built at compile time instead of at startup like
//! C++) and bounds-checked windows into byte slices.

/// `slice[at..at + N]` as an array; `None` when out of bounds.
#[inline(always)]
pub(crate) fn window<const N: usize>(slice: &[u8], at: usize) -> Option<&[u8; N]> {
    slice.get(at..)?.first_chunk::<N>()
}

#[inline(always)]
pub(crate) fn window_mut<const N: usize>(slice: &mut [u8], at: usize) -> Option<&mut [u8; N]> {
    slice.get_mut(at..)?.first_chunk_mut::<N>()
}

/// `kDecodeBytesGroupShuffle`: for each 8-bit sentinel mask, the shuffle that expands consecutive bytes into the
/// masked lanes (0x80 zeroes the others).
pub(crate) const GROUP_SHUFFLE: [[u8; 8]; 256] = {
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

/// `kDecodeBytesGroupCount`: number of set bits of each mask (bytes consumed by a shuffle from `GROUP_SHUFFLE`).
#[cfg(target_arch = "aarch64")]
pub(crate) const GROUP_COUNT: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut mask = 0;
    while mask < 256 {
        table[mask] = (mask as u32).count_ones() as u8;
        mask += 1;
    }
    table
};

/// `dispatchSimd`: runs `process` on whole groups of 4 elements, then on a zero-padded copy of the tail.
#[inline]
pub(crate) fn dispatch<const S: usize>(data: &mut [u8], count: usize, process: impl Fn(&mut [u8], usize)) {
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

/// `count` elements of 8 bytes as blocks of 4 elements, each split into two 16-byte halves.
#[inline]
pub(crate) fn blocks8(data: &mut [u8], count: usize) -> &mut [[[u8; 16]; 2]] {
    data[..count * 8].as_chunks_mut::<16>().0.as_chunks_mut::<2>().0
}

/// `kDecodeTableMasks` / `kDecodeTableExtra`: for each pair of 4-bit triangle codes, the shuffle (bytes 6..16) and
/// per-byte `next` increments (bytes 0..6) that roll the 16-byte state forward by two triangles, and the extra bytes
/// consumed. State layout: 0..5 next extra bytes, 6..14 three triangles of index data, 15 the `next` index.
pub(crate) const TRIANGLE_TABLES: ([[u8; 16]; 256], [u8; 256]) = {
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
pub(crate) const VERTEX_TABLES: ([[u8; 16]; 256], [u8; 256]) = {
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
