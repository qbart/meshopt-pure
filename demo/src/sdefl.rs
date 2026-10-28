//! Port of `extern/sdefl.h` (Small Deflate, Micha Mettke, MIT / public domain): a small Deflate (RFC 1951)
//! compressor used by the demo to print post-deflate sizes. Output is byte-identical to the C version.
//!
//! Divergences from C (only where C would be UB or out of bounds):
//! - writes are bounds checked: if `out` is too small, [`sdeflate`] / [`zsdeflate`] return 0 instead of overflowing
//!   (note that [`sdefl_bound`] mirrors C and can be too small for >2MB of incompressible data);
//! - `lvl` outside `SDEFL_LVL_MIN..=SDEFL_LVL_MAX` or inputs longer than `i32::MAX` return 0;
//! - the sequence buffer grows past `SDEFL_SEQ_SIZ` instead of overflowing.

pub const SDEFL_MAX_OFF: usize = 1 << 15;
pub const SDEFL_WIN_SIZ: usize = SDEFL_MAX_OFF;
pub const SDEFL_WIN_MSK: usize = SDEFL_WIN_SIZ - 1;

pub const SDEFL_HASH_BITS: u32 = 15;
pub const SDEFL_HASH_SIZ: usize = 1 << SDEFL_HASH_BITS;
pub const SDEFL_HASH_MSK: usize = SDEFL_HASH_SIZ - 1;

pub const SDEFL_MIN_MATCH: usize = 4;
pub const SDEFL_BLK_MAX: usize = 256 * 1024;
pub const SDEFL_SEQ_SIZ: usize = SDEFL_BLK_MAX.div_ceil(3);

pub const SDEFL_SYM_MAX: usize = 288;
pub const SDEFL_OFF_MAX: usize = 32;
pub const SDEFL_PRE_MAX: usize = 19;

pub const SDEFL_LVL_MIN: i32 = 0;
pub const SDEFL_LVL_DEF: i32 = 5;
pub const SDEFL_LVL_MAX: i32 = 8;

#[derive(Clone, Copy, Debug)]
pub struct SdeflFreq {
    pub lit: [u32; SDEFL_SYM_MAX],
    pub off: [u32; SDEFL_OFF_MAX],
}
#[derive(Clone, Copy, Debug)]
pub struct SdeflCodeWords {
    pub lit: [u32; SDEFL_SYM_MAX],
    pub off: [u32; SDEFL_OFF_MAX],
}
#[derive(Clone, Copy, Debug)]
pub struct SdeflLens {
    pub lit: [u8; SDEFL_SYM_MAX],
    pub off: [u8; SDEFL_OFF_MAX],
}
#[derive(Clone, Copy, Debug, Default)]
pub struct SdeflCodes {
    pub word: SdeflCodeWords,
    pub len: SdeflLens,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct SdeflSeqt {
    pub off: i32,
    pub len: i32,
}

impl Default for SdeflFreq {
    fn default() -> Self {
        Self {
            lit: [0; SDEFL_SYM_MAX],
            off: [0; SDEFL_OFF_MAX],
        }
    }
}
impl Default for SdeflCodeWords {
    fn default() -> Self {
        Self {
            lit: [0; SDEFL_SYM_MAX],
            off: [0; SDEFL_OFF_MAX],
        }
    }
}
impl Default for SdeflLens {
    fn default() -> Self {
        Self {
            lit: [0; SDEFL_SYM_MAX],
            off: [0; SDEFL_OFF_MAX],
        }
    }
}

/// Compressor state (C `struct sdefl`); the large tables are heap allocated.
#[derive(Clone, Debug)]
pub struct Sdefl {
    bits: i32,
    bitcnt: i32,
    tbl: Vec<i32>, // SDEFL_HASH_SIZ
    prv: Vec<i32>, // SDEFL_WIN_SIZ

    seq_cnt: usize,
    seq: Vec<SdeflSeqt>, // SDEFL_SEQ_SIZ, grows if needed
    freq: SdeflFreq,
    cod: SdeflCodes,
}

impl Sdefl {
    pub fn new() -> Self {
        Self {
            bits: 0,
            bitcnt: 0,
            tbl: vec![0; SDEFL_HASH_SIZ],
            prv: vec![0; SDEFL_WIN_SIZ],
            seq_cnt: 0,
            seq: vec![SdeflSeqt::default(); SDEFL_SEQ_SIZ],
            freq: SdeflFreq::default(),
            cod: SdeflCodes::default(),
        }
    }
}

impl Default for Sdefl {
    fn default() -> Self {
        Self::new()
    }
}

const SDEFL_NIL: i64 = -1;
const SDEFL_MAX_MATCH: usize = 258;
const SDEFL_MAX_CODE_LEN: usize = 15;
const SDEFL_SYM_BITS: u32 = 10;
const SDEFL_SYM_MSK: u32 = (1 << SDEFL_SYM_BITS) - 1;
const SDEFL_RAW_BLK_SIZE: usize = 65535;
const SDEFL_LIT_LEN_CODES: u32 = 14;
const SDEFL_OFF_CODES: u32 = 15;
const SDEFL_PRE_CODES: u32 = 7;
const SDEFL_EOB: usize = 256;

/// C `SDEFL_CNT_NUM(n)`: `(((n)+3u/4u)+3u)&~3u`, where `3u/4u` is 0.
const fn sdefl_cnt_num(n: usize) -> usize {
    (n + 3) & !3
}

/// Bounds checked output cursor (C `unsigned char **dst`).
struct SdeflOut<'a> {
    buf: &'a mut [u8],
    pos: usize,
    overflow: bool,
}

impl SdeflOut<'_> {
    fn byte(&mut self, b: u8) {
        match self.buf.get_mut(self.pos) {
            Some(d) => {
                *d = b;
                self.pos += 1;
            }
            None => self.overflow = true,
        }
    }
    fn copy(&mut self, src: &[u8]) {
        match self.buf.get_mut(self.pos..self.pos + src.len()) {
            Some(d) => {
                d.copy_from_slice(src);
                self.pos += src.len();
            }
            None => self.overflow = true,
        }
    }
}

fn sdefl_ilog2(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    31 - n.leading_zeros()
}
fn sdefl_npow2(n: u32) -> u32 {
    1 << (sdefl_ilog2(n - 1) + 1)
}
fn sdefl_uload32(p: &[u8]) -> u32 {
    u32::from_le_bytes([p[0], p[1], p[2], p[3]])
}
fn sdefl_hash32(p: &[u8]) -> u32 {
    let n = sdefl_uload32(p);
    n.wrapping_mul(0x9E377989) >> (32 - SDEFL_HASH_BITS)
}
fn sdefl_put(dst: &mut SdeflOut, s: &mut Sdefl, code: i32, bitcnt: i32) {
    s.bits |= code << s.bitcnt;
    s.bitcnt += bitcnt;
    while s.bitcnt >= 8 {
        dst.byte((s.bits & 0xFF) as u8);
        s.bits >>= 8;
        s.bitcnt -= 8;
    }
}
/// `a` is 1-based: `a[k - 1]` is C `A[k]`.
fn sdefl_heap_sub(a: &mut [u32], len: usize, sub: usize) {
    let mut p = sub;
    let v = a[sub - 1];
    loop {
        let mut c = p << 1;
        if c > len {
            break;
        }
        if c < len && a[c] > a[c - 1] {
            c += 1;
        }
        if v >= a[c - 1] {
            break;
        }
        a[p - 1] = a[c - 1];
        p = c;
    }
    a[p - 1] = v;
}
fn sdefl_heap_array(a: &mut [u32], len: usize) {
    let mut sub = len >> 1;
    while sub >= 1 {
        sdefl_heap_sub(a, len, sub);
        sub -= 1;
    }
}
fn sdefl_heap_sort(a: &mut [u32]) {
    let mut n = a.len();
    sdefl_heap_array(a, n);
    while n >= 2 {
        a.swap(n - 1, 0);
        n -= 1;
        sdefl_heap_sub(a, n, 1);
    }
}
fn sdefl_sort_sym(sym_cnt: usize, freqs: &[u32], lens: &mut [u8], sym_out: &mut [u32]) -> usize {
    let mut cnts = [0u32; sdefl_cnt_num(SDEFL_SYM_MAX)];
    let cnt_num = sdefl_cnt_num(sym_cnt);
    let mut used_sym = 0u32;
    for &freq in &freqs[..sym_cnt] {
        cnts[(freq as usize).min(cnt_num - 1)] += 1;
    }
    for cnt in &mut cnts[1..cnt_num] {
        let c = *cnt;
        *cnt = used_sym;
        used_sym += c;
    }
    for sym in 0..sym_cnt {
        let freq = freqs[sym];
        if freq != 0 {
            let idx = (freq as usize).min(cnt_num - 1);
            sym_out[cnts[idx] as usize] = sym as u32 | (freq << SDEFL_SYM_BITS);
            cnts[idx] += 1;
        } else {
            lens[sym] = 0;
        }
    }
    sdefl_heap_sort(&mut sym_out[cnts[cnt_num - 2] as usize..cnts[cnt_num - 1] as usize]);
    used_sym as usize
}
fn sdefl_build_tree(a: &mut [u32], sym_cnt: usize) {
    let (mut i, mut b, mut e) = (0usize, 0usize, 0usize);
    loop {
        let m;
        let n;
        if i != sym_cnt && (b == e || (a[i] >> SDEFL_SYM_BITS) <= (a[b] >> SDEFL_SYM_BITS)) {
            m = i;
            i += 1;
        } else {
            m = b;
            b += 1;
        }
        if i != sym_cnt && (b == e || (a[i] >> SDEFL_SYM_BITS) <= (a[b] >> SDEFL_SYM_BITS)) {
            n = i;
            i += 1;
        } else {
            n = b;
            b += 1;
        }

        let freq_shift = (a[m] & !SDEFL_SYM_MSK).wrapping_add(a[n] & !SDEFL_SYM_MSK);
        a[m] = (a[m] & SDEFL_SYM_MSK) | ((e as u32) << SDEFL_SYM_BITS);
        a[n] = (a[n] & SDEFL_SYM_MSK) | ((e as u32) << SDEFL_SYM_BITS);
        a[e] = (a[e] & SDEFL_SYM_MSK) | freq_shift;
        e += 1;
        if sym_cnt - e <= 1 {
            break;
        }
    }
}
fn sdefl_gen_len_cnt(a: &mut [u32], root: usize, len_cnt: &mut [u32], max_code_len: usize) {
    for c in &mut len_cnt[..=max_code_len] {
        *c = 0;
    }
    len_cnt[1] = 2;

    a[root] &= SDEFL_SYM_MSK;
    for n in (0..root).rev() {
        let p = (a[n] >> SDEFL_SYM_BITS) as usize;
        let pdepth = a[p] >> SDEFL_SYM_BITS;
        let depth = pdepth + 1;
        let mut len = depth as usize;

        a[n] = (a[n] & SDEFL_SYM_MSK) | (depth << SDEFL_SYM_BITS);
        if len >= max_code_len {
            len = max_code_len;
            loop {
                len -= 1;
                if len == 0 || len_cnt[len] != 0 {
                    break;
                }
            }
        }
        len_cnt[len] = len_cnt[len].wrapping_sub(1);
        len_cnt[len + 1] += 2;
    }
}
fn sdefl_gen_codes(a: &mut [u32], lens: &mut [u8], len_cnt: &[u32], max_code_word_len: usize, sym_cnt: usize) {
    let mut nxt = [0u32; SDEFL_MAX_CODE_LEN + 1];
    let mut i = 0usize;
    for len in (1..=max_code_word_len).rev() {
        for _ in 0..len_cnt[len] {
            lens[(a[i] & SDEFL_SYM_MSK) as usize] = len as u8;
            i += 1;
        }
    }
    nxt[0] = 0;
    nxt[1] = 0;
    for len in 2..=max_code_word_len {
        nxt[len] = (nxt[len - 1] + len_cnt[len - 1]) << 1;
    }
    for sym in 0..sym_cnt {
        let l = lens[sym] as usize;
        a[sym] = nxt[l];
        nxt[l] = nxt[l].wrapping_add(1);
    }
}
fn sdefl_rev(mut c: u32, n: u8) -> u32 {
    c = ((c & 0x5555) << 1) | ((c & 0xAAAA) >> 1);
    c = ((c & 0x3333) << 2) | ((c & 0xCCCC) >> 2);
    c = ((c & 0x0F0F) << 4) | ((c & 0xF0F0) >> 4);
    c = ((c & 0x00FF) << 8) | ((c & 0xFF00) >> 8);
    c >> (16 - n as u32)
}
fn sdefl_huff(lens: &mut [u8], codes: &mut [u32], freqs: &[u32], num_syms: usize, max_code_len: u32) {
    let a = codes;
    let mut len_cnt = [0u32; SDEFL_MAX_CODE_LEN + 1];
    let used_syms = sdefl_sort_sym(num_syms, freqs, lens, a);
    if used_syms == 0 {
        return;
    }
    if used_syms == 1 {
        let s = a[0] & SDEFL_SYM_MSK;
        let i = if s != 0 { s as usize } else { 1 };
        a[0] = 0;
        lens[0] = 1;
        a[i] = 1;
        lens[i] = 1;
        return;
    }
    sdefl_build_tree(a, used_syms);
    sdefl_gen_len_cnt(a, used_syms - 2, &mut len_cnt, max_code_len as usize);
    sdefl_gen_codes(a, lens, &len_cnt, max_code_len as usize, num_syms);
    for c in 0..num_syms {
        a[c] = sdefl_rev(a[c], lens[c]);
    }
}
#[derive(Default)]
struct SdeflSymcnt {
    items: usize,
    lit: usize,
    off: usize,
}
fn sdefl_precode(cnt: &mut SdeflSymcnt, freqs: &mut [u32], items: &mut [u32], litlen: &[u8], offlen: &[u8]) {
    let mut at = 0usize;
    let mut run_start = 0usize;

    let mut lens = [0u8; SDEFL_SYM_MAX + SDEFL_OFF_MAX];
    cnt.lit = SDEFL_SYM_MAX;
    while cnt.lit > 257 {
        if litlen[cnt.lit - 1] != 0 {
            break;
        }
        cnt.lit -= 1;
    }
    cnt.off = SDEFL_OFF_MAX;
    while cnt.off > 1 {
        if offlen[cnt.off - 1] != 0 {
            break;
        }
        cnt.off -= 1;
    }

    let total = cnt.lit + cnt.off;
    lens[..cnt.lit].copy_from_slice(&litlen[..cnt.lit]);
    lens[cnt.lit..total].copy_from_slice(&offlen[..cnt.off]);
    loop {
        let len = lens[run_start] as usize;
        let mut run_end = run_start;
        loop {
            run_end += 1;
            if run_end == total || len != lens[run_end] as usize {
                break;
            }
        }
        if len == 0 {
            while (run_end - run_start) >= 11 {
                let n = (run_end - run_start) - 11;
                let xbits = n.min(0x7f);
                freqs[18] += 1;
                items[at] = 18 | ((xbits as u32) << 5);
                at += 1;
                run_start += 11 + xbits;
            }
            if (run_end - run_start) >= 3 {
                let n = (run_end - run_start) - 3;
                let xbits = n.min(0x7);
                freqs[17] += 1;
                items[at] = 17 | ((xbits as u32) << 5);
                at += 1;
                run_start += 3 + xbits;
            }
        } else if (run_end - run_start) >= 4 {
            freqs[len] += 1;
            items[at] = len as u32;
            at += 1;
            run_start += 1;
            loop {
                let xbits = ((run_end - run_start) - 3).min(0x03);
                items[at] = 16 | ((xbits as u32) << 5);
                at += 1;
                run_start += 3 + xbits;
                freqs[16] += 1;
                if (run_end - run_start) < 3 {
                    break;
                }
            }
        }
        while run_start != run_end {
            freqs[len] += 1;
            items[at] = len as u32;
            at += 1;
            run_start += 1;
        }
        if run_start == total {
            break;
        }
    }
    cnt.items = at;
}
struct SdeflMatchCodest {
    ls: usize,
    lc: usize,
    dc: usize,
    dx: u32,
}
fn sdefl_match_codes(dist: usize, len: usize) -> SdeflMatchCodest {
    const DXMAX: [usize; 14] = [0, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288, 24576];
    #[rustfmt::skip]
    const LSLOT: [u8; 258 + 1] = [
        0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 12,
        12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 16,
        16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18,
        18, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
        21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
        22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
        23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
        24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25,
        25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
        25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26,
        26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
        26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
        27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
        27, 27, 28,
    ];
    let ls = LSLOT[len] as usize;
    let dx = sdefl_ilog2(sdefl_npow2(dist as u32) >> 2);
    let dc = if dx != 0 {
        (((dx as usize) + 1) << 1) + (dist > DXMAX[dx as usize]) as usize
    } else {
        dist - 1
    };
    SdeflMatchCodest { ls, lc: 257 + ls, dc, dx }
}
enum SdeflBlkType {
    Ucompr,
    Dyn,
}
fn sdefl_blk_type(s: &Sdefl, blk_len: usize, pre_item_len: usize, pre_freq: &[u32], pre_len: &[u8]) -> SdeflBlkType {
    const X_PRE_BITS: [u8; 19] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7];
    const X_LEN_BITS: [u8; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    const X_OFF_BITS: [u8; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
    ];

    let mut dyn_cost: i64 = 0;
    let mut fix_cost: i64 = 0;

    dyn_cost += 5 + 5 + 4 + (3 * pre_item_len as i64);
    for sym in 0..SDEFL_PRE_MAX {
        dyn_cost += pre_freq[sym] as i64 * (X_PRE_BITS[sym] as i64 + pre_len[sym] as i64);
    }
    for sym in 0..256 {
        dyn_cost += s.freq.lit[sym] as i64 * s.cod.len.lit[sym] as i64;
    }
    dyn_cost += s.cod.len.lit[SDEFL_EOB] as i64;
    for sym in 257..286 {
        dyn_cost += s.freq.lit[sym] as i64 * (X_LEN_BITS[sym - 257] as i64 + s.cod.len.lit[sym] as i64);
    }
    for (sym, &x) in X_OFF_BITS.iter().enumerate() {
        dyn_cost += s.freq.off[sym] as i64 * (x as i64 + s.cod.len.off[sym] as i64);
    }

    fix_cost += 8 * (5 * blk_len.div_ceil(SDEFL_RAW_BLK_SIZE) as i64 + blk_len as i64 + 1 + 2);
    if dyn_cost < fix_cost {
        SdeflBlkType::Dyn
    } else {
        SdeflBlkType::Ucompr
    }
}
fn sdefl_put16(dst: &mut SdeflOut, x: u16) {
    dst.byte((x & 0xff) as u8);
    dst.byte((x >> 8) as u8);
}
fn sdefl_match(dst: &mut SdeflOut, s: &mut Sdefl, dist: usize, len: usize) {
    const LXN: [u8; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    const LMIN: [u16; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
    ];
    const DMIN: [u16; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289,
        16385, 24577,
    ];

    let cod = sdefl_match_codes(dist, len);
    let (word, wlen) = (s.cod.word.lit[cod.lc], s.cod.len.lit[cod.lc]);
    sdefl_put(dst, s, word as i32, wlen as i32);
    sdefl_put(dst, s, (len - LMIN[cod.ls] as usize) as i32, LXN[cod.ls] as i32);
    let (word, wlen) = (s.cod.word.off[cod.dc], s.cod.len.off[cod.dc]);
    sdefl_put(dst, s, word as i32, wlen as i32);
    sdefl_put(dst, s, (dist - DMIN[cod.dc] as usize) as i32, cod.dx as i32);
}
fn sdefl_flush(dst: &mut SdeflOut, s: &mut Sdefl, is_last: bool, input: &[u8], blk_begin: usize, blk_end: usize) {
    let mut blk_len = blk_end - blk_begin;
    let mut symcnt = SdeflSymcnt::default();
    let mut codes = [0u32; SDEFL_PRE_MAX];
    let mut lens = [0u8; SDEFL_PRE_MAX];
    let mut freqs = [0u32; SDEFL_PRE_MAX];
    let mut items = [0u32; SDEFL_SYM_MAX + SDEFL_OFF_MAX];
    const PERM: [usize; SDEFL_PRE_MAX] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

    /* calculate huffman codes */
    s.freq.lit[SDEFL_EOB] += 1;
    sdefl_huff(
        &mut s.cod.len.lit,
        &mut s.cod.word.lit,
        &s.freq.lit,
        SDEFL_SYM_MAX,
        SDEFL_LIT_LEN_CODES,
    );
    sdefl_huff(&mut s.cod.len.off, &mut s.cod.word.off, &s.freq.off, SDEFL_OFF_MAX, SDEFL_OFF_CODES);
    sdefl_precode(&mut symcnt, &mut freqs, &mut items, &s.cod.len.lit, &s.cod.len.off);
    sdefl_huff(&mut lens, &mut codes, &freqs, SDEFL_PRE_MAX, SDEFL_PRE_CODES);
    let mut item_cnt = SDEFL_PRE_MAX;
    while item_cnt > 4 {
        if lens[PERM[item_cnt - 1]] != 0 {
            break;
        }
        item_cnt -= 1;
    }
    /* write block */
    match sdefl_blk_type(s, blk_len, item_cnt, &freqs, &lens) {
        SdeflBlkType::Ucompr => {
            /* uncompressed blocks */
            let n = blk_len.div_ceil(SDEFL_RAW_BLK_SIZE);
            for i in 0..n {
                let fin = is_last && (i + 1 == n);
                let amount = blk_len.min(SDEFL_RAW_BLK_SIZE);
                sdefl_put(dst, s, fin as i32, 1); /* block */
                sdefl_put(dst, s, 0x00, 2); /* stored block */
                if s.bitcnt != 0 {
                    let pad = 8 - s.bitcnt;
                    sdefl_put(dst, s, 0x00, pad);
                }
                sdefl_put16(dst, amount as u16);
                sdefl_put16(dst, !(amount as u16));
                let begin = blk_begin + i * SDEFL_RAW_BLK_SIZE;
                dst.copy(&input[begin..begin + amount]);
                blk_len -= amount;
            }
        }
        SdeflBlkType::Dyn => {
            /* dynamic huffman block */
            sdefl_put(dst, s, is_last as i32, 1); /* block */
            sdefl_put(dst, s, 0x02, 2); /* dynamic huffman */
            sdefl_put(dst, s, symcnt.lit as i32 - 257, 5);
            sdefl_put(dst, s, symcnt.off as i32 - 1, 5);
            sdefl_put(dst, s, item_cnt as i32 - 4, 4);
            for &p in &PERM[..item_cnt] {
                sdefl_put(dst, s, lens[p] as i32, 3);
            }
            for &item in &items[..symcnt.items] {
                let sym = (item & 0x1F) as usize;
                sdefl_put(dst, s, codes[sym] as i32, lens[sym] as i32);
                if sym < 16 {
                    continue;
                }
                let xbits = (item >> 5) as i32;
                if sym == 16 {
                    sdefl_put(dst, s, xbits, 2);
                } else if sym == 17 {
                    sdefl_put(dst, s, xbits, 3);
                } else {
                    sdefl_put(dst, s, xbits, 7);
                }
            }
            /* block sequences */
            for i in 0..s.seq_cnt {
                let seq = s.seq[i];
                if seq.off >= 0 {
                    for &c in &input[seq.off as usize..(seq.off + seq.len) as usize] {
                        let (word, wlen) = (s.cod.word.lit[c as usize], s.cod.len.lit[c as usize]);
                        sdefl_put(dst, s, word as i32, wlen as i32);
                    }
                } else {
                    sdefl_match(dst, s, (-seq.off) as usize, seq.len as usize);
                }
            }
            let (word, wlen) = (s.cod.word.lit[SDEFL_EOB], s.cod.len.lit[SDEFL_EOB]);
            sdefl_put(dst, s, word as i32, wlen as i32);
        }
    }
    s.freq = SdeflFreq::default();
    s.seq_cnt = 0;
}
fn sdefl_seq(s: &mut Sdefl, off: i32, len: i32) {
    let seq = SdeflSeqt { off, len };
    match s.seq.get_mut(s.seq_cnt) {
        Some(slot) => *slot = seq,
        None => s.seq.push(seq),
    }
    s.seq_cnt += 1;
}
fn sdefl_reg_match(s: &mut Sdefl, off: usize, len: usize) {
    let cod = sdefl_match_codes(off, len);
    s.freq.lit[cod.lc] += 1;
    s.freq.off[cod.dc] += 1;
}
#[derive(Default)]
struct SdeflMatch {
    off: usize,
    len: usize,
}
fn sdefl_fnd(m: &mut SdeflMatch, s: &Sdefl, mut chain_len: i32, max_match: usize, input: &[u8], p: usize) {
    let mut i = s.tbl[sdefl_hash32(&input[p..]) as usize] as i64;
    let limit = (p as i64 - SDEFL_WIN_SIZ as i64).max(SDEFL_NIL);

    while i > limit {
        let iu = i as usize;
        if input[iu + m.len] == input[p + m.len] && sdefl_uload32(&input[iu..]) == sdefl_uload32(&input[p..]) {
            let mut n = SDEFL_MIN_MATCH;
            while n < max_match && input[iu + n] == input[p + n] {
                n += 1;
            }
            if n > m.len {
                m.len = n;
                m.off = p - iu;
                if n == max_match {
                    break;
                }
            }
        }
        chain_len -= 1;
        if chain_len == 0 {
            break;
        }
        i = s.prv[iu & SDEFL_WIN_MSK] as i64;
    }
}
fn sdefl_compr(s: &mut Sdefl, q: &mut SdeflOut, input: &[u8], lvl: i32) -> usize {
    let start = q.pos;
    const PREF: [u8; 9] = [8, 10, 14, 24, 30, 48, 65, 96, 130];
    let max_chain = if lvl < 8 { 1 << (lvl + 1) } else { 1 << 13 };
    let in_len = input.len();
    let mut i = 0usize;
    let mut litlen = 0usize;
    s.tbl.fill(SDEFL_NIL as i32);
    loop {
        let blk_begin = i;
        let blk_end = (i + SDEFL_BLK_MAX).min(in_len);
        while i < blk_end {
            let mut m = SdeflMatch::default();
            let left = blk_end - i;
            let max_match = left.min(SDEFL_MAX_MATCH);
            let nice_match = (PREF[lvl as usize] as usize).min(max_match);
            let mut run = 1usize;
            let mut inc = 1usize;
            if max_match > SDEFL_MIN_MATCH {
                sdefl_fnd(&mut m, s, max_chain, max_match, input, i);
            }
            if lvl >= 5 && m.len >= SDEFL_MIN_MATCH && m.len + 1 < nice_match {
                let mut m2 = SdeflMatch::default();
                sdefl_fnd(&mut m2, s, max_chain, m.len + 1, input, i + 1);
                m.len = if m2.len > m.len { 0 } else { m.len };
            }
            if m.len >= SDEFL_MIN_MATCH {
                if litlen != 0 {
                    sdefl_seq(s, (i - litlen) as i32, litlen as i32);
                    litlen = 0;
                }
                sdefl_seq(s, -(m.off as i32), m.len as i32);
                sdefl_reg_match(s, m.off, m.len);
                if lvl < 2 && m.len >= nice_match {
                    inc = m.len;
                } else {
                    run = m.len;
                }
            } else {
                s.freq.lit[input[i] as usize] += 1;
                litlen += 1;
            }
            let run_inc = run * inc;
            if in_len - (i + run_inc) > SDEFL_MIN_MATCH {
                while run > 0 {
                    run -= 1;
                    let h = sdefl_hash32(&input[i..]) as usize;
                    s.prv[i & SDEFL_WIN_MSK] = s.tbl[h];
                    s.tbl[h] = i as i32;
                    i += inc;
                }
            } else {
                i += run_inc;
            }
        }
        if litlen != 0 {
            sdefl_seq(s, (i - litlen) as i32, litlen as i32);
            litlen = 0;
        }
        sdefl_flush(q, s, blk_end == in_len, input, blk_begin, blk_end);
        if i >= in_len {
            break;
        }
    }
    if s.bitcnt != 0 {
        let pad = 8 - s.bitcnt;
        sdefl_put(q, s, 0x00, pad);
    }
    q.pos - start
}
fn sdefl_valid(input: &[u8], lvl: i32) -> bool {
    (SDEFL_LVL_MIN..=SDEFL_LVL_MAX).contains(&lvl) && input.len() <= i32::MAX as usize
}
/// Compresses `input` into `out` as a raw Deflate stream and returns the number of bytes written.
/// Returns 0 if `out` is too small, `lvl` is out of range or `input` is longer than `i32::MAX`
/// (an empty `input` also yields 0 bytes, as in C).
pub fn sdeflate(s: &mut Sdefl, out: &mut [u8], input: &[u8], lvl: i32) -> usize {
    if !sdefl_valid(input, lvl) {
        return 0;
    }
    s.bits = 0;
    s.bitcnt = 0;
    let mut q = SdeflOut {
        buf: out,
        pos: 0,
        overflow: false,
    };
    let n = sdefl_compr(s, &mut q, input, lvl);
    if q.overflow { 0 } else { n }
}
const SDEFL_ADLER_INIT: u32 = 1;
fn sdefl_adler32(adler32: u32, input: &[u8]) -> u32 {
    const ADLER_MOD: u32 = 65521;
    let mut s1 = adler32 & 0xffff;
    let mut s2 = adler32 >> 16;

    let mut blk_len = input.len() % 5552;
    let mut rest = input;
    while !rest.is_empty() {
        let (blk, tail) = rest.split_at(blk_len);
        for &b in blk {
            s1 += b as u32;
            s2 += s1;
        }
        s1 %= ADLER_MOD;
        s2 %= ADLER_MOD;
        rest = tail;
        blk_len = 5552;
    }
    (s2 << 16) + s1
}
/// Like [`sdeflate`], but wraps the stream in a zlib header and adler32 trailer.
/// Returns 0 if `out` is too small, `lvl` is out of range or `input` is longer than `i32::MAX`.
pub fn zsdeflate(s: &mut Sdefl, out: &mut [u8], input: &[u8], lvl: i32) -> usize {
    if !sdefl_valid(input, lvl) {
        return 0;
    }
    let mut q = SdeflOut {
        buf: out,
        pos: 0,
        overflow: false,
    };

    s.bits = 0;
    s.bitcnt = 0;
    sdefl_put(&mut q, s, 0x78, 8); /* deflate, 32k window */
    sdefl_put(&mut q, s, 0x01, 8); /* fast compression */
    sdefl_compr(s, &mut q, input, lvl);

    /* append adler checksum */
    let mut a = sdefl_adler32(SDEFL_ADLER_INIT, input);
    for _ in 0..4 {
        sdefl_put(&mut q, s, ((a >> 24) & 0xFF) as i32, 8);
        a <<= 8;
    }
    if q.overflow { 0 } else { q.pos }
}
/// Output buffer size for `len` input bytes (as in C; may be too small for >2MB of incompressible data).
pub fn sdefl_bound(len: usize) -> usize {
    let max_blocks = 1 + len.div_ceil(SDEFL_RAW_BLK_SIZE);
    (5 * max_blocks).saturating_add(len).saturating_add(1 + 4 + 8 + 3)
}
