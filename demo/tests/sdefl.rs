//! sdefl tests: golden (size, FNV-1a 64) values come from the C `extern/sdefl.h` built out of tree with the same
//! deterministic input generators.

use meshoptimizer_demo::sdefl::*;

fn fnv1a(data: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

fn gen_input(name: &str) -> Vec<u8> {
    match name {
        "one" => vec![0x42],
        "rand0" | "rand1" | "rand2" => {
            let k = (name.as_bytes()[4] - b'0') as usize;
            let mut rng = Lcg(k as u64 + 1);
            (0..[1000, 70000, 600000][k]).map(|_| rng.next() as u8).collect()
        }
        "zeros" => vec![0; 300000],
        "abcd" => {
            let mut rng = Lcg(4);
            (0..20000).map(|_| b"abcd"[(rng.next() % 4) as usize]).collect()
        }
        "text" => {
            const WORDS: [&str; 16] = [
                "the",
                "quick",
                "brown",
                "fox",
                "jumps",
                "over",
                "lazy",
                "dog",
                "mesh",
                "optimizer",
                "vertex",
                "index",
                "cache",
                "fetch",
                "overdraw",
                "simplify",
            ];
            let n = 100000;
            let mut rng = Lcg(5);
            let mut b = Vec::with_capacity(n);
            while b.len() < n {
                for &c in WORDS[(rng.next() % 16) as usize].as_bytes() {
                    if b.len() < n {
                        b.push(c);
                    }
                }
                if b.len() < n {
                    b.push(if rng.next().is_multiple_of(8) { b'\n' } else { b' ' });
                }
            }
            b
        }
        "pattern" => {
            let mut rng = Lcg(6);
            (0..200000)
                .map(|i| ((i % 251) ^ if (i % 997) == 0 { rng.next() & 0xff } else { 0 }) as u8)
                .collect()
        }
        "mesh" => {
            let g = 64u32;
            let mut b = Vec::new();
            for y in 0..g {
                for x in 0..g {
                    for v in [x as f32 * 0.1, y as f32 * 0.1, (x * y) as f32 * 0.01] {
                        b.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
            for y in 0..g - 1 {
                for x in 0..g - 1 {
                    let i0 = y * g + x;
                    for i in [i0, i0 + 1, i0 + g, i0 + 1, i0 + g + 1, i0 + g] {
                        b.extend_from_slice(&i.to_le_bytes());
                    }
                }
            }
            b
        }
        _ => Vec::new(),
    }
}

const INPUT_HASHES: &[(&str, u64)] = &[
    ("empty", 0xcbf29ce484222325),
    ("one", 0xaf63ff4c86022805),
    ("rand0", 0xd80e4d945f00e3d7),
    ("rand1", 0xaeda3cf996cea6aa),
    ("rand2", 0xf2b3388e993467f5),
    ("zeros", 0x4fa0ef91d99420a5),
    ("abcd", 0xa7aa04b6606d24ca),
    ("text", 0x85bb0b937bbf4ad8),
    ("pattern", 0x68a266e0ef937928),
    ("mesh", 0xa59ede37884cc191),
];

/// (input, level, sdeflate size, sdeflate hash, zsdeflate size, zsdeflate hash)
const GOLDEN: &[(&str, i32, usize, u64, usize, u64)] = &[
    ("empty", 0, 0, 0xcbf29ce484222325, 6, 0x0c1c05f3b9316505),
    ("empty", 1, 0, 0xcbf29ce484222325, 6, 0x0c1c05f3b9316505),
    ("empty", 2, 0, 0xcbf29ce484222325, 6, 0x0c1c05f3b9316505),
    ("empty", 3, 0, 0xcbf29ce484222325, 6, 0x0c1c05f3b9316505),
    ("empty", 4, 0, 0xcbf29ce484222325, 6, 0x0c1c05f3b9316505),
    ("empty", 5, 0, 0xcbf29ce484222325, 6, 0x0c1c05f3b9316505),
    ("empty", 6, 0, 0xcbf29ce484222325, 6, 0x0c1c05f3b9316505),
    ("empty", 7, 0, 0xcbf29ce484222325, 6, 0x0c1c05f3b9316505),
    ("empty", 8, 0, 0xcbf29ce484222325, 6, 0x0c1c05f3b9316505),
    ("one", 0, 6, 0x1847d9f4965a7018, 12, 0xe1e20eebfb1cbe67),
    ("one", 1, 6, 0x1847d9f4965a7018, 12, 0xe1e20eebfb1cbe67),
    ("one", 2, 6, 0x1847d9f4965a7018, 12, 0xe1e20eebfb1cbe67),
    ("one", 3, 6, 0x1847d9f4965a7018, 12, 0xe1e20eebfb1cbe67),
    ("one", 4, 6, 0x1847d9f4965a7018, 12, 0xe1e20eebfb1cbe67),
    ("one", 5, 6, 0x1847d9f4965a7018, 12, 0xe1e20eebfb1cbe67),
    ("one", 6, 6, 0x1847d9f4965a7018, 12, 0xe1e20eebfb1cbe67),
    ("one", 7, 6, 0x1847d9f4965a7018, 12, 0xe1e20eebfb1cbe67),
    ("one", 8, 6, 0x1847d9f4965a7018, 12, 0xe1e20eebfb1cbe67),
    ("rand0", 0, 1005, 0x5990d7be4e8b7944, 1011, 0x51f89e452cee48e3),
    ("rand0", 1, 1005, 0x5990d7be4e8b7944, 1011, 0x51f89e452cee48e3),
    ("rand0", 2, 1005, 0x5990d7be4e8b7944, 1011, 0x51f89e452cee48e3),
    ("rand0", 3, 1005, 0x5990d7be4e8b7944, 1011, 0x51f89e452cee48e3),
    ("rand0", 4, 1005, 0x5990d7be4e8b7944, 1011, 0x51f89e452cee48e3),
    ("rand0", 5, 1005, 0x5990d7be4e8b7944, 1011, 0x51f89e452cee48e3),
    ("rand0", 6, 1005, 0x5990d7be4e8b7944, 1011, 0x51f89e452cee48e3),
    ("rand0", 7, 1005, 0x5990d7be4e8b7944, 1011, 0x51f89e452cee48e3),
    ("rand0", 8, 1005, 0x5990d7be4e8b7944, 1011, 0x51f89e452cee48e3),
    ("rand1", 0, 70010, 0x86ed145c490f5e75, 70016, 0xc468187da8150468),
    ("rand1", 1, 70010, 0x86ed145c490f5e75, 70016, 0xc468187da8150468),
    ("rand1", 2, 70010, 0x86ed145c490f5e75, 70016, 0xc468187da8150468),
    ("rand1", 3, 70010, 0x86ed145c490f5e75, 70016, 0xc468187da8150468),
    ("rand1", 4, 70010, 0x86ed145c490f5e75, 70016, 0xc468187da8150468),
    ("rand1", 5, 70010, 0x86ed145c490f5e75, 70016, 0xc468187da8150468),
    ("rand1", 6, 70010, 0x86ed145c490f5e75, 70016, 0xc468187da8150468),
    ("rand1", 7, 70010, 0x86ed145c490f5e75, 70016, 0xc468187da8150468),
    ("rand1", 8, 70010, 0x86ed145c490f5e75, 70016, 0xc468187da8150468),
    ("rand2", 0, 600060, 0xf5bad60bacd7cc8a, 600066, 0x0890bb40cd530535),
    ("rand2", 1, 600060, 0xf5bad60bacd7cc8a, 600066, 0x0890bb40cd530535),
    ("rand2", 2, 600060, 0xf5bad60bacd7cc8a, 600066, 0x0890bb40cd530535),
    ("rand2", 3, 600060, 0xf5bad60bacd7cc8a, 600066, 0x0890bb40cd530535),
    ("rand2", 4, 600060, 0xf5bad60bacd7cc8a, 600066, 0x0890bb40cd530535),
    ("rand2", 5, 600060, 0xf5bad60bacd7cc8a, 600066, 0x0890bb40cd530535),
    ("rand2", 6, 600060, 0xf5bad60bacd7cc8a, 600066, 0x0890bb40cd530535),
    ("rand2", 7, 600060, 0xf5bad60bacd7cc8a, 600066, 0x0890bb40cd530535),
    ("rand2", 8, 600060, 0xf5bad60bacd7cc8a, 600066, 0x0890bb40cd530535),
    ("zeros", 0, 1342, 0x28d6ca2ecb56a27b, 1348, 0x0bff8682a14aad23),
    ("zeros", 1, 1342, 0x28d6ca2ecb56a27b, 1348, 0x0bff8682a14aad23),
    ("zeros", 2, 322, 0x713db39e058df11a, 328, 0xe9b324cfb604bb52),
    ("zeros", 3, 322, 0x713db39e058df11a, 328, 0xe9b324cfb604bb52),
    ("zeros", 4, 322, 0x713db39e058df11a, 328, 0xe9b324cfb604bb52),
    ("zeros", 5, 322, 0x713db39e058df11a, 328, 0xe9b324cfb604bb52),
    ("zeros", 6, 322, 0x713db39e058df11a, 328, 0xe9b324cfb604bb52),
    ("zeros", 7, 322, 0x713db39e058df11a, 328, 0xe9b324cfb604bb52),
    ("zeros", 8, 322, 0x713db39e058df11a, 328, 0xe9b324cfb604bb52),
    ("abcd", 0, 6276, 0x7c0ac954d1bdbc72, 6282, 0x18fd2f329f14cec1),
    ("abcd", 1, 6265, 0x1cd3ec7d62f3940c, 6271, 0x35a9ec8741dd6d5d),
    ("abcd", 2, 6173, 0xf01a536a6117e326, 6179, 0x153ddb8d28a29243),
    ("abcd", 3, 6185, 0xabdd7f71447aebb0, 6191, 0xb3c0d3c782ac44dd),
    ("abcd", 4, 6256, 0x6e9761f2b6e22c3d, 6262, 0x5916ec7a642a4aba),
    ("abcd", 5, 6180, 0x9c6f6c2ddf5dbc06, 6186, 0x2090395bef160c79),
    ("abcd", 6, 6204, 0xeebbe328521232b8, 6210, 0x3dbad0f6f4175277),
    ("abcd", 7, 6204, 0xeebbe328521232b8, 6210, 0x3dbad0f6f4175277),
    ("abcd", 8, 6204, 0xeebbe328521232b8, 6210, 0x3dbad0f6f4175277),
    ("text", 0, 22859, 0xad32cc61fa3f9df6, 22865, 0x777fbf6f9b9a14ba),
    ("text", 1, 21944, 0x6a13711d83c2c626, 21950, 0x7e421d7229cd5638),
    ("text", 2, 20506, 0xc0a7e9d24fb97cc3, 20512, 0x3a35f1de92b143ad),
    ("text", 3, 19290, 0x94269aae87a9820e, 19296, 0x91fb158dc920e618),
    ("text", 4, 17955, 0x0cc1df6d8165c083, 17961, 0x0e40c8fe6ebdae17),
    ("text", 5, 17402, 0xc773b203cdf6b68b, 17408, 0xe4259faec5bf63bd),
    ("text", 6, 16856, 0x47988aea07846c25, 16862, 0x899b25f77713ad87),
    ("text", 7, 16411, 0x87e1933b89c88292, 16417, 0xa5bfa5edf05b9f72),
    ("text", 8, 16163, 0xe5f595035e5523de, 16169, 0xc1b2a9c63857617a),
    ("pattern", 0, 2493, 0x167131977edf523b, 2499, 0x85422687bd542d91),
    ("pattern", 1, 2493, 0x167131977edf523b, 2499, 0x85422687bd542d91),
    ("pattern", 2, 1595, 0x168645a2a84b994f, 1601, 0x1acd64336f7a3275),
    ("pattern", 3, 1595, 0x168645a2a84b994f, 1601, 0x1acd64336f7a3275),
    ("pattern", 4, 1595, 0x168645a2a84b994f, 1601, 0x1acd64336f7a3275),
    ("pattern", 5, 1595, 0x168645a2a84b994f, 1601, 0x1acd64336f7a3275),
    ("pattern", 6, 1595, 0x168645a2a84b994f, 1601, 0x1acd64336f7a3275),
    ("pattern", 7, 1595, 0x168645a2a84b994f, 1601, 0x1acd64336f7a3275),
    ("pattern", 8, 1595, 0x168645a2a84b994f, 1601, 0x1acd64336f7a3275),
    ("mesh", 0, 43530, 0x5387a723ad43ef8e, 43536, 0x52201c0b7dc25669),
    ("mesh", 1, 43911, 0x0de5b25667f5d8fb, 43917, 0xb29cf893b396d34e),
    ("mesh", 2, 44820, 0x304f58ee86a12e36, 44826, 0xded13250bed4b301),
    ("mesh", 3, 44909, 0x12d35eb5bd9646e8, 44915, 0x2f4cf0e483412d09),
    ("mesh", 4, 44886, 0x24662a09ca8004ca, 44892, 0x7d9b7fe93803f5d9),
    ("mesh", 5, 43311, 0xf7d7dd21be2b7450, 43317, 0x06d5da7fd8aeda69),
    ("mesh", 6, 43224, 0x35b77365750705b6, 43230, 0x286ffdac5221afc5),
    ("mesh", 7, 43233, 0x9c208a5d13759100, 43239, 0x39fb424fbf6c6821),
    ("mesh", 8, 43225, 0xa0efd979b6fba0db, 43231, 0x58b508927e0d277e),
];

#[test]
fn inputs() {
    for &(name, hash) in INPUT_HASHES {
        assert_eq!(fnv1a(&gen_input(name)), hash, "{name}");
    }
}

#[test]
fn golden() {
    let mut s = Sdefl::new();
    let mut input_name = "";
    let mut input = Vec::new();
    for &(name, lvl, size, hash, zsize, zhash) in GOLDEN {
        if name != input_name {
            input_name = name;
            input = gen_input(name);
        }
        let mut out = vec![0u8; sdefl_bound(input.len())];
        let n = sdeflate(&mut s, &mut out, &input, lvl);
        assert_eq!((n, fnv1a(&out[..n])), (size, hash), "sdeflate {name} lvl {lvl}");
        let n = zsdeflate(&mut s, &mut out, &input, lvl);
        assert_eq!((n, fnv1a(&out[..n])), (zsize, zhash), "zsdeflate {name} lvl {lvl}");
    }
}

#[test]
fn bound() {
    for (len, bound) in [
        (0, 21),
        (1, 27),
        (1000, 1026),
        (65535, 65561),
        (65536, 65567),
        (262144, 262190),
        (1000000, 1000101),
    ] {
        assert_eq!(sdefl_bound(len), bound);
    }
}

#[test]
fn errors() {
    let mut s = Sdefl::default();
    let input = gen_input("text");
    let mut out = vec![0u8; sdefl_bound(input.len())];
    let n = sdeflate(&mut s, &mut out, &input, SDEFL_LVL_DEF);
    assert!(n > 0);
    // output buffer too small
    assert_eq!(sdeflate(&mut s, &mut out[..n - 1], &input, SDEFL_LVL_DEF), 0);
    assert_eq!(zsdeflate(&mut s, &mut out[..n + 5], &input, SDEFL_LVL_DEF), 0);
    assert_eq!(zsdeflate(&mut s, &mut out[..n + 6], &input, SDEFL_LVL_DEF), n + 6);
    // invalid level
    assert_eq!(sdeflate(&mut s, &mut out, &input, SDEFL_LVL_MIN - 1), 0);
    assert_eq!(zsdeflate(&mut s, &mut out, &input, SDEFL_LVL_MAX + 1), 0);
}
