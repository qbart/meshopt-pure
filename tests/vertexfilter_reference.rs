//! Not a C++ test: filter decoders on random data (all bit patterns, counts 1..=67 for tail handling), hashed and
//! compared with C++ `meshopt_decodeFilter*` built for x86-64 with and without SIMD and for AArch64 with NEON (oracles
//! out of repo, NEON run under qemu). SSE oct8/color8 use `rsqrtps`/`rcpps`, whose precision depends on the CPU vendor,
//! so their hashes are not checked; NEON estimates (`frsqrte`/`frecpe`) are architecturally defined.

use meshoptimizer::{decode_filter_color, decode_filter_exp, decode_filter_oct, decode_filter_quat};

/// (filter, stride, scalar hash, SSE hash, NEON hash), in the oracle's order (the random stream is shared)
const EXPECTED: [(&str, usize, u64, Option<u64>, u64); 6] = [
    ("oct", 4, 0xc43f9079e59d5ddf, None, 0x99b9f7332f2556c8),
    ("oct", 8, 0x0f14cf655fa44e97, Some(0x5fe1a91b845219c0), 0xe02872d64d9be370),
    ("quat", 8, 0xc0357fade4e7a74b, Some(0x18031c3e218f6163), 0xd39d184fc0f65eb6),
    ("color", 8, 0x524bfd7a90576022, Some(0x8714a05685e9f2f8), 0x299eb67eca7739f5),
    ("color", 4, 0x12c4c503ae5ed91a, None, 0xef96e7260c6fff16),
    ("exp", 12, 0xc4ab217b2d350a72, Some(0xc4ab217b2d350a72), 0xc4ab217b2d350a72),
];

const SSE: bool = cfg!(all(feature = "simd", target_arch = "x86_64"));
const NEON: bool = cfg!(all(feature = "simd", target_arch = "aarch64"));

#[test]
fn filters_match_reference() -> meshoptimizer::Result<()> {
    let mut seed = 12345u32;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };

    for (filter, stride, scalar, sse, neon) in EXPECTED {
        let mut h = 0xcbf29ce484222325u64;

        for _ in 0..200 {
            for count in 1..=67 {
                let mut d: Vec<u8> = (0..count * stride).map(|_| (rnd() >> 24) as u8).collect();

                match filter {
                    "oct" => decode_filter_oct(&mut d, count, stride)?,
                    "quat" => decode_filter_quat(&mut d, count, stride)?,
                    "color" => decode_filter_color(&mut d, count, stride)?,
                    _ => decode_filter_exp(&mut d, count, stride)?,
                }

                h = d.iter().fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3));
            }
        }

        let expected = if SSE {
            sse
        } else if NEON {
            Some(neon)
        } else {
            Some(scalar)
        };
        if let Some(expected) = expected {
            assert_eq!(h, expected, "{filter} stride {stride} (sse: {SSE}, neon: {NEON})");
        }
    }

    Ok(())
}
