//! Port of vertex filter tests from `demo/tests.cpp`.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

use meshoptimizer::*;

fn u16_bytes(data: &[u16]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn u32_bytes(data: &[u32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i16s(data: &[u8]) -> Vec<i16> {
    data.as_chunks::<2>().0.iter().map(|b| i16::from_le_bytes(*b)).collect()
}

fn u16s(data: &[u8]) -> Vec<u16> {
    data.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect()
}

fn f32s(data: &[u8]) -> Vec<f32> {
    data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

#[test]
fn decode_filter_oct8() -> Result<()> {
    #[rustfmt::skip]
    let data: [u8; 16] = [
        0, 1, 127, 0,
        0, 187, 127, 1,
        255, 1, 127, 0,
        14, 130, 127, 1,
    ];

    #[rustfmt::skip]
    let expected: [u8; 16] = [
        0, 1, 127, 0,
        0, 159, 82, 1,
        255, 1, 127, 0,
        1, 130, 241, 1,
    ];

    // Aligned by 4
    let mut full = data;
    decode_filter_oct(&mut full, 4, 4)?;
    assert_eq!(full, expected);

    // Tail processing for unaligned data
    let mut tail = data[..12].to_vec();
    decode_filter_oct(&mut tail, 3, 4)?;
    assert_eq!(tail, expected[..12]);
    Ok(())
}

#[test]
fn decode_filter_oct12() -> Result<()> {
    #[rustfmt::skip]
    let data: [u16; 16] = [
        0, 1, 2047, 0,
        0, 1870, 2047, 1,
        2017, 1, 2047, 0,
        14, 1300, 2047, 1,
    ];

    #[rustfmt::skip]
    let expected: [u16; 16] = [
        0, 16, 32767, 0,
        0, 32621, 3088, 1,
        32764, 16, 471, 0,
        307, 28541, 16093, 1,
    ];

    // Aligned by 4
    let mut full = u16_bytes(&data);
    decode_filter_oct(&mut full, 4, 8)?;
    assert_eq!(full, u16_bytes(&expected));

    // Tail processing for unaligned data
    let mut tail = u16_bytes(&data[..12]);
    decode_filter_oct(&mut tail, 3, 8)?;
    assert_eq!(tail, u16_bytes(&expected[..12]));
    Ok(())
}

#[test]
fn decode_filter_quat12() -> Result<()> {
    #[rustfmt::skip]
    let data: [u16; 16] = [
        0, 1, 0, 0x7fc,
        0, 1870, 0, 0x7fd,
        2017, 1, 0, 0x7fe,
        14, 1300, 0, 0x7ff,
    ];

    #[rustfmt::skip]
    let expected: [u16; 16] = [
        32767, 0, 11, 0,
        0, 25013, 0, 21166,
        11, 0, 23504, 22830,
        158, 14715, 0, 29277,
    ];

    // Aligned by 4
    let mut full = u16_bytes(&data);
    decode_filter_quat(&mut full, 4, 8)?;
    assert_eq!(full, u16_bytes(&expected));

    // Tail processing for unaligned data
    let mut tail = u16_bytes(&data[..12]);
    decode_filter_quat(&mut tail, 3, 8)?;
    assert_eq!(tail, u16_bytes(&expected[..12]));
    Ok(())
}

#[test]
fn decode_filter_exp_test() -> Result<()> {
    let data: [u32; 4] = [0, 0xff000003, 0x02fffff7, 0xfe7fffff];

    let expected: [u32; 4] = [0, 0x3fc00000, 0xc2100000, 0x49fffffe];

    // Aligned by 4
    let mut full = u32_bytes(&data);
    decode_filter_exp(&mut full, 4, 4)?;
    assert_eq!(full, u32_bytes(&expected));

    // Tail processing for unaligned data
    let mut tail = u32_bytes(&data[..3]);
    decode_filter_exp(&mut tail, 3, 4)?;
    assert_eq!(tail, u32_bytes(&expected[..3]));
    Ok(())
}

#[test]
fn encode_filter_oct8() -> Result<()> {
    #[rustfmt::skip]
    let data: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.7071068, 0.0, 0.707168, 1.0,
        -0.7071068, 0.0, -0.707168, 1.0,
    ];

    #[rustfmt::skip]
    let expected: [u8; 16] = [
        0x7f, 0, 0x7f, 0,
        0, 0x81, 0x7f, 0,
        0x3f, 0, 0x7f, 0x7f,
        0x81, 0x40, 0x7f, 0x7f,
    ];

    let mut encoded = [0u8; 16];
    encode_filter_oct(&mut encoded, 4, 4, 8, &data)?;

    assert_eq!(encoded, expected);

    let mut decoded = encoded;
    decode_filter_oct(&mut decoded, 4, 4)?;

    for i in 0..16 {
        assert!((f32::from(decoded[i] as i8) / 127.0 - data[i]).abs() < 1e-2);
    }
    Ok(())
}

#[test]
fn encode_filter_oct12() -> Result<()> {
    #[rustfmt::skip]
    let data: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.7071068, 0.0, 0.707168, 1.0,
        -0.7071068, 0.0, -0.707168, 1.0,
    ];

    #[rustfmt::skip]
    let expected: [u16; 16] = [
        0x7ff, 0, 0x7ff, 0,
        0x0, 0xf801, 0x7ff, 0,
        0x3ff, 0, 0x7ff, 0x7fff,
        0xf801, 0x400, 0x7ff, 0x7fff,
    ];

    let mut encoded = [0u8; 32];
    encode_filter_oct(&mut encoded, 4, 8, 12, &data)?;

    assert_eq!(encoded.to_vec(), u16_bytes(&expected));

    let mut decoded = encoded;
    decode_filter_oct(&mut decoded, 4, 8)?;

    for (d, x) in i16s(&decoded).iter().zip(&data) {
        assert!((f32::from(*d) / 32767.0 - x).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn encode_filter_quat12() -> Result<()> {
    #[rustfmt::skip]
    let data: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.7071068, 0.0, 0.0, 0.707168,
        -0.7071068, 0.0, 0.0, -0.707168,
    ];

    #[rustfmt::skip]
    let expected: [u16; 16] = [
        0, 0, 0, 0x7fc,
        0, 0, 0, 0x7fd,
        0x7ff, 0, 0, 0x7ff,
        0x7ff, 0, 0, 0x7ff,
    ];

    let mut encoded = [0u8; 32];
    encode_filter_quat(&mut encoded, 4, 8, 12, &data)?;

    assert_eq!(encoded.to_vec(), u16_bytes(&expected));

    let mut decoded = encoded;
    decode_filter_quat(&mut decoded, 4, 8)?;
    let decoded = i16s(&decoded);

    for i in 0..4 {
        let dx = f32::from(decoded[i * 4]) / 32767.0;
        let dy = f32::from(decoded[i * 4 + 1]) / 32767.0;
        let dz = f32::from(decoded[i * 4 + 2]) / 32767.0;
        let dw = f32::from(decoded[i * 4 + 3]) / 32767.0;

        let dp = data[i * 4] * dx + data[i * 4 + 1] * dy + data[i * 4 + 2] * dz + data[i * 4 + 3] * dw;

        assert!((dp.abs() - 1.0).abs() < 1e-4);
    }
    Ok(())
}

const EXP_DATA: [f32; 4] = [1.0, -23.4, -0.1, 11.0];

// separate exponents: each component gets its own value
const EXP_EXPECTED1: [u32; 4] = [0xf3002000, 0xf7ffd133, 0xefffcccd, 0xf6002c00];

// shared exponents (vector): all components of each vector get the same value
const EXP_EXPECTED2: [u32; 4] = [0xf7000200, 0xf7ffd133, 0xf6ffff9a, 0xf6002c00];

// shared exponents (component): each component gets the same value across all vectors
const EXP_EXPECTED3: [u32; 4] = [0xf3002000, 0xf7ffd133, 0xf3fffccd, 0xf7001600];

#[test]
fn encode_filter_exp_test() -> Result<()> {
    let data = EXP_DATA;

    let mut encoded1 = [0u8; 16];
    encode_filter_exp(&mut encoded1, 2, 8, 15, &data, EncodeExpMode::Separate)?;

    let mut encoded2 = [0u8; 16];
    encode_filter_exp(&mut encoded2, 2, 8, 15, &data, EncodeExpMode::SharedVector)?;

    let mut encoded3 = [0u8; 16];
    encode_filter_exp(&mut encoded3, 2, 8, 15, &data, EncodeExpMode::SharedComponent)?;

    assert_eq!(encoded1.to_vec(), u32_bytes(&EXP_EXPECTED1));
    assert_eq!(encoded2.to_vec(), u32_bytes(&EXP_EXPECTED2));
    assert_eq!(encoded3.to_vec(), u32_bytes(&EXP_EXPECTED3));

    let mut decoded1 = encoded1;
    decode_filter_exp(&mut decoded1, 2, 8)?;

    let mut decoded2 = encoded2;
    decode_filter_exp(&mut decoded2, 2, 8)?;

    let mut decoded3 = encoded3;
    decode_filter_exp(&mut decoded3, 2, 8)?;

    for (i, x) in data.iter().enumerate() {
        assert!((f32s(&decoded1)[i] - x).abs() < 1e-3);
        assert!((f32s(&decoded2)[i] - x).abs() < 1e-3);
        assert!((f32s(&decoded3)[i] - x).abs() < 1e-3);
    }
    Ok(())
}

#[test]
fn encode_filter_exp_zero() -> Result<()> {
    let data: [f32; 4] = [0.0, -0.0, 1.1754944e-38, -1.1754944e-38];
    let expected: [u32; 4] = [0xf2000000, 0xf2000000, 0x8e000000, 0x8e000000];

    let mut encoded = [0u8; 16];
    encode_filter_exp(&mut encoded, 4, 4, 15, &data, EncodeExpMode::Separate)?;

    assert_eq!(encoded.to_vec(), u32_bytes(&expected));

    let mut decoded = encoded;
    decode_filter_exp(&mut decoded, 4, 4)?;

    for d in f32s(&decoded) {
        assert_eq!(d, 0.0);
    }
    Ok(())
}

#[test]
fn encode_filter_exp_overflow() -> Result<()> {
    // at bits=24, values with an all-ones mantissa round up to 2^23 which must be clamped
    let data: [f32; 4] = [
        0.99999994, // largest float below 1.0
        -0.99999994,
        255.99998, // largest float below 256.0
        1.0,
    ];
    let expected: [u32; 4] = [
        0xe97fffff, //
        0xe9800000, // -2^23 is representable as is
        0xf17fffff, //
        0xea400000,
    ];

    let mut encoded = [0u8; 16];
    encode_filter_exp(&mut encoded, 4, 4, 24, &data, EncodeExpMode::Separate)?;

    assert_eq!(encoded.to_vec(), u32_bytes(&expected));

    let mut decoded = encoded;
    decode_filter_exp(&mut decoded, 4, 4)?;
    let decoded = f32s(&decoded);

    assert_eq!(decoded[0], 0.99999988);
    assert_eq!(decoded[1], -1.0);
    assert_eq!(decoded[2], 255.99997);
    assert_eq!(decoded[3], 1.0);
    Ok(())
}

#[test]
fn encode_filter_exp_zero_shared() -> Result<()> {
    let data: [f32; 4] = [0.0, 0.1, -0.025, -0.0];

    // shared exponents (vector)
    let expected1: [u32; 4] = [0xef000000, 0xef003333, 0xedffcccd, 0xed000000];

    // shared exponents (component)
    let expected2: [u32; 4] = [0xed000000, 0xef003333, 0xedffcccd, 0xef000000];

    let mut encoded1 = [0u8; 16];
    encode_filter_exp(&mut encoded1, 2, 8, 15, &data, EncodeExpMode::SharedVector)?;

    let mut encoded2 = [0u8; 16];
    encode_filter_exp(&mut encoded2, 2, 8, 15, &data, EncodeExpMode::SharedComponent)?;

    assert_eq!(encoded1.to_vec(), u32_bytes(&expected1));
    assert_eq!(encoded2.to_vec(), u32_bytes(&expected2));

    let mut decoded1 = encoded1;
    decode_filter_exp(&mut decoded1, 2, 8)?;
    let decoded1 = f32s(&decoded1);

    let mut decoded2 = encoded2;
    decode_filter_exp(&mut decoded2, 2, 8)?;
    let decoded2 = f32s(&decoded2);

    for (i, x) in data.iter().enumerate() {
        assert!((decoded1[i] - x).abs() < 1e-5);
        assert!((decoded2[i] - x).abs() < 1e-5);
    }

    // zeroes should be preserved exactly
    assert!(decoded1[0] == 0.0 && decoded1[3] == 0.0);
    assert!(decoded2[0] == 0.0 && decoded2[3] == 0.0);
    Ok(())
}

#[test]
fn encode_filter_exp_alias() -> Result<()> {
    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();

    let mut encoded1 = EXP_DATA;
    encode_filter_exp_in_place(&mut encoded1, 2, 8, 15, EncodeExpMode::Separate)?;

    let mut encoded2 = EXP_DATA;
    encode_filter_exp_in_place(&mut encoded2, 2, 8, 15, EncodeExpMode::SharedVector)?;

    let mut encoded3 = EXP_DATA;
    encode_filter_exp_in_place(&mut encoded3, 2, 8, 15, EncodeExpMode::SharedComponent)?;

    assert_eq!(bits(&encoded1), EXP_EXPECTED1);
    assert_eq!(bits(&encoded2), EXP_EXPECTED2);
    assert_eq!(bits(&encoded3), EXP_EXPECTED3);
    Ok(())
}

#[test]
fn encode_filter_exp_clamp() -> Result<()> {
    let data = EXP_DATA;

    // separate exponents: each component gets its own value
    // note: third value is exponent clamped
    let expected: [u32; 4] = [0xf3002000, 0xf7ffd133, 0xf2fff99a, 0xf6002c00];

    let mut encoded = [0u8; 16];
    encode_filter_exp(&mut encoded, 2, 8, 15, &data, EncodeExpMode::Clamped)?;

    assert_eq!(encoded.to_vec(), u32_bytes(&expected));

    let mut decoded = encoded;
    decode_filter_exp(&mut decoded, 2, 8)?;

    for (d, x) in f32s(&decoded).iter().zip(&data) {
        assert!((d - x).abs() < 1e-3);
    }
    Ok(())
}

#[rustfmt::skip]
const COLOR_DATA: [f32; 16] = [
    1.0, 0.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 0.5,
    0.0, 0.0, 1.0, 0.25,
    0.4, 0.4, 0.4, 0.75,
];

#[test]
fn encode_filter_color8() -> Result<()> {
    let data = COLOR_DATA;

    #[rustfmt::skip]
    let expected: [u8; 16] = [
        0x40, 0x7f, 0xc1, 0xff,
        0x7f, 0x00, 0x7f, 0xc0,
        0x40, 0x81, 0xc0, 0xa0,
        0x66, 0x00, 0x00, 0xdf,
    ];

    let mut encoded = [0u8; 16];
    encode_filter_color(&mut encoded, 4, 4, 8, &data)?;

    assert_eq!(encoded, expected);

    let mut decoded = encoded;
    decode_filter_color(&mut decoded, 4, 4)?;

    for i in 0..16 {
        assert!((f32::from(decoded[i]) / 255.0 - data[i]).abs() < 1e-2);
    }

    // ensure grayscale is preserved
    assert!(decoded[12] == decoded[13] && decoded[12] == decoded[14]);
    Ok(())
}

#[test]
fn encode_filter_color12() -> Result<()> {
    let data = COLOR_DATA;

    #[rustfmt::skip]
    let expected: [u16; 16] = [
        0x0400, 0x07ff, 0xfc01, 0x0fff,
        0x07ff, 0x0000, 0x07ff, 0x0c00,
        0x0400, 0xf801, 0xfc00, 0x0a00,
        0x0666, 0x0000, 0x0000, 0x0dff,
    ];

    let mut encoded = [0u8; 32];
    encode_filter_color(&mut encoded, 4, 8, 12, &data)?;

    assert_eq!(encoded.to_vec(), u16_bytes(&expected));

    let mut decoded = encoded;
    decode_filter_color(&mut decoded, 4, 8)?;
    let decoded = u16s(&decoded);

    for i in 0..16 {
        assert!((f32::from(decoded[i]) / 65535.0 - data[i]).abs() < 1e-3);
    }

    // ensure grayscale is preserved
    assert!(decoded[12] == decoded[13] && decoded[12] == decoded[14]);
    Ok(())
}
