//! Port of meshlet codec tests from `demo/tests.cpp`.

use meshoptimizer::*;

#[rustfmt::skip]
const SAFETY_TRIANGLES: [u8; 5 * 3] = [
    0, 1, 2,
    2, 1, 3,
    3, 5, 4,
    2, 0, 6,
    6, 6, 6,
];

const VERTICES: [u32; 7] = [5, 12, 140, 0, 12389, 123456789, 7];

#[test]
fn encode_meshlet_bound_test() -> Result<()> {
    let bound = encode_meshlet_bound(7, 5);

    let mut enc = [0u8; 256];
    let size = encode_meshlet(&mut enc, 256, &VERTICES, 7, &SAFETY_TRIANGLES, 5)?;
    assert!(size > 0 && size <= bound);

    assert_eq!(encode_meshlet(&mut enc, size - 1, &VERTICES, 7, &SAFETY_TRIANGLES, 5)?, 0);
    Ok(())
}

trait Element: Copy + Default {
    fn value(self) -> u32;
}

impl Element for u8 {
    fn value(self) -> u32 {
        u32::from(self)
    }
}

impl Element for u16 {
    fn value(self) -> u32 {
        u32::from(self)
    }
}

impl Element for u32 {
    fn value(self) -> u32 {
        self
    }
}

fn validate_decode_meshlet<V: IndexType + Element, T: MeshletTriangleType + Element>(
    data: &[u8],
    vertices: &[u32],
    triangles: &[u8],
) -> Result<()> {
    let (vertex_count, triangle_count) = (vertices.len(), triangles.len() / 3);

    let mut rv = vec![V::default(); vertex_count];
    let mut rt = vec![<T as Default>::default(); triangle_count * T::PER_TRIANGLE];

    decode_meshlet(&mut rv, vertex_count, &mut rt, triangle_count, data, data.len())?;

    for (r, &v) in rv.iter().zip(vertices) {
        assert_eq!(r.value(), V::from_u32(v).value());
    }

    for j in 0..triangle_count {
        let (a, b, c) = (
            u32::from(triangles[j * 3]),
            u32::from(triangles[j * 3 + 1]),
            u32::from(triangles[j * 3 + 2]),
        );

        let tri = if T::PER_TRIANGLE == 3 {
            rt[j * 3].value() | (rt[j * 3 + 1].value() << 8) | (rt[j * 3 + 2].value() << 16)
        } else {
            rt[j].value()
        };

        let abc = a | (b << 8) | (c << 16);
        let bca = b | (c << 8) | (a << 16);
        let cba = c | (a << 8) | (b << 16);

        assert!(tri == abc || tri == bca || tri == cba);
    }
    Ok(())
}

fn is_decode_error(r: Result<()>) -> bool {
    matches!(r, Err(Error::Decode(c)) if c < 0)
}

#[test]
fn decode_meshlet_safety() -> Result<()> {
    let mut encb = [0u8; 256];
    let size = encode_meshlet(&mut encb, 256, &VERTICES, 7, &SAFETY_TRIANGLES, 5)?;
    assert!(size > 0);

    // the encoded buffer is sliced exactly, so any over-read would panic
    let enc = &encb[..size];

    validate_decode_meshlet::<u32, u32>(enc, &VERTICES, &SAFETY_TRIANGLES)?;
    validate_decode_meshlet::<u32, u8>(enc, &VERTICES, &SAFETY_TRIANGLES)?;
    validate_decode_meshlet::<u16, u32>(enc, &VERTICES, &SAFETY_TRIANGLES)?;
    validate_decode_meshlet::<u16, u8>(enc, &VERTICES, &SAFETY_TRIANGLES)?;

    // any truncated input should not be decodable; we check both prefix and suffix truncation
    let mut rv = [0u32; 7];
    let mut rt = [0u32; 5];

    for i in 1..size {
        assert!(is_decode_error(decode_meshlet(&mut rv, 7, &mut rt, 5, &enc[..i], i)));
        assert!(is_decode_error(decode_meshlet(&mut rv, 7, &mut rt, 5, &enc[i..], size - i)));
    }

    // short representations
    let mut rvs = [0u16; 7];
    let mut rts = [0u8; 5 * 3];

    for i in 1..size {
        assert!(is_decode_error(decode_meshlet(&mut rvs, 7, &mut rts, 5, &enc[..i], i)));
        assert!(is_decode_error(decode_meshlet(&mut rvs, 7, &mut rts, 5, &enc[i..], size - i)));
    }

    let mut rvr = [0u32; 8];
    let mut rtr = [0u32; 8];

    for i in 1..size {
        assert!(is_decode_error(decode_meshlet_raw(&mut rvr, 7, &mut rtr, 5, &enc[..i], i)));
        assert!(is_decode_error(decode_meshlet_raw(&mut rvr, 7, &mut rtr, 5, &enc[i..], size - i)));
    }

    // otherwise, decodeMeshlet and decodeMeshletRaw should agree
    decode_meshlet(&mut rv, 7, &mut rt, 5, enc, size)?;
    decode_meshlet_raw(&mut rvr, 7, &mut rtr, 5, enc, size)?;

    assert_eq!(rv, rvr[..7]);
    assert_eq!(rt, rtr[..5]);
    Ok(())
}

#[test]
fn decode_meshlet_basic() -> Result<()> {
    #[rustfmt::skip]
    let triangles: [u8; 5 * 3] = [
        0, 1, 2,
        2, 1, 3,
        4, 3, 5,
        2, 0, 6,
        6, 6, 6,
    ];

    #[rustfmt::skip]
    let encoded: [u8; 46] = [
        0x0a, 0x0c, 0xfe, 0x19, 0x01, 0xc8, 0x60, 0x00, 0x00, 0x5e, 0x39, 0xb7, 0x0e, 0x1d, 0x9a, 0xb7,
        0x0e, 0x00, 0x00, 0x00, 0x00, 0x04, 0x03, 0x05, 0x02, 0x00, 0x06, 0x06, 0x06, 0x06, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x87, 0xff, 0x2c, 0xff, 0x0f,
    ];

    let mut rv = [0u32; 7];
    let mut rt = [0u8; 5 * 3];
    decode_meshlet(&mut rv, 7, &mut rt, 5, &encoded, encoded.len())?;

    assert_eq!(rv, VERTICES);
    assert_eq!(rt, triangles);
    Ok(())
}

#[test]
fn decode_meshlet_typical() -> Result<()> {
    #[rustfmt::skip]
    let triangles: [u8; 44 * 3] = [
        0, 1, 2, 0, 2, 3, 3, 2, 4, 3, 4, 5, 0, 3, 6, 6, 3, 5, 0, 6, 7, 7, 6, 8,
        8, 6, 5, 7, 8, 9, 8, 5, 10, 10, 5, 4, 10, 4, 11, 11, 4, 12, 11, 12, 13, 10, 11, 14,
        10, 14, 8, 14, 11, 15, 15, 11, 13, 15, 13, 16, 15, 16, 14, 16, 13, 17, 14, 16, 18, 14, 18, 8,
        18, 16, 19, 19, 16, 20, 20, 16, 17, 20, 17, 21, 20, 21, 22, 20, 22, 19, 22, 21, 23, 19, 22, 24,
        19, 24, 25, 19, 25, 18, 18, 25, 26, 18, 26, 8, 8, 26, 9, 22, 23, 27, 22, 27, 24, 27, 23, 28,
        27, 28, 29, 27, 29, 24, 29, 28, 30, 24, 29, 31,
    ];

    #[rustfmt::skip]
    let vertices: [u32; 32] = [
        10, 11, 9, 12, 8, 13, 14, 15, 16, 17, 18, 19, 0, 20, 21, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
    ];

    #[rustfmt::skip]
    let encoded: [u8; 53] = [
        0x14, 0x05, 0x04, 0x09, 0x08, 0x27, 0x26, 0x05, 0x05, 0x04, 0x08, 0x0d, 0x0e, 0x08, 0x11, 0x13,
        0x12, 0x08, 0x09, 0x16, 0x17, 0x18, 0x18, 0x0d, 0x03, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x0c,
        0x02, 0x38, 0x24, 0x43, 0x34, 0x20, 0x80, 0x61, 0x03, 0x61, 0x16, 0x26, 0x03, 0x10, 0x66, 0x10,
        0x12, 0xe3, 0x61, 0x10, 0x66,
    ];

    let mut rv = [0u32; 32];
    let mut rt = [0u8; 44 * 3];
    decode_meshlet(&mut rv, 32, &mut rt, 44, &encoded, encoded.len())?;

    assert_eq!(rv, vertices);
    assert_eq!(rt, triangles);
    Ok(())
}
