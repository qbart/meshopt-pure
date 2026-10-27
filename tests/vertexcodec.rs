//! Port of vertex codec tests from `demo/tests.cpp`.

use std::sync::Mutex;

use meshoptimizer::*;

/// Tests that encode with the global vertex version must not run concurrently with tests that change it.
static VERSION_LOCK: Mutex<()> = Mutex::new(());

/// `struct PV { unsigned short px, py, pz; unsigned char nu, nv; unsigned short tx, ty; }`
const PV_SIZE: usize = 12;

#[rustfmt::skip]
const VERTEX_BUFFER: [[u16; 6]; 4] = [
    // px, py, pz, (nu | nv << 8), tx, ty
    [0, 0, 0, 0, 0, 0],
    [300, 0, 0, 0, 500, 0],
    [0, 300, 0, 0, 0, 500],
    [300, 300, 0, 0, 500, 500],
];

fn vertex_buffer() -> Vec<u8> {
    VERTEX_BUFFER.iter().flatten().flat_map(|v| v.to_le_bytes()).collect()
}

fn u16_bytes(data: &[u16]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn u32_bytes(data: &[u32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[rustfmt::skip]
const VERTEX_DATA_V0: [u8; 85] = [
    0xa0, 0x01, 0x3f, 0x00, 0x00, 0x00, 0x58, 0x57, 0x58, 0x01, 0x26, 0x00, 0x00, 0x00, 0x01, 0x0c,
    0x00, 0x00, 0x00, 0x58, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x3f, 0x00,
    0x00, 0x00, 0x17, 0x18, 0x17, 0x01, 0x26, 0x00, 0x00, 0x00, 0x01, 0x0c, 0x00, 0x00, 0x00, 0x17,
    0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00,
];

#[rustfmt::skip]
const VERTEX_DATA_V1: [u8; 44] = [
    0xa1, 0xee, 0xaa, 0xee, 0x00, 0x4b, 0x4b, 0x4b, 0x00, 0x00, 0x4b, 0x00, 0x00, 0x7d, 0x7d, 0x7d,
    0x00, 0x00, 0x7d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x62, 0x00, 0x62,
];

// This binary blob is a valid v1 encoding of vertex buffer but it used a custom version of
// the encoder that exercised all features of the format; because of this it is much larger
// and will never be produced by the encoder itself.
#[rustfmt::skip]
const VERTEX_DATA_V1_CUSTOM: [u8; 103] = [
    0xa1, 0xd4, 0x94, 0xd4, 0x01, 0x0e, 0x00, 0x58, 0x57, 0x58, 0x02, 0x02, 0x12, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x58, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x0e, 0x00, 0x7d, 0x7d, 0x7d, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7d, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x62,
];

fn encode(data: &[u8], vertex_count: usize, vertex_size: usize) -> Result<Vec<u8>> {
    let mut buffer = vec![0u8; encode_vertex_buffer_bound(vertex_count, vertex_size)?];
    let size = buffer.len();
    let n = encode_vertex_buffer(&mut buffer, size, data, vertex_count, vertex_size)?;
    buffer.truncate(n);
    Ok(buffer)
}

fn encode_level(data: &[u8], vertex_count: usize, vertex_size: usize, level: i32) -> Result<Vec<u8>> {
    let mut buffer = vec![0u8; encode_vertex_buffer_bound(vertex_count, vertex_size)?];
    let size = buffer.len();
    let n = encode_vertex_buffer_level(&mut buffer, size, data, vertex_count, vertex_size, level, -1)?;
    buffer.truncate(n);
    Ok(buffer)
}

fn is_decode_error(r: Result<()>) -> bool {
    matches!(r, Err(Error::Decode(c)) if c < 0)
}

#[test]
fn decode_vertex_v0() -> Result<()> {
    let mut decoded = vec![0u8; 4 * PV_SIZE];
    decode_vertex_buffer(&mut decoded, 4, PV_SIZE, &VERTEX_DATA_V0, VERTEX_DATA_V0.len())?;
    assert_eq!(decoded, vertex_buffer());
    Ok(())
}

#[test]
fn decode_vertex_v0_more() -> Result<()> {
    #[rustfmt::skip]
    let expected: [u8; 64] = [
        0, 0, 0, 0, 0, 1, 2, 8, 0, 2, 4, 16, 0, 3, 6, 24,
        0, 4, 8, 32, 0, 5, 10, 40, 0, 6, 12, 48, 0, 7, 14, 56,
        0, 8, 16, 64, 0, 9, 18, 72, 0, 10, 20, 80, 0, 11, 22, 88,
        0, 12, 24, 96, 0, 13, 26, 104, 0, 14, 28, 112, 0, 15, 30, 120,
    ];

    #[rustfmt::skip]
    let input: [u8; 65] = [
        0xa0, 0x00, 0x01, 0x2a, 0xaa, 0xaa, 0xaa, 0x02, 0x04, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44,
        0x03, 0x00, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
        0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00,
    ];

    let mut decoded = [0u8; 64];
    decode_vertex_buffer(&mut decoded, 16, 4, &input, input.len())?;
    assert_eq!(decoded, expected);
    Ok(())
}

#[test]
fn decode_vertex_v0_mode2() -> Result<()> {
    #[rustfmt::skip]
    let expected: [u8; 64] = [
        0, 0, 0, 0, 4, 5, 6, 7, 8, 10, 12, 14, 12, 15, 18, 21,
        16, 20, 24, 28, 20, 25, 30, 35, 24, 30, 36, 42, 28, 35, 42, 49,
        32, 40, 48, 56, 36, 45, 54, 63, 40, 50, 60, 70, 44, 55, 66, 77,
        48, 60, 72, 84, 52, 65, 78, 91, 56, 70, 84, 98, 60, 75, 90, 105,
    ];

    #[rustfmt::skip]
    let input: [u8; 69] = [
        0xa0, 0x02, 0x08, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x02, 0x0a, 0xaa, 0xaa, 0xaa, 0xaa,
        0xaa, 0xaa, 0xaa, 0x02, 0x0c, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0x02, 0x0e, 0xee, 0xee,
        0xee, 0xee, 0xee, 0xee, 0xee, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    let mut decoded = [0u8; 64];
    decode_vertex_buffer(&mut decoded, 16, 4, &input, input.len())?;
    assert_eq!(decoded, expected);
    Ok(())
}

#[test]
fn decode_vertex_v1() -> Result<()> {
    let mut decoded = vec![0u8; 4 * PV_SIZE];
    decode_vertex_buffer(&mut decoded, 4, PV_SIZE, &VERTEX_DATA_V1, VERTEX_DATA_V1.len())?;
    assert_eq!(decoded, vertex_buffer());
    Ok(())
}

#[test]
fn decode_vertex_v1_custom() -> Result<()> {
    let mut decoded = vec![0u8; 4 * PV_SIZE];
    decode_vertex_buffer(&mut decoded, 4, PV_SIZE, &VERTEX_DATA_V1_CUSTOM, VERTEX_DATA_V1_CUSTOM.len())?;
    assert_eq!(decoded, vertex_buffer());
    Ok(())
}

#[test]
fn decode_vertex_v1_deltas() -> Result<()> {
    #[rustfmt::skip]
    let expected: [u16; 64] = [
        248, 248, 240, 240, 249, 250, 243, 244, 250, 252, 246, 248, 251, 254, 249, 252,
        252, 256, 252, 256, 253, 258, 255, 260, 254, 260, 258, 264, 255, 262, 261, 268,
        256, 264, 264, 272, 257, 262, 267, 268, 258, 260, 270, 264, 259, 258, 273, 260,
        260, 256, 276, 256, 261, 254, 279, 252, 262, 252, 282, 248, 263, 250, 285, 244,
    ];

    #[rustfmt::skip]
    let input: [u8; 59] = [
        0xa1, 0x99, 0x99, 0x01, 0x2a, 0xaa, 0xaa, 0xaa, 0x02, 0x04, 0x44, 0x44, 0x44, 0x43, 0x33, 0x33,
        0x33, 0x02, 0x06, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x02, 0x08, 0x88, 0x88, 0x88, 0x87,
        0x77, 0x77, 0x77, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0xf8, 0x00, 0xf8, 0x00, 0xf0, 0x00, 0xf0, 0x00, 0x01, 0x01,
    ];

    let mut decoded = [0u8; 128];
    decode_vertex_buffer(&mut decoded, 16, 8, &input, input.len())?;
    assert_eq!(decoded.to_vec(), u16_bytes(&expected));
    Ok(())
}

/// Runs a versioned test body for both encoder versions (C: `for version in 0..=1 { meshopt_encodeVertexVersion(version); ... }`).
fn for_each_version(body: fn() -> Result<()>) -> Result<()> {
    let _guard = VERSION_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let result = (|| {
        for version in 0..=1 {
            encode_vertex_version(version)?;
            body()?;
        }
        Ok(())
    })();

    encode_vertex_version(1)?;
    result
}

fn decode_vertex_memory_safe_body() -> Result<()> {
    let vb = vertex_buffer();
    let buffer = encode(&vb, 4, PV_SIZE)?;

    // check that decode is memory-safe
    let mut decoded = vec![0u8; 4 * PV_SIZE];

    for i in 0..=buffer.len() {
        let shortbuffer = buffer[..i].to_vec();
        let result = decode_vertex_buffer(&mut decoded, 4, PV_SIZE, &shortbuffer, i);

        if i == buffer.len() {
            assert_eq!(result, Ok(()));
        } else {
            assert!(is_decode_error(result));
        }
    }
    Ok(())
}

#[test]
fn decode_vertex_memory_safe() -> Result<()> {
    for_each_version(decode_vertex_memory_safe_body)
}

fn decode_vertex_reject_extra_bytes_body() -> Result<()> {
    let vb = vertex_buffer();
    let buffer = encode(&vb, 4, PV_SIZE)?;

    // check that decoder doesn't accept extra bytes after a valid stream
    let mut largebuffer = buffer.clone();
    largebuffer.push(0);

    let mut decoded = vec![0u8; 4 * PV_SIZE];
    assert!(is_decode_error(decode_vertex_buffer(
        &mut decoded,
        4,
        PV_SIZE,
        &largebuffer,
        largebuffer.len()
    )));
    Ok(())
}

#[test]
fn decode_vertex_reject_extra_bytes() -> Result<()> {
    for_each_version(decode_vertex_reject_extra_bytes_body)
}

fn decode_vertex_reject_malformed_headers_body() -> Result<()> {
    let vb = vertex_buffer();
    let buffer = encode(&vb, 4, PV_SIZE)?;

    // check that decoder doesn't accept malformed headers
    let mut brokenbuffer = buffer.clone();
    brokenbuffer[0] = 0;

    let mut decoded = vec![0u8; 4 * PV_SIZE];
    assert!(is_decode_error(decode_vertex_buffer(
        &mut decoded,
        4,
        PV_SIZE,
        &brokenbuffer,
        brokenbuffer.len()
    )));
    Ok(())
}

#[test]
fn decode_vertex_reject_malformed_headers() -> Result<()> {
    for_each_version(decode_vertex_reject_malformed_headers_body)
}

fn decode_vertex_bit_groups_body() -> Result<()> {
    let mut data = [0u8; 16 * 4];

    // this tests 0/2/4/8 bit groups in one stream
    for i in 0..16 {
        data[i * 4] = 0;
        data[i * 4 + 1] = i as u8;
        data[i * 4 + 2] = (i * 2) as u8;
        data[i * 4 + 3] = (i * 8) as u8;
    }

    let buffer = encode(&data, 16, 4)?;

    let mut decoded = [0u8; 16 * 4];
    decode_vertex_buffer(&mut decoded, 16, 4, &buffer, buffer.len())?;
    assert_eq!(decoded, data);
    Ok(())
}

#[test]
fn decode_vertex_bit_groups() -> Result<()> {
    for_each_version(decode_vertex_bit_groups_body)
}

fn decode_vertex_bit_group_sentinels_body() -> Result<()> {
    let mut data = [0u8; 16 * 4];

    // this tests 0/2/4/8 bit groups and sentinels in one stream
    for i in 0..16 {
        if i == 7 || i == 13 {
            data[i * 4..i * 4 + 4].fill(42);
        } else {
            data[i * 4] = 0;
            data[i * 4 + 1] = i as u8;
            data[i * 4 + 2] = (i * 2) as u8;
            data[i * 4 + 3] = (i * 8) as u8;
        }
    }

    let buffer = encode(&data, 16, 4)?;

    let mut decoded = [0u8; 16 * 4];
    decode_vertex_buffer(&mut decoded, 16, 4, &buffer, buffer.len())?;
    assert_eq!(decoded, data);
    Ok(())
}

#[test]
fn decode_vertex_bit_group_sentinels() -> Result<()> {
    for_each_version(decode_vertex_bit_group_sentinels_body)
}

fn decode_vertex_bit_group_sentinel_count_body() -> Result<()> {
    #[rustfmt::skip]
    let expected: [u8; 13 * 4] = [
        0xff, 0, 0, 0, 0xfe, 0, 0, 0, 0xfd, 0, 0, 0, 0xfd, 0, 0, 0,
        0xfd, 0, 0, 0, 0xfc, 0, 0, 0, 0xfb, 0, 0, 0, 0xfb, 0, 0, 0,
        0xfa, 0, 0, 0, 0xfa, 0, 0, 0, 0xf9, 0, 0, 0, 0xf9, 0, 0, 0,
        0xf8, 0, 0, 0,
    ];

    // encodes several 2-bit sentinels including lane 12; lane 3 is clear so bit 30 does not alias bit 0 during counting
    #[rustfmt::skip]
    let input: [u8; 40] = [
        0xa1, 0xa9,
        0x01, 0xfc, 0x3c, 0xcc, 0xcc,
        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
    ];

    let mut decoded = [0u8; 13 * 4];
    decode_vertex_buffer(&mut decoded, 13, 4, &input, input.len())?;
    assert_eq!(decoded, expected);
    Ok(())
}

#[test]
fn decode_vertex_bit_group_sentinel_count() -> Result<()> {
    for_each_version(decode_vertex_bit_group_sentinel_count_body)
}

fn decode_vertex_deltas_body() -> Result<()> {
    let mut data = [0u16; 16 * 4];

    // this forces wider deltas by using values that cross byte boundary
    for i in 0..16u16 {
        let tri = if i < 8 { i } else { 16 - i };
        data[usize::from(i) * 4] = 0xf8 + i;
        data[usize::from(i) * 4 + 1] = 0xf8 + tri * 2;
        data[usize::from(i) * 4 + 2] = 0xf0 + i * 3;
        data[usize::from(i) * 4 + 3] = 0xf0 + tri * 4;
    }

    let bytes = u16_bytes(&data);
    let buffer = encode_level(&bytes, 16, 8, 2)?;

    let mut decoded = [0u8; 16 * 8];
    decode_vertex_buffer(&mut decoded, 16, 8, &buffer, buffer.len())?;
    assert_eq!(decoded.to_vec(), bytes);
    Ok(())
}

#[test]
fn decode_vertex_deltas() -> Result<()> {
    for_each_version(decode_vertex_deltas_body)
}

fn decode_vertex_bit_xor_body() -> Result<()> {
    let mut data = [0u32; 16 * 4];

    // this forces xors by using bit values at an offset
    for i in 0..16u32 {
        data[i as usize * 4] = i;
        data[i as usize * 4 + 1] = i << 2;
        data[i as usize * 4 + 2] = i << 15;
        data[i as usize * 4 + 3] = i << 28;
    }

    let bytes = u32_bytes(&data);
    let buffer = encode_level(&bytes, 16, 16, 3)?;

    let mut decoded = [0u8; 16 * 16];
    decode_vertex_buffer(&mut decoded, 16, 16, &buffer, buffer.len())?;
    assert_eq!(decoded.to_vec(), bytes);
    Ok(())
}

#[test]
fn decode_vertex_bit_xor() -> Result<()> {
    for_each_version(decode_vertex_bit_xor_body)
}

fn decode_vertex_large_body() -> Result<()> {
    let mut data = [0u8; 128 * 4];

    // this tests 0/2/4/8 bit groups in one stream
    for i in 0..128 {
        data[i * 4] = 0;
        data[i * 4 + 1] = i as u8;
        data[i * 4 + 2] = (i * 2) as u8;
        data[i * 4 + 3] = (i * 8) as u8;
    }

    let buffer = encode(&data, 128, 4)?;

    let mut decoded = [0u8; 128 * 4];
    decode_vertex_buffer(&mut decoded, 128, 4, &buffer, buffer.len())?;
    assert_eq!(decoded, data);
    Ok(())
}

#[test]
fn decode_vertex_large() -> Result<()> {
    for_each_version(decode_vertex_large_body)
}

fn decode_vertex_small_body() -> Result<()> {
    let mut data = [0u8; 13 * 4];

    // this tests 0/2/4/8 bit groups in one stream
    for i in 0..13 {
        data[i * 4] = 0;
        data[i * 4 + 1] = i as u8;
        data[i * 4 + 2] = (i * 2) as u8;
        data[i * 4 + 3] = (i * 8) as u8;
    }

    let buffer = encode(&data, 13, 4)?;

    let mut decoded = [0u8; 13 * 4];
    decode_vertex_buffer(&mut decoded, 13, 4, &buffer, buffer.len())?;
    assert_eq!(decoded, data);
    Ok(())
}

#[test]
fn decode_vertex_small() -> Result<()> {
    for_each_version(decode_vertex_small_body)
}

fn encode_vertex_empty_body() -> Result<()> {
    let buffer = encode(&[], 0, 16)?;

    decode_vertex_buffer(&mut [], 0, 16, &buffer, buffer.len())?;
    Ok(())
}

#[test]
fn encode_vertex_empty() -> Result<()> {
    for_each_version(encode_vertex_empty_body)
}

fn encode_vertex_memory_safe_body() -> Result<()> {
    let vb = vertex_buffer();
    let buffer = encode(&vb, 4, PV_SIZE)?;

    // check that encode is memory-safe
    for i in 0..=buffer.len() {
        let mut shortbuffer = vec![0u8; i];
        let result = encode_vertex_buffer(&mut shortbuffer, i, &vb, 4, PV_SIZE)?;

        if i == buffer.len() {
            assert_eq!(result, buffer.len());
        } else {
            assert_eq!(result, 0);
        }
    }
    Ok(())
}

#[test]
fn encode_vertex_memory_safe() -> Result<()> {
    for_each_version(encode_vertex_memory_safe_body)
}

#[test]
fn decode_version() {
    assert_eq!(decode_vertex_version(b"\xa0", 1), Ok(0));
    assert_eq!(decode_vertex_version(b"\xa1", 1), Ok(1));
    assert_eq!(decode_vertex_version(b"\xa1hello", 6), Ok(1));

    assert_eq!(decode_vertex_version(&[], 0), Err(Error::Decode(-1)));
    assert_eq!(decode_vertex_version(b"\xa7", 1), Err(Error::Decode(-1)));
    assert_eq!(decode_vertex_version(b"\xb1", 1), Err(Error::Decode(-1)));

    assert_eq!(decode_index_version(b"\xe0", 1), Ok(0));
    assert_eq!(decode_index_version(b"\xd1", 1), Ok(1));
    assert_eq!(decode_index_version(b"\xe1hello", 6), Ok(1));

    assert_eq!(decode_index_version(&[], 0), Err(Error::Decode(-1)));
    assert_eq!(decode_index_version(b"\xa7", 1), Err(Error::Decode(-1)));
    assert_eq!(decode_index_version(b"\xa1", 1), Err(Error::Decode(-1)));
}
