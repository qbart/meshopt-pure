//! Port of index codec tests from `demo/tests.cpp`.

use meshoptimizer::*;

// note: 4 6 5 triangle here is a combo-breaker:
// we encode it without rotating, a=next, c=next - this means we do *not* bump next to 6
// which means that the next triangle can't be encoded via next sequencing!
const INDEX_BUFFER: [u32; 12] = [0, 1, 2, 2, 1, 3, 4, 6, 5, 7, 8, 9];

#[rustfmt::skip]
const INDEX_DATA_V0: [u8; 27] = [
    0xe0, 0xf0, 0x10, 0xfe, 0xff, 0xf0, 0x0c, 0xff, 0x02, 0x02, 0x02, 0x00, 0x76, 0x87, 0x56, 0x67,
    0x78, 0xa9, 0x86, 0x65, 0x89, 0x68, 0x98, 0x01, 0x69, 0x00, 0x00,
];

// note: this exercises two features of v1 format, restarts (0 1 2) and last
const INDEX_BUFFER_TRICKY: [u32; 15] = [0, 1, 2, 2, 1, 3, 0, 1, 2, 2, 1, 5, 2, 1, 4];

#[rustfmt::skip]
const INDEX_DATA_V1: [u8; 24] = [
    0xe1, 0xf0, 0x10, 0xfe, 0x1f, 0x3d, 0x00, 0x0a, 0x00, 0x76, 0x87, 0x56, 0x67, 0x78, 0xa9, 0x86,
    0x65, 0x89, 0x68, 0x98, 0x01, 0x69, 0x00, 0x00,
];

const INDEX_SEQUENCE: [u32; 6] = [0, 1, 51, 2, 49, 1000];

#[rustfmt::skip]
const INDEX_SEQUENCE_V1: [u8; 13] = [
    0xd1, 0x00, 0x04, 0xcd, 0x01, 0x04, 0x07, 0x98, 0x1f, 0x00, 0x00, 0x00, 0x00,
];

fn encode_index(indices: &[u32], vertex_count: usize) -> Result<Vec<u8>> {
    let mut buffer = vec![0u8; encode_index_buffer_bound(indices.len(), vertex_count)?];
    let size = buffer.len();
    let n = encode_index_buffer(&mut buffer, size, indices, indices.len())?;
    buffer.truncate(n);
    Ok(buffer)
}

fn encode_sequence(indices: &[u32], vertex_count: usize) -> Result<Vec<u8>> {
    let mut buffer = vec![0u8; encode_index_sequence_bound(indices.len(), vertex_count)];
    let size = buffer.len();
    let n = encode_index_sequence(&mut buffer, size, indices, indices.len())?;
    buffer.truncate(n);
    Ok(buffer)
}

#[test]
fn decode_index_v0() -> Result<()> {
    let mut decoded = [0u32; INDEX_BUFFER.len()];
    decode_index_buffer(&mut decoded, INDEX_BUFFER.len(), &INDEX_DATA_V0, INDEX_DATA_V0.len())?;
    assert_eq!(decoded, INDEX_BUFFER);
    Ok(())
}

#[test]
fn decode_index_v1() -> Result<()> {
    let mut decoded = [0u32; INDEX_BUFFER_TRICKY.len()];
    decode_index_buffer(&mut decoded, INDEX_BUFFER_TRICKY.len(), &INDEX_DATA_V1, INDEX_DATA_V1.len())?;
    assert_eq!(decoded, INDEX_BUFFER_TRICKY);
    Ok(())
}

#[test]
fn decode_index_v1_more() -> Result<()> {
    #[rustfmt::skip]
    let input: [u8; 27] = [
        0xe1, 0xf0, 0x10, 0xfe, 0xff, 0xf0, 0x0c, 0xff, 0x02, 0x02, 0x02, 0x00, 0x76, 0x87, 0x56, 0x67,
        0x78, 0xa9, 0x86, 0x65, 0x89, 0x68, 0x98, 0x01, 0x69, 0x00, 0x00,
    ];

    let ib = [0u32, 1, 2, 2, 1, 3, 4, 6, 5, 7, 8, 9];

    let mut decoded = [0u32; 12];
    decode_index_buffer(&mut decoded, ib.len(), &input, input.len())?;
    assert_eq!(decoded, ib);
    Ok(())
}

#[test]
fn decode_index_v1_three_edges() -> Result<()> {
    #[rustfmt::skip]
    let input: [u8; 21] = [
        0xe1, 0xf0, 0x20, 0x30, 0x40, 0x00, 0x76, 0x87, 0x56, 0x67, 0x78, 0xa9, 0x86, 0x65, 0x89, 0x68,
        0x98, 0x01, 0x69, 0x00, 0x00,
    ];

    let ib = [0u32, 1, 2, 1, 0, 3, 2, 1, 4, 0, 2, 5];

    let mut decoded = [0u32; 12];
    decode_index_buffer(&mut decoded, ib.len(), &input, input.len())?;
    assert_eq!(decoded, ib);
    Ok(())
}

#[test]
fn decode_index_16() -> Result<()> {
    let buffer = encode_index(&INDEX_BUFFER, 10)?;

    let mut decoded = [0u16; INDEX_BUFFER.len()];
    decode_index_buffer(&mut decoded, INDEX_BUFFER.len(), &buffer, buffer.len())?;

    for (d, &i) in decoded.iter().zip(&INDEX_BUFFER) {
        assert_eq!(u32::from(*d), i);
    }
    Ok(())
}

#[test]
fn encode_index_memory_safe() -> Result<()> {
    let buffer = encode_index(&INDEX_BUFFER, 10)?;

    // check that encode is memory-safe
    for i in 0..=buffer.len() {
        let mut shortbuffer = vec![0u8; i];
        let result = encode_index_buffer(&mut shortbuffer, i, &INDEX_BUFFER, INDEX_BUFFER.len())?;

        if i == buffer.len() {
            assert_eq!(result, buffer.len());
        } else {
            assert_eq!(result, 0);
        }
    }
    Ok(())
}

#[test]
fn decode_index_memory_safe() -> Result<()> {
    let buffer = encode_index(&INDEX_BUFFER, 10)?;

    // check that decode is memory-safe
    let mut decoded = [0u32; INDEX_BUFFER.len()];

    for i in 0..=buffer.len() {
        let shortbuffer = buffer[..i].to_vec();
        let result = decode_index_buffer(&mut decoded, INDEX_BUFFER.len(), &shortbuffer, i);

        if i == buffer.len() {
            assert_eq!(result, Ok(()));
        } else {
            assert!(matches!(result, Err(Error::Decode(c)) if c < 0));
        }
    }
    Ok(())
}

#[test]
fn decode_index_reject_extra_bytes() -> Result<()> {
    let buffer = encode_index(&INDEX_BUFFER, 10)?;

    // check that decoder doesn't accept extra bytes after a valid stream
    let mut largebuffer = buffer.clone();
    largebuffer.push(0);

    let mut decoded = [0u32; INDEX_BUFFER.len()];
    assert!(decode_index_buffer(&mut decoded, INDEX_BUFFER.len(), &largebuffer, largebuffer.len()).is_err());
    Ok(())
}

#[test]
fn decode_index_reject_malformed_headers() -> Result<()> {
    let buffer = encode_index(&INDEX_BUFFER, 10)?;

    // check that decoder doesn't accept malformed headers
    let mut brokenbuffer = buffer.clone();
    brokenbuffer[0] = 0;

    let mut decoded = [0u32; INDEX_BUFFER.len()];
    assert!(decode_index_buffer(&mut decoded, INDEX_BUFFER.len(), &brokenbuffer, brokenbuffer.len()).is_err());
    Ok(())
}

#[test]
fn decode_index_reject_invalid_version() -> Result<()> {
    let buffer = encode_index(&INDEX_BUFFER, 10)?;

    // check that decoder doesn't accept invalid version
    let mut brokenbuffer = buffer.clone();
    brokenbuffer[0] |= 0x0f;

    let mut decoded = [0u32; INDEX_BUFFER.len()];
    assert!(decode_index_buffer(&mut decoded, INDEX_BUFFER.len(), &brokenbuffer, brokenbuffer.len()).is_err());
    Ok(())
}

#[test]
fn decode_index_malformed_vbyte() {
    #[rustfmt::skip]
    let input: [u8; 39] = [
        0xe1, 0x20, 0x20, 0x20, 0xff, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0xff, 0xff, 0xff, 0xff, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20,
    ];

    let mut decoded = [0u32; 66];
    assert!(decode_index_buffer(&mut decoded, 66, &input, input.len()).is_err());
}

#[test]
fn roundtrip_index_tricky() -> Result<()> {
    let buffer = encode_index(&INDEX_BUFFER_TRICKY, 6)?;

    let mut decoded = [0u32; INDEX_BUFFER_TRICKY.len()];
    decode_index_buffer(&mut decoded, INDEX_BUFFER_TRICKY.len(), &buffer, buffer.len())?;
    assert_eq!(decoded, INDEX_BUFFER_TRICKY);
    Ok(())
}

#[test]
fn encode_index_empty() -> Result<()> {
    let buffer = encode_index(&[], 0)?;

    decode_index_buffer::<u32>(&mut [], 0, &buffer, buffer.len())?;
    Ok(())
}

#[test]
fn decode_index_sequence_test() -> Result<()> {
    let mut decoded = [0u32; INDEX_SEQUENCE.len()];
    decode_index_sequence(&mut decoded, INDEX_SEQUENCE.len(), &INDEX_SEQUENCE_V1, INDEX_SEQUENCE_V1.len())?;
    assert_eq!(decoded, INDEX_SEQUENCE);
    Ok(())
}

#[test]
fn decode_index_sequence_16() -> Result<()> {
    let buffer = encode_sequence(&INDEX_SEQUENCE, 1001)?;

    let mut decoded = [0u16; INDEX_SEQUENCE.len()];
    decode_index_sequence(&mut decoded, INDEX_SEQUENCE.len(), &buffer, buffer.len())?;

    for (d, &i) in decoded.iter().zip(&INDEX_SEQUENCE) {
        assert_eq!(u32::from(*d), i);
    }
    Ok(())
}

#[test]
fn encode_index_sequence_memory_safe() -> Result<()> {
    let buffer = encode_sequence(&INDEX_SEQUENCE, 1001)?;

    // check that encode is memory-safe
    for i in 0..=buffer.len() {
        let mut shortbuffer = vec![0u8; i];
        let result = encode_index_sequence(&mut shortbuffer, i, &INDEX_SEQUENCE, INDEX_SEQUENCE.len())?;

        if i == buffer.len() {
            assert_eq!(result, buffer.len());
        } else {
            assert_eq!(result, 0);
        }
    }
    Ok(())
}

#[test]
fn decode_index_sequence_memory_safe() -> Result<()> {
    let buffer = encode_sequence(&INDEX_SEQUENCE, 1001)?;

    // check that decode is memory-safe
    let mut decoded = [0u32; INDEX_SEQUENCE.len()];

    for i in 0..=buffer.len() {
        let shortbuffer = buffer[..i].to_vec();
        let result = decode_index_sequence(&mut decoded, INDEX_SEQUENCE.len(), &shortbuffer, i);

        if i == buffer.len() {
            assert_eq!(result, Ok(()));
        } else {
            assert!(matches!(result, Err(Error::Decode(c)) if c < 0));
        }
    }
    Ok(())
}

#[test]
fn decode_index_sequence_reject_extra_bytes() -> Result<()> {
    let buffer = encode_sequence(&INDEX_SEQUENCE, 1001)?;

    // check that decoder doesn't accept extra bytes after a valid stream
    let mut largebuffer = buffer.clone();
    largebuffer.push(0);

    let mut decoded = [0u32; INDEX_SEQUENCE.len()];
    assert!(decode_index_sequence(&mut decoded, INDEX_SEQUENCE.len(), &largebuffer, largebuffer.len()).is_err());
    Ok(())
}

#[test]
fn decode_index_sequence_reject_malformed_headers() -> Result<()> {
    let buffer = encode_sequence(&INDEX_SEQUENCE, 1001)?;

    // check that decoder doesn't accept malformed headers
    let mut brokenbuffer = buffer.clone();
    brokenbuffer[0] = 0;

    let mut decoded = [0u32; INDEX_SEQUENCE.len()];
    assert!(decode_index_sequence(&mut decoded, INDEX_SEQUENCE.len(), &brokenbuffer, brokenbuffer.len()).is_err());
    Ok(())
}

#[test]
fn decode_index_sequence_reject_invalid_version() -> Result<()> {
    let buffer = encode_sequence(&INDEX_SEQUENCE, 1001)?;

    // check that decoder doesn't accept invalid version
    let mut brokenbuffer = buffer.clone();
    brokenbuffer[0] |= 0x0f;

    let mut decoded = [0u32; INDEX_SEQUENCE.len()];
    assert!(decode_index_sequence(&mut decoded, INDEX_SEQUENCE.len(), &brokenbuffer, brokenbuffer.len()).is_err());
    Ok(())
}

#[test]
fn encode_index_sequence_empty() -> Result<()> {
    let buffer = encode_sequence(&[], 0)?;

    decode_index_sequence::<u32>(&mut [], 0, &buffer, buffer.len())?;
    Ok(())
}
