//! `tools/codecfuzz.cpp`: decoders on untrusted data, vertex and meshlet encoder roundtrips.

use meshoptimizer::{
    decode_index_buffer, decode_index_sequence, decode_meshlet, decode_meshlet_raw, decode_vertex_buffer, encode_meshlet,
    encode_meshlet_bound, encode_vertex_buffer_bound, encode_vertex_buffer_level, encode_vertex_version,
};

// must be divisible by 3 for decode_index_buffer; should be >=64 to cover large vertex blocks
const COUNT: usize = 66;

fn fuzz_decoder_index<T: meshoptimizer::IndexType + Default + Clone>(data: &[u8], sequence: bool) {
    let mut destination = vec![T::default(); COUNT];

    let rc = if sequence {
        decode_index_sequence(&mut destination, COUNT, data, data.len())
    } else {
        decode_index_buffer(&mut destination, COUNT, data, data.len())
    };
    let _ = rc;
}

fn fuzz_decoder_vertex(data: &[u8], stride: usize) {
    let mut destination = vec![0u8; COUNT * stride];

    let rc = decode_vertex_buffer(&mut destination, COUNT, stride, data, data.len());
    let _ = rc;
}

fn fuzz_roundtrip(data: &[u8], stride: usize, level: i32) {
    let count = data.len() / stride;

    let bound = encode_vertex_buffer_bound(count, stride);
    assert!(bound.is_ok());
    let bound = bound.unwrap_or(0);
    let mut encoded = vec![0u8; bound];
    let mut decoded = vec![0u8; count * stride];

    let res = encode_vertex_buffer_level(&mut encoded, bound, data, count, stride, level, -1).unwrap_or(0);
    assert!(res > 0 && res <= bound);

    // encode again at the boundary to check for memory safety
    // this should produce the same output because encoder is deterministic
    let tail = &mut encoded[bound - res..];
    let rese = encode_vertex_buffer_level(tail, res, data, count, stride, level, -1);
    assert_eq!(rese, Ok(res));

    let rc = decode_vertex_buffer(&mut decoded, count, stride, &encoded[bound - res..], res);
    assert_eq!(rc, Ok(()));

    assert!(data[..count * stride] == decoded[..]);
}

fn align(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

fn fuzz_decode_meshlet(vertex_count: usize, triangle_count: usize, data: &[u8]) {
    // raw decoding: allowed to write align(count, 4) elements
    let mut rt = [0u32; 256];
    let mut rv = [0u32; 256];
    let _ = decode_meshlet_raw(
        &mut rv[256 - align(vertex_count, 4)..],
        vertex_count,
        &mut rt[256 - align(triangle_count, 4)..],
        triangle_count,
        data,
        data.len(),
    );

    // regular decoding: allowed to write align(count * size, 4) bytes
    // with variations for 3-byte triangles and 2-byte vertex references
    let mut rsv = [0u16; 256];
    let mut rbt = [0u8; 256 * 3];

    let _ = decode_meshlet(
        &mut rv[256 - vertex_count..],
        vertex_count,
        &mut rt[256 - triangle_count..],
        triangle_count,
        data,
        data.len(),
    );
    let _ = decode_meshlet(
        &mut rsv[256 - align(vertex_count, 2)..],
        vertex_count,
        &mut rt[256 - triangle_count..],
        triangle_count,
        data,
        data.len(),
    );
    let _ = decode_meshlet(
        &mut rv[256 - vertex_count..],
        vertex_count,
        &mut rbt[256 * 3 - align(triangle_count * 3, 4)..],
        triangle_count,
        data,
        data.len(),
    );
    let _ = decode_meshlet(
        &mut rsv[256 - align(vertex_count, 2)..],
        vertex_count,
        &mut rbt[256 * 3 - align(triangle_count * 3, 4)..],
        triangle_count,
        data,
        data.len(),
    );
}

fn rotations(a: u8, b: u8, c: u8) -> [u32; 3] {
    let (a, b, c) = (a as u32, b as u32, c as u32);

    let abc = a | (b << 8) | (c << 16);
    let bca = b | (c << 8) | (a << 16);
    let cba = c | (a << 8) | (b << 16);

    [abc, bca, cba]
}

fn fuzz_roundtrip_meshlet(data: &[u8]) {
    let triangle_count = (data.len() / 3).min(256);

    let mut buf = [0u8; 4096];
    let enc = encode_meshlet(&mut buf, 4096, &[], 0, data, triangle_count).unwrap_or(0);
    assert!(enc > 0);
    assert!(enc <= encode_meshlet_bound(0, triangle_count));

    let mut rt4 = [0u32; 256];
    let rc4 = decode_meshlet::<u32, u32>(&mut [], 0, &mut rt4, triangle_count, &buf, enc);
    assert_eq!(rc4, Ok(()));

    for i in 0..triangle_count {
        let tri = rt4[i];
        assert!(rotations(data[i * 3], data[i * 3 + 1], data[i * 3 + 2]).contains(&tri));
    }

    let mut rt3 = [0u8; 256 * 3];
    let rc3 = decode_meshlet::<u32, u8>(&mut [], 0, &mut rt3, triangle_count, &buf, enc);
    assert_eq!(rc3, Ok(()));

    for i in 0..triangle_count {
        let tri = rt3[i * 3] as u32 | (rt3[i * 3 + 1] as u32) << 8 | (rt3[i * 3 + 2] as u32) << 16;
        assert!(rotations(data[i * 3], data[i * 3 + 1], data[i * 3 + 2]).contains(&tri));
    }
}

fn fuzz_roundtrip_meshlet_v(data: &[u8]) {
    let vertex_count = (data.len() / 4).min(256);
    let vertices: Vec<u32> = data[..vertex_count * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();

    let tri = [0u8, 1, 2, 0];

    let mut buf = [0u8; 4096];
    let enc = encode_meshlet(&mut buf, 4096, &vertices, vertex_count, &tri, 1).unwrap_or(0);
    assert!(enc > 0);
    assert!(enc <= encode_meshlet_bound(vertex_count, 1));

    let mut rt = [0u8; 3];

    let mut rv4 = [0u32; 256];
    let rc4 = decode_meshlet(&mut rv4, vertex_count, &mut rt, 1, &buf, enc);
    assert_eq!(rc4, Ok(()));

    for i in 0..vertex_count {
        assert_eq!(rv4[i], vertices[i]);
    }

    let mut rv2 = [0u16; 256];
    let rc2 = decode_meshlet(&mut rv2, vertex_count, &mut rt, 1, &buf, enc);
    assert_eq!(rc2, Ok(()));

    for i in 0..vertex_count {
        assert_eq!(rv2[i], vertices[i] as u16);
    }
}

pub fn test_one_input(data: &[u8]) {
    // decode_index_buffer supports 2 and 4-byte indices
    fuzz_decoder_index::<u16>(data, false);
    fuzz_decoder_index::<u32>(data, false);

    // decode_index_sequence supports 2 and 4-byte indices
    fuzz_decoder_index::<u16>(data, true);
    fuzz_decoder_index::<u32>(data, true);

    // decode_vertex_buffer supports any strides divisible by 4 in 4-256 interval
    // it's a waste of time to check all of them, so we'll just check a few with different alignment mod 16
    fuzz_decoder_vertex(data, 4);
    fuzz_decoder_vertex(data, 16);
    fuzz_decoder_vertex(data, 24);
    fuzz_decoder_vertex(data, 32);

    // encode_vertex_buffer/decode_vertex_buffer should roundtrip for any stride, check a few with different alignment mod 16
    // this also checks memory safety properties of the encoder
    // to conserve time, we only check one version/level combination, biased towards version 1
    let data0 = data.first().copied().unwrap_or(0);
    let level = (data0 % 5) as i32;

    let _ = encode_vertex_version(if level < 4 { 1 } else { 0 });

    fuzz_roundtrip(data, 4, level);
    fuzz_roundtrip(data, 16, level);
    fuzz_roundtrip(data, 24, level);
    fuzz_roundtrip(data, 32, level);

    // validate that decode_meshlet works on untrusted data and is memory safe within documented limits
    if data.len() > 2 {
        fuzz_decode_meshlet(data[0] as usize + 1, data[1] as usize + 1, &data[2..]);
    }

    // validate that index data roundtrips in meshlet encoding modulo rotation
    fuzz_roundtrip_meshlet(data);

    // validate that vertex data roundtrips in meshlet encoding
    fuzz_roundtrip_meshlet_v(data);
}
