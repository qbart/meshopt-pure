//! Port of index generator tests from `demo/tests.cpp`.

use meshoptimizer::*;

fn bytes_u32(data: &[u32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn remap_custom() -> Result<()> {
    #[rustfmt::skip]
    let vb: [f32; 18] = [
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 0.0, 0.0,
        0.0, -0.0, 1.0,
    ];

    let mut remap = [0u32; 6];

    let res = generate_vertex_remap_custom(&mut remap, None, 6, &vb, 6, 4 * 3, None)?;
    assert_eq!(res, 4);
    for (i, &r) in remap.iter().enumerate().take(4) {
        assert_eq!(r, i as u32);
    }
    assert_eq!(remap[4], 1);
    assert_eq!(remap[5], 3);

    let res = generate_vertex_remap_custom(&mut remap, None, 6, &vb, 6, 4 * 3, Some(&mut |_, _| true))?;
    assert_eq!(res, 4);
    for (i, &r) in remap.iter().enumerate().take(4) {
        assert_eq!(r, i as u32);
    }
    assert_eq!(remap[4], 1);
    assert_eq!(remap[5], 3);

    let res = generate_vertex_remap_custom(&mut remap, None, 6, &vb, 6, 4 * 3, Some(&mut |_, _| false))?;
    assert_eq!(res, 6);
    for (i, &r) in remap.iter().enumerate() {
        assert_eq!(r, i as u32);
    }
    Ok(())
}

#[test]
fn filter_triangles() -> Result<()> {
    // v0/v3 match fully; v0/v4 match on prefix only
    #[rustfmt::skip]
    let vb = bytes_u32(&[
        1, 0, 10,
        2, 0, 20,
        3, 0, 30,
        1, 1, 10,
        1, 1, 99,
    ]);

    #[rustfmt::skip]
    let ib: [u32; 15] = [
        0, 1, 2, //
        3, 1, 2, // dup of (0,1,2)
        4, 1, 2, // dup prefix
        2, 1, 0, // opposite winding of (0,1,2)
        0, 4, 1, // degen prefix
    ];

    // prefix only: vertex key is first 4 bytes
    let mut sib = [0u32; 15];
    let slen = filter_index_buffer(&mut sib, &ib, 15, &vb, 5, 4, 12)?;

    let expecteds = [0, 1, 2, 2, 1, 0];
    assert_eq!(slen, expecteds.len());
    assert_eq!(&sib[..slen], &expecteds);

    // prefix+suffix: vertex key is first 4 and last 4 bytes
    let streams = [
        Stream {
            data: &vb[0..],
            size: 4,
            stride: 12,
        },
        Stream {
            data: &vb[8..],
            size: 4,
            stride: 12,
        },
    ];

    let mut mib = [0u32; 15];
    let mlen = filter_index_buffer_multi(&mut mib, &ib, 15, 5, &streams, 2)?;
    let expectedm = [0, 1, 2, 4, 1, 2, 2, 1, 0, 0, 4, 1];

    assert_eq!(mlen, expectedm.len());
    assert_eq!(&mib[..mlen], &expectedm);
    Ok(())
}

#[test]
fn adjacency() -> Result<()> {
    // 0 1/4
    // 2/5 3
    let vb = [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ];
    let ib = [0, 1, 2, 5, 4, 3];

    let mut adjib = [0u32; 12];
    generate_adjacency_index_buffer(&mut adjib, &ib, 6, &vb, 6, 12)?;

    #[rustfmt::skip]
    let expected = [
        // patch 0
        0, 0,
        1, 3,
        2, 2,

        // patch 1
        5, 0,
        4, 4,
        3, 3,
    ];

    assert_eq!(adjib, expected);
    Ok(())
}

#[test]
fn tessellation() -> Result<()> {
    // 0 1/4
    // 2/5 3
    let vb = [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ];
    let ib = [0, 1, 2, 5, 4, 3];

    let mut tessib = [0u32; 24];
    generate_tessellation_index_buffer(&mut tessib, &ib, 6, &vb, 6, 12)?;

    #[rustfmt::skip]
    let expected = [
        // patch 0
        0, 1, 2,
        0, 1,
        4, 5,
        2, 0,
        0, 1, 2,

        // patch 1
        5, 4, 3,
        2, 1,
        4, 3,
        3, 5,
        2, 1, 3,
    ];

    assert_eq!(tessib, expected);
    Ok(())
}

#[test]
fn provoking() -> Result<()> {
    // 0 1 2
    // 3 4 5
    #[rustfmt::skip]
    let ib = [
        0, 1, 3,
        3, 1, 4,
        1, 2, 4,
        4, 2, 5,
        0, 2, 4,
    ];

    let mut pib = [0u32; 15];
    let mut pre = [0u32; 6 + 5]; // limit is vertex count + triangle count
    let res = generate_provoking_index_buffer(&mut pib, &mut pre, &ib, 15, 6)?;

    #[rustfmt::skip]
    let expectedib = [
        0, 5, 1,
        1, 4, 0,
        2, 4, 1,
        3, 4, 2,
        4, 5, 2,
    ];

    let expectedre = [3, 1, 2, 5, 4, 0];

    assert_eq!(res, 6);
    assert_eq!(pib, expectedib);
    assert_eq!(&pre[..6], &expectedre);
    Ok(())
}
