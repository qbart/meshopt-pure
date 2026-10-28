//! Sanity tests for analyzers and vertex fetch optimizer (no 1:1 counterpart in `demo/tests.cpp`).

use meshoptimizer::*;

#[test]
fn analyze_vertex_cache_basic() -> Result<()> {
    // two triangles sharing an edge: 4 unique vertices, all transformed once
    let ib = [0u32, 1, 2, 2, 1, 3];
    let stats = analyze_vertex_cache(&ib, 6, 4, 16, 0, 0)?;
    assert_eq!(stats.vertices_transformed, 4);
    assert_eq!(stats.warps_executed, 1);
    assert_eq!(stats.acmr, 2.0);
    assert_eq!(stats.atvr, 1.0);

    // warp of 3 vertices forces a flush before the second triangle
    let stats = analyze_vertex_cache(&ib, 6, 4, 16, 3, 0)?;
    assert_eq!(stats.vertices_transformed, 6);
    assert_eq!(stats.warps_executed, 2);

    assert_eq!(analyze_vertex_cache(&ib, 6, 3, 16, 0, 0), Err(Error::IndexOutOfRange));
    assert!(analyze_vertex_cache(&ib, 5, 4, 16, 0, 0).is_err());
    assert!(analyze_vertex_cache(&ib, 6, 4, 2, 0, 0).is_err());
    Ok(())
}

#[test]
fn analyze_vertex_fetch_basic() -> Result<()> {
    let ib = [0u32, 1, 2, 2, 1, 3];
    let stats = analyze_vertex_fetch(&ib, 6, 4, 16)?;
    assert_eq!(stats.bytes_fetched, 64);
    assert_eq!(stats.overfetch, 1.0);

    // vertices 64 bytes apart each touch a separate line
    let ib = [0u32, 4, 8];
    let stats = analyze_vertex_fetch(&ib, 3, 9, 16)?;
    assert_eq!(stats.bytes_fetched, 192);
    assert_eq!(stats.overfetch, 4.0);

    assert!(analyze_vertex_fetch(&ib, 3, 9, 0).is_err());
    Ok(())
}

#[test]
fn optimize_vertex_fetch_basic() -> Result<()> {
    let ib = [3u32, 1, 0, 0, 1, 2];
    let mut remap = [0u32; 5];
    assert_eq!(optimize_vertex_fetch_remap(&mut remap, &ib, 6, 5)?, 4);
    assert_eq!(remap, [2, 1, 3, 0, !0]);

    let vb: Vec<u8> = (0..5u8).flat_map(|v| [v, v + 10]).collect();
    let mut ibm = ib;
    let mut dst = vec![0u8; 10];
    assert_eq!(optimize_vertex_fetch(&mut dst, &mut ibm, 6, &vb, 5, 2)?, 4);
    assert_eq!(ibm, [0, 1, 2, 2, 1, 3]);
    assert_eq!(&dst[..8], &[3, 13, 1, 11, 0, 10, 2, 12]);

    let mut vbi = vb.clone();
    let mut ibi = ib;
    assert_eq!(optimize_vertex_fetch_in_place(&mut vbi, &mut ibi, 6, 5, 2)?, 4);
    assert_eq!(ibi, ibm);
    assert_eq!(&vbi[..8], &dst[..8]);

    assert!(optimize_vertex_fetch(&mut dst[..4], &mut ibm, 6, &vb, 5, 2).is_err());
    Ok(())
}
