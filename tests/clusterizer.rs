//! Port of meshlet builder tests from `demo/tests.cpp`.

use meshoptimizer::*;

#[test]
fn meshlets_empty() -> Result<()> {
    let vbd = [0.0f32; 4 * 3];

    let mut ml = [Meshlet::default(); 1];
    let mut mv = [0u32; 4];
    let mut mt = [0u8; 8];
    let mc = build_meshlets(&mut ml, &mut mv, &mut mt, &[], 0, &vbd, 4, 4 * 3, 64, 64, 0.0)?;
    assert_eq!(mc, 0);
    Ok(())
}

fn check_two_triangles(ml: &[Meshlet], mv: &[u32], mt: &[u8], ibd: &[u32; 6]) {
    assert_eq!(ml[0].triangle_count, 2);
    assert_eq!(ml[0].vertex_count, 4);

    let tri0 = [mv[usize::from(mt[0])], mv[usize::from(mt[1])], mv[usize::from(mt[2])]];
    let tri1 = [mv[usize::from(mt[3])], mv[usize::from(mt[4])], mv[usize::from(mt[5])]];

    // technically triangles could also be flipped in the meshlet but for now just assume they aren't
    assert_eq!(tri0, ibd[0..3]);
    assert_eq!(tri1, ibd[3..6]);
}

#[test]
fn meshlets_dense() -> Result<()> {
    let vbd = [0.0f32; 4 * 3];
    let ibd = [0, 2, 1, 1, 2, 3];

    let mut ml = [Meshlet::default(); 1];
    let mut mv = [0u32; 4];
    let mut mt = [0u8; 8];
    let mc = build_meshlets(&mut ml, &mut mv, &mut mt, &ibd, 6, &vbd, 4, 4 * 3, 64, 64, 0.0)?;

    assert_eq!(mc, 1);
    check_two_triangles(&ml, &mv, &mt, &ibd);
    Ok(())
}

#[test]
fn meshlets_sparse() -> Result<()> {
    let vbd = [0.0f32; 16 * 3];
    let ibd = [0, 7, 15, 15, 7, 3];

    let mut ml = [Meshlet::default(); 1];
    let mut mv = [0u32; 4];
    let mut mt = [0u8; 8];
    let mc = build_meshlets(&mut ml, &mut mv, &mut mt, &ibd, 6, &vbd, 16, 4 * 3, 64, 64, 0.0)?;

    assert_eq!(mc, 1);
    check_two_triangles(&ml, &mv, &mt, &ibd);
    Ok(())
}

// two tetrahedrons far apart
#[rustfmt::skip]
const TETRA_VB: [f32; 2 * 4 * 3] = [
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    10.0, 0.0, 0.0, 11.0, 0.0, 0.0, 10.0, 1.0, 0.0, 10.0, 0.0, 1.0,
];

#[rustfmt::skip]
const TETRA_IB: [u32; 2 * 4 * 3] = [
    0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2,
    4, 5, 6, 4, 6, 7, 4, 7, 5, 5, 7, 6,
];

fn assert_meshlet(ml: &Meshlet, triangles: u32, vertices: u32) {
    assert_eq!(ml.triangle_count, triangles);
    assert_eq!(ml.vertex_count, vertices);
}

#[test]
fn meshlets_flex() -> Result<()> {
    let (vb, ib) = (TETRA_VB, TETRA_IB);

    // up to 2 meshlets with min_triangles=4
    assert_eq!(build_meshlets_bound(2 * 4 * 3, 16, 4)?, 2);

    let mut ml = [Meshlet::default(); 2];
    let mut mv = [0u32; 2 * 16];
    let mut mt = [0u8; 2 * 8 * 3]; // 2 meshlets with up to 8 triangles

    // with regular function, we should get one meshlet (maxt=8) or two (maxt=4)
    assert_eq!(
        build_meshlets(&mut ml, &mut mv, &mut mt, &ib, ib.len(), &vb, 8, 4 * 3, 16, 8, 0.0)?,
        1
    );
    assert_meshlet(&ml[0], 8, 8);

    assert_eq!(
        build_meshlets(&mut ml, &mut mv, &mut mt, &ib, ib.len(), &vb, 8, 4 * 3, 16, 4, 0.0)?,
        2
    );
    assert_meshlet(&ml[0], 4, 4);
    assert_meshlet(&ml[1], 4, 4);

    // with flex function and mint=4 maxt=8 we should get one meshlet if split_factor is zero, or large enough to accomodate both
    assert_eq!(
        build_meshlets_flex(&mut ml, &mut mv, &mut mt, &ib, ib.len(), &vb, 8, 4 * 3, 16, 4, 8, 0.0, 0.0)?,
        1
    );
    assert_meshlet(&ml[0], 8, 8);

    assert_eq!(
        build_meshlets_flex(&mut ml, &mut mv, &mut mt, &ib, ib.len(), &vb, 8, 4 * 3, 16, 4, 8, 0.0, 10.0)?,
        1
    );
    assert_meshlet(&ml[0], 8, 8);

    // however, with a smaller split factor we should get two meshlets
    assert_eq!(
        build_meshlets_flex(&mut ml, &mut mv, &mut mt, &ib, ib.len(), &vb, 8, 4 * 3, 16, 4, 8, 0.0, 1.0)?,
        2
    );
    assert_meshlet(&ml[0], 4, 4);
    assert_meshlet(&ml[1], 4, 4);
    Ok(())
}

#[test]
fn meshlets_spatial() -> Result<()> {
    let (vb, ib) = (TETRA_VB, TETRA_IB);

    // up to 2 meshlets with min_triangles=4
    assert_eq!(build_meshlets_bound(2 * 4 * 3, 16, 4)?, 2);

    let mut ml = [Meshlet::default(); 2];
    let mut mv = [0u32; 2 * 16];
    let mut mt = [0u8; 2 * 8 * 3]; // 2 meshlets with up to 8 triangles

    // with strict limits, we should get one meshlet (maxt=8) or two (maxt=4)
    assert_eq!(
        build_meshlets_spatial(&mut ml, &mut mv, &mut mt, &ib, ib.len(), &vb, 8, 4 * 3, 16, 8, 8, 0.0)?,
        1
    );
    assert_meshlet(&ml[0], 8, 8);

    assert_eq!(
        build_meshlets_spatial(&mut ml, &mut mv, &mut mt, &ib, ib.len(), &vb, 8, 4 * 3, 16, 4, 4, 0.0)?,
        2
    );
    assert_meshlet(&ml[0], 4, 4);
    assert_meshlet(&ml[1], 4, 4);

    // with maxv=4 we should get two meshlets since we can't accomodate both
    assert_eq!(
        build_meshlets_spatial(&mut ml, &mut mv, &mut mt, &ib, ib.len(), &vb, 8, 4 * 3, 4, 4, 8, 0.0)?,
        2
    );
    assert_meshlet(&ml[0], 4, 4);
    assert_meshlet(&ml[1], 4, 4);
    Ok(())
}

#[test]
fn meshlets_spatial_deep() -> Result<()> {
    const N: usize = 400;
    let max_vertices = 4;
    let max_triangles = 4;

    let mut vb = vec![0.0f32; (N + 1) * 3];
    let mut ib = vec![0u32; N * 3];

    for i in 0..N {
        let v = 1.2f32.powf(i as f32);
        vb[(i + 1) * 3..(i + 2) * 3].fill(v);

        ib[i * 3] = 0;
        ib[i * 3 + 1] = (i + 1) as u32;
        ib[i * 3 + 2] = (i + 1) as u32;
    }

    let max_meshlets = build_meshlets_bound(N * 3, max_vertices, max_triangles)?;
    let mut meshlets = vec![Meshlet::default(); max_meshlets];
    let mut meshlet_vertices = vec![0u32; N * 3];
    let mut meshlet_triangles = vec![0u8; N * 3];

    let result = build_meshlets_spatial(
        &mut meshlets,
        &mut meshlet_vertices,
        &mut meshlet_triangles,
        &ib,
        N * 3,
        &vb,
        N + 1,
        4 * 3,
        max_vertices,
        max_triangles,
        max_triangles,
        0.0,
    )?;
    assert_eq!(result, N);
    Ok(())
}

#[test]
fn meshlets_output_too_small() {
    let vbd = [0.0f32; 4 * 3];
    let ibd = [0, 2, 1, 1, 2, 3];

    let mut ml = [Meshlet::default(); 1];
    let mut mv = [0u32; 3];
    let mut mt = [0u8; 8];
    let r = build_meshlets(&mut ml, &mut mv, &mut mt, &ibd, 6, &vbd, 4, 4 * 3, 64, 64, 0.0);
    assert_eq!(r, Err(Error::BufferTooSmall("meshlet_vertices")));
}
