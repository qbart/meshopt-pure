//! Port of simplifier tests from `demo/tests.cpp`.
//! C calls of the form `meshopt_simplify(ib, ib, ...)` alias destination and source; here the source is copied first.
#![allow(clippy::needless_range_loop)]

use meshoptimizer::*;

#[rustfmt::skip]
const STAIRS_IB: [u32; 12] = [
    0, 2, 1,
    1, 2, 3,
    3, 2, 4,
    2, 5, 4,
];

// 0
// 1 2
// 3 4 5
#[rustfmt::skip]
const STAIRS_VB: [f32; 18] = [
    0.0, 4.0, 0.0,
    0.0, 1.0, 0.0,
    2.0, 2.0, 0.0,
    0.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    4.0, 0.0, 0.0,
];

/// `meshopt_simplify(ib, ib, ...)` with default options/result_error.
fn simplify_in_place(
    ib: &mut [u32],
    index_count: usize,
    vb: &[f32],
    vertex_count: usize,
    stride: usize,
    target: usize,
    error: f32,
    options: u32,
    result_error: Option<&mut f32>,
) -> Result<usize> {
    let src = ib.to_vec();
    simplify(
        ib,
        &src,
        index_count,
        vb,
        vertex_count,
        stride,
        target,
        error,
        options,
        result_error,
    )
}

#[test]
fn simplify_basic() -> Result<()> {
    let mut ib = STAIRS_IB;
    let expected = [0, 5, 3];

    let mut error = 0.0;
    assert_eq!(simplify_in_place(&mut ib, 12, &STAIRS_VB, 6, 12, 3, 1e-2, 0, Some(&mut error))?, 3);
    assert!(error < 1e-4);
    assert_eq!(ib[..3], expected);
    Ok(())
}

#[test]
fn simplify_stuck() -> Result<()> {
    // tetrahedron can't be simplified due to collapse error restrictions
    let vb1 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut ib1 = [0, 1, 2, 0, 2, 3, 0, 3, 1, 2, 1, 3];

    assert_eq!(simplify_in_place(&mut ib1, 12, &vb1, 4, 12, 6, 1e-3, 0, None)?, 12);

    // 5-vertex strip can't be simplified due to topology restriction since middle triangle has flipped winding
    let vb2 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.5, 1.0, 0.0];
    let mut ib2 = [0, 1, 3, 3, 1, 4, 1, 2, 4]; // ok
    let mut ib3 = [0, 1, 3, 1, 3, 4, 1, 2, 4]; // flipped

    assert_eq!(simplify_in_place(&mut ib2, 9, &vb2, 5, 12, 6, 1e-3, 0, None)?, 6);
    assert_eq!(simplify_in_place(&mut ib3, 9, &vb2, 5, 12, 6, 1e-3, 0, None)?, 9);

    // 4-vertex quad with a locked corner can't be simplified due to border error-induced restriction
    let vb4 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let mut ib4 = [0, 1, 3, 0, 3, 2];

    assert_eq!(simplify_in_place(&mut ib4, 6, &vb4, 4, 12, 3, 1e-3, 0, None)?, 6);

    // 4-vertex quad with a locked corner can't be simplified due to border error-induced restriction
    let vb5 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let mut ib5 = [0, 1, 4, 0, 3, 2];

    assert_eq!(simplify_in_place(&mut ib5, 6, &vb5, 5, 12, 3, 1e-3, 0, None)?, 6);
    Ok(())
}

#[test]
fn simplify_sloppy_stuck() -> Result<()> {
    let vb = [0.0f32; 9];
    let ib = [0, 1, 2, 0, 1, 2];

    // simplifying down to 0 triangles results in 0 immediately
    assert_eq!(simplify_sloppy(&mut [], &ib, 3, &vb, 3, 12, None, 0, 0.0, None)?, 0);

    // simplifying down to 2 triangles given that all triangles are degenerate results in 0 as well
    assert_eq!(simplify_sloppy(&mut [], &ib, 6, &vb, 3, 12, None, 6, 0.0, None)?, 0);
    Ok(())
}

#[test]
fn simplify_sloppy_locks() -> Result<()> {
    let mut ib = STAIRS_IB;

    // lock spine
    let locks = [1u8, 0, 1, 0, 0, 1];

    let expected = [0, 2, 1, 1, 2, 5];

    let mut error = 0.0;
    let src = ib;
    assert_eq!(
        simplify_sloppy(&mut ib, &src, 12, &STAIRS_VB, 6, 12, Some(&locks), 3, 1.0, Some(&mut error))?,
        6
    );
    assert_eq!(error, 0.0);
    assert_eq!(ib[..6], expected);
    Ok(())
}

#[test]
fn simplify_points_stuck() -> Result<()> {
    let vb = [0.0f32; 9];

    // simplifying down to 0 points results in 0 immediately
    assert_eq!(simplify_points(&mut [], &vb, 3, 12, None, 0, 0.0, 0)?, 0);
    Ok(())
}

#[test]
fn simplify_flip() -> Result<()> {
    // this mesh has been constructed by taking a tessellated irregular grid with a square cutout
    // and progressively collapsing edges until the only ones left violate border or flip constraints.
    // there is only one valid non-flip collapse, so we validate that we take it; when flips are allowed,
    // the wrong collapse is picked instead.
    #[rustfmt::skip]
    let vb = [
        1.000000, 1.000000, -1.000000,
        1.000000, 1.000000, 1.000000,
        1.000000, -1.000000, 1.000000,
        1.000000, -0.200000, -0.200000,
        1.000000, 0.200000, -0.200000,
        1.000000, -0.200000, 0.200000,
        1.000000, 0.200000, 0.200000,
        1.000000, 0.500000, -0.500000,
        1.000000, -1.000000, 0.000000,
    ];

    // the collapse we expect is 7 -> 0
    #[rustfmt::skip]
    let mut ib = [
        7, 4, 3,
        1, 2, 5,
        7, 1, 6,
        // gets removed
        7, 8, 0,
        7, 6, 4,
        8, 5, 2,
        8, 7, 3,
        8, 3, 5,
        5, 6, 1,
        // gets removed
        7, 0, 1,
    ];

    #[rustfmt::skip]
    let expected = [
        0, 4, 3,
        1, 2, 5,
        0, 1, 6,
        0, 6, 4,
        8, 5, 2,
        8, 0, 3,
        8, 3, 5,
        5, 6, 1,
    ];

    assert_eq!(simplify_in_place(&mut ib, 30, &vb, 9, 12, 3, 1e-3, 0, None)?, 24);
    assert_eq!(ib[..24], expected);
    Ok(())
}

#[test]
fn simplify_scale_test() -> Result<()> {
    let vb = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];

    assert_eq!(simplify_scale(&vb, 4, 12)?, 3.0);
    Ok(())
}

#[test]
fn simplify_degenerate() -> Result<()> {
    #[rustfmt::skip]
    let vb = [
        0.000000, 0.000000, 0.000000,
        0.000000, 1.000000, 0.000000,
        0.000000, 2.000000, 0.000000,
        1.000000, 0.000000, 0.000000,
        2.000000, 0.000000, 0.000000,
        1.000000, 1.000000, 0.000000,
    ];

    // 0 1 2
    // 3 5
    // 4

    #[rustfmt::skip]
    let mut ib = [
        0, 1, 3,
        3, 1, 5,
        1, 2, 5,
        3, 5, 4,
        // these two degenerate triangles create a fake reverse edge
        1, 0, 1,
        // which breaks border classification
        0, 3, 0,
    ];

    let expected = [0, 1, 4, 4, 1, 2];

    assert_eq!(simplify_in_place(&mut ib, 18, &vb, 6, 12, 3, 1e-3, 0, None)?, 6);
    assert_eq!(ib[..6], expected);
    Ok(())
}

// 0 1 2
// 3 4 5
// 6 7 8
#[rustfmt::skip]
const GRID_VB: [f32; 27] = [
    0.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 2.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 1.0, 0.0,
    1.0, 2.0, 0.0,
    2.0, 0.0, 0.0,
    2.0, 1.0, 0.0,
    2.0, 2.0, 0.0,
];

#[rustfmt::skip]
const GRID_IB: [u32; 24] = [
    0, 1, 3,
    3, 1, 4,
    1, 2, 4,
    4, 2, 5,
    3, 4, 6,
    6, 4, 7,
    4, 5, 7,
    7, 5, 8,
];

#[rustfmt::skip]
const GRID_EXPECTED: [u32; 18] = [
    0, 1, 3,
    1, 2, 3,
    3, 2, 5,
    6, 3, 7,
    3, 5, 7,
    7, 5, 8,
];

#[test]
fn simplify_lock_border() -> Result<()> {
    let mut ib = GRID_IB;

    assert_eq!(
        simplify_in_place(&mut ib, 24, &GRID_VB, 9, 12, 3, 1e-3, SIMPLIFY_LOCK_BORDER, None)?,
        18
    );
    assert_eq!(ib[..18], GRID_EXPECTED);
    Ok(())
}

fn simplify_attr(skip_g: bool) -> Result<()> {
    let mut vb = [[0.0f32; 6]; 8 * 3];

    for y in 0..8 {
        // first four rows are a blue gradient, next four rows are a yellow gradient
        let r = if y < 4 { 0.8 + y as f32 * 0.05 } else { 0.0 };
        let g = if y < 4 { 0.8 + y as f32 * 0.05 } else { 0.0 };
        let b = if y < 4 { 0.0 } else { 0.8 + (7 - y) as f32 * 0.05 };

        for x in 0..3 {
            let v = &mut vb[y * 3 + x];
            v[0] = x as f32;
            v[1] = y as f32;
            v[2] = 0.03 * x as f32 + 0.03 * (y % 2) as f32 + f32::from(u8::from(x == 2 && y == 7)) * 0.03;
            v[3] = r;
            v[4] = g;
            v[5] = b;
        }
    }

    let mut ib = [[0u32; 6]; 7 * 2];

    for y in 0..7u32 {
        for x in 0..2u32 {
            ib[(y * 2 + x) as usize] = [
                y * 3 + x,
                y * 3 + (x + 1),
                (y + 1) * 3 + x,
                (y + 1) * 3 + x,
                y * 3 + (x + 1),
                (y + 1) * 3 + (x + 1),
            ];
        }
    }

    let attr_weights = [0.5, if skip_g { 0.0 } else { 0.5 }, 0.5];

    // *0  1   *2
    //  3  4    5
    //  6  7    8
    // *9  10 *11
    // *12 13 *14
    //  15 16  17
    //  18 19  20
    // *21 22 *23
    let expected: [[u32; 6]; 3] = [[0, 2, 11, 0, 11, 9], [9, 11, 12, 12, 11, 14], [12, 14, 23, 12, 23, 21]];

    let vbf = vb.as_flattened();
    let src = ib.as_flattened().to_vec();
    let n = simplify_with_attributes(
        ib.as_flattened_mut(),
        &src,
        7 * 2 * 6,
        vbf,
        8 * 3,
        6 * 4,
        &vbf[3..],
        6 * 4,
        &attr_weights,
        3,
        None,
        6 * 3,
        1e-2,
        0,
        None,
    )?;
    assert_eq!(n, 18);
    assert_eq!(ib[..3], expected);
    Ok(())
}

#[test]
fn simplify_attr_all() -> Result<()> {
    simplify_attr(false)
}

#[test]
fn simplify_attr_skip_g() -> Result<()> {
    simplify_attr(true)
}

#[test]
fn simplify_lock_flags() -> Result<()> {
    #[rustfmt::skip]
    let lock = [
        1u8, 1, 1,
        1, 0, 1,
        1, 1, 1,
    ];

    let mut ib = GRID_IB;
    let src = ib;

    assert_eq!(
        simplify_with_attributes(&mut ib, &src, 24, &GRID_VB, 9, 12, &[], 0, &[], 0, Some(&lock), 3, 1e-3, 0, None)?,
        18
    );
    assert_eq!(ib[..18], GRID_EXPECTED);
    Ok(())
}

#[test]
fn simplify_lock_flags_seam() -> Result<()> {
    #[rustfmt::skip]
    let vb = [
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 2.0, 0.0,
        1.0, 0.0, 0.0,
        1.0, 1.0, 0.0,
        1.0, 1.0, 0.0,
        1.0, 2.0, 0.0,
        2.0, 0.0, 0.0,
        2.0, 1.0, 0.0,
        2.0, 1.0, 0.0,
        2.0, 2.0, 0.0,
    ];

    let lock0 = [1u8, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1];
    let lock1 = [1u8, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1];
    let lock2 = [1u8, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1];
    let lock3 = [1u8, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1];

    // 0 1-2 3
    // 4 5-6 7
    // 8 9-10 11

    #[rustfmt::skip]
    let ib = [
        0, 1, 4,
        4, 1, 5,
        4, 5, 8,
        8, 5, 9,
        2, 3, 6,
        6, 3, 7,
        6, 7, 10,
        10, 7, 11,
    ];

    let mut res = [0u32; 24];
    let mut run = |lock: Option<&[u8]>| simplify_with_attributes(&mut res, &ib, 24, &vb, 12, 12, &[], 0, &[], 0, lock, 0, 1.0, 0, None);

    // with no locks, we should be able to collapse the entire mesh (vertices 1-2 and 9-10 are locked but others can move towards them)
    assert_eq!(run(None)?, 0);

    // with corners locked, we should get two quads
    assert_eq!(run(Some(&lock0))?, 12);

    // with both sides locked, we can only collapse the seam spine
    assert_eq!(run(Some(&lock1))?, 18);

    // with seam spine locked, we can collapse nothing; note that we intentionally test two different lock configurations
    // they each lock only one side of the seam spine, which should be equivalent
    assert_eq!(run(Some(&lock2))?, 24);
    assert_eq!(run(Some(&lock3))?, 24);
    Ok(())
}

#[test]
fn simplify_sparse() -> Result<()> {
    #[rustfmt::skip]
    let vb = [
        0.0, 0.0, 100.0,
        0.0, 1.0, 0.0,
        0.0, 2.0, 100.0,
        1.0, 0.0, 0.1,
        1.0, 1.0, 0.1,
        1.0, 2.0, 0.1,
        2.0, 0.0, 100.0,
        2.0, 1.0, 0.0,
        2.0, 2.0, 100.0,
    ];

    let vba = [100.0, 0.5, 100.0, 0.5, 0.5, 0.0, 100.0, 0.5, 100.0];

    let aw = [0.5];

    #[rustfmt::skip]
    let lock = [
        8u8, 1, 8,
        1, 0, 1,
        8, 1, 8,
    ];

    //   1
    // 3 4 5
    //   7

    #[rustfmt::skip]
    let mut ib = [
        3, 1, 4,
        1, 5, 4,
        3, 4, 7,
        4, 5, 7,
    ];

    let mut res = [0u32; 12];

    // vertices 3-4-5 are slightly elevated along Z which guides the collapses when only using geometry
    let expected = [1, 5, 3, 3, 5, 7];

    assert_eq!(simplify(&mut res, &ib, 12, &vb, 9, 12, 6, 1e-3, SIMPLIFY_SPARSE, None)?, 6);
    assert_eq!(res[..6], expected);

    // vertices 1-4-7 have a crease in the attribute value which guides the collapses the opposite way when weighing attributes sufficiently
    let expecteda = [3, 1, 7, 1, 5, 7];

    assert_eq!(
        simplify_with_attributes(
            &mut res,
            &ib,
            12,
            &vb,
            9,
            12,
            &vba,
            4,
            &aw,
            1,
            Some(&lock),
            6,
            1e-1,
            SIMPLIFY_SPARSE,
            None
        )?,
        6
    );
    assert_eq!(res[..6], expecteda);

    // a final test validates that destination can alias when using sparsity
    assert_eq!(simplify_in_place(&mut ib, 12, &vb, 9, 12, 6, 1e-3, SIMPLIFY_SPARSE, None)?, 6);
    assert_eq!(ib[..6], expected);
    Ok(())
}

#[test]
fn simplify_error_absolute() -> Result<()> {
    let mut vb = GRID_VB;
    vb[4 * 3 + 2] = 1.0; // vertex 4 is elevated

    let mut ib = GRID_IB;

    let mut error = 0.0;
    assert_eq!(
        simplify_in_place(
            &mut ib,
            24,
            &vb,
            9,
            12,
            18,
            2.0,
            SIMPLIFY_LOCK_BORDER | SIMPLIFY_ERROR_ABSOLUTE,
            Some(&mut error)
        )?,
        18
    );
    assert!((error - 0.85).abs() < 0.01);
    Ok(())
}

#[test]
fn simplify_seam() -> Result<()> {
    // xyz+attr
    #[rustfmt::skip]
    let vb = [
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 2.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 0.3, 0.0,
        1.0, 1.0, 0.3, 1.0,
        1.0, 2.0, 0.0, 1.0,
        2.0, 0.0, 0.0, 0.0,
        2.0, 1.0, 0.1, 0.0,
        2.0, 1.0, 0.1, 1.0,
        2.0, 2.0, 0.0, 1.0,
        3.0, 0.0, 0.0, 0.0,
        3.0, 1.0, 0.0, 0.0,
        3.0, 1.0, 0.0, 1.0,
        3.0, 2.0, 0.0, 1.0,
    ];

    // 0   1-2   3
    // 4   5-6   7
    // 8   9-10 11
    // 12 13-14 15

    #[rustfmt::skip]
    let ib = [
        0, 1, 4,
        4, 1, 5,
        2, 3, 6,
        6, 3, 7,
        4, 5, 8,
        8, 5, 9,
        6, 7, 10,
        10, 7, 11,
        8, 9, 12,
        12, 9, 13,
        10, 11, 14,
        14, 11, 15,
    ];

    // note: vertices 1-2 and 13-14 are classified as locked, because they are on a seam & a border
    // 0   1-2   3
    //     5-6
    //     9-10
    // 12 13-14 15
    #[rustfmt::skip]
    let expected = [
        0, 1, 13,
        2, 3, 14,
        0, 13, 12,
        14, 3, 15,
    ];

    let mut res = [0u32; 36];
    let mut error = 0.0;

    assert_eq!(simplify(&mut res, &ib, 36, &vb, 16, 16, 12, 1.0, 0, Some(&mut error))?, 12);
    assert_eq!(res[..12], expected);
    assert!((error - 0.1).abs() < 0.01); // note: the error is not zero because there is a difference in height between the seam vertices

    let aw = [1.0];
    assert_eq!(
        simplify_with_attributes(
            &mut res,
            &ib,
            36,
            &vb,
            16,
            16,
            &vb[3..],
            16,
            &aw,
            1,
            None,
            12,
            2.0,
            0,
            Some(&mut error)
        )?,
        12
    );
    assert_eq!(res[..12], expected);
    assert!((error - 0.1).abs() < 0.01); // note: this is the same error as above because the attribute is constant on either side of the seam
    Ok(())
}

#[test]
fn simplify_seam_fake() -> Result<()> {
    // xyz+attr
    #[rustfmt::skip]
    let vb = [
        0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 2.0,
        0.0, 0.0, 0.0, 3.0,
    ];

    let mut ib = [0, 1, 2, 2, 1, 3];

    assert_eq!(simplify_in_place(&mut ib, 6, &vb, 4, 16, 0, 1.0, 0, None)?, 6);
    Ok(())
}

#[test]
fn simplify_seam_attr() -> Result<()> {
    // xyz+attr
    #[rustfmt::skip]
    let vb = [
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 1.0,
        1.0, 1.0, 0.0, 1.0,
        1.0, 1.0, 0.0, 1.0,
        1.0, 2.0, 0.0, 1.0,
        4.0, 0.0, 0.0, 2.0,
        4.0, 1.0, 0.0, 2.0,
        4.0, 1.0, 0.0, 2.0,
        4.0, 2.0, 0.0, 2.0,
    ];

    // 0   1-2   3
    // 4   5-6   7
    // 8   9-10 11

    #[rustfmt::skip]
    let ib = [
        0, 1, 4,
        4, 1, 5,
        2, 3, 6,
        6, 3, 7,
        4, 5, 8,
        8, 5, 9,
        6, 7, 10,
        10, 7, 11,
    ];

    // note: vertices 1-2 and 9-10 are classified as locked, because they are on a seam & a border
    // 0   1-2   3
    // 4         7
    // 8   9-10 11
    #[rustfmt::skip]
    let expected = [
        0, 1, 4,
        2, 3, 7,
        4, 1, 8,
        8, 1, 9,
        2, 7, 10,
        10, 7, 11,
    ];

    let mut res = [0u32; 24];
    let mut error = 0.0;

    let aw = [1.0];
    assert_eq!(
        simplify_with_attributes(
            &mut res,
            &ib,
            24,
            &vb,
            12,
            16,
            &vb[3..],
            16,
            &aw,
            1,
            None,
            12,
            2.0,
            SIMPLIFY_LOCK_BORDER,
            Some(&mut error)
        )?,
        18
    );
    assert_eq!(res[..18], expected);
    assert!((error - 0.35).abs() < 0.01);
    Ok(())
}

#[test]
fn simplify_debug() -> Result<()> {
    let mut ib = STAIRS_IB;

    let expected = [9u32 << 28, 5 | (9u32 << 28), 3 | (9u32 << 28)];

    let mut error = 0.0;
    assert_eq!(
        simplify_in_place(&mut ib, 12, &STAIRS_VB, 6, 12, 3, 1e-2, SIMPLIFY_INTERNAL_DEBUG, Some(&mut error))?,
        3
    );
    assert!(error < 1e-4);
    assert_eq!(ib[..3], expected);
    Ok(())
}

#[test]
fn simplify_prune_test() -> Result<()> {
    // 0
    // 1 2
    // 3 4 5
    // +
    // 6 7 8 (same position)
    #[rustfmt::skip]
    let mut ib = [
        3, 2, 4,
        0, 2, 1,
        1, 2, 3,
        2, 5, 4,
        6, 7, 8,
    ];

    let mut vb = [1.0f32; 27];
    vb[..18].copy_from_slice(&STAIRS_VB);

    let expected = [0, 5, 3];

    let mut error = 0.0;
    assert_eq!(
        simplify_in_place(&mut ib, 15, &vb, 9, 12, 3, 1e-2, SIMPLIFY_PRUNE, Some(&mut error))?,
        3
    );
    assert!(error < 1e-4);
    assert_eq!(ib[..3], expected);

    // re-run prune with and without sparsity on a small subset to make sure the component code correctly handles sparse subsets
    assert_eq!(
        simplify_in_place(&mut ib, 3, &vb, 9, 12, 3, 1e-2, SIMPLIFY_PRUNE, Some(&mut error))?,
        3
    );
    assert_eq!(
        simplify_in_place(&mut ib, 3, &vb, 9, 12, 3, 1e-2, SIMPLIFY_PRUNE | SIMPLIFY_SPARSE, Some(&mut error))?,
        3
    );
    assert_eq!(ib[..3], expected);
    Ok(())
}

#[rustfmt::skip]
const PRUNE_VB: [f32; 27] = [
    0.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 0.0, 1.0,
    0.0, 2.0, 1.0,
    2.0, 0.0, 1.0,
    0.0, 0.0, 2.0,
    0.0, 4.0, 2.0,
    4.0, 0.0, 2.0,
];

#[test]
fn simplify_prune_cleanup() -> Result<()> {
    let mut ib = [0, 1, 2, 3, 4, 5, 6, 7, 8];

    let expected = [6, 7, 8];

    let mut error = 0.0;
    assert_eq!(
        simplify_in_place(
            &mut ib,
            9,
            &PRUNE_VB,
            9,
            12,
            3,
            1.0,
            SIMPLIFY_LOCK_BORDER | SIMPLIFY_PRUNE,
            Some(&mut error)
        )?,
        3
    );
    assert!((error - 0.37).abs() < 0.01);
    assert_eq!(ib[..3], expected);
    Ok(())
}

#[test]
fn simplify_prune_func() -> Result<()> {
    let mut ib = [0, 1, 2, 3, 4, 5, 6, 7, 8];

    let expected = [6, 7, 8];

    let src = ib;
    assert_eq!(simplify_prune(&mut ib, &src, 9, &PRUNE_VB, 9, 12, 0.5)?, 3);
    assert_eq!(ib[..3], expected);
    Ok(())
}

#[rustfmt::skip]
const UPDATE_VB: [[f32; 4]; 5] = [
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
    [2.0, 0.0, 0.0, 0.0],
    [0.9, 0.2, 0.1, 0.2],
    [1.1, 0.2, 0.1, 0.1],
];

//     1
//    3 4
// 0       2
const UPDATE_IB: [u32; 15] = [0, 1, 3, 3, 1, 4, 4, 1, 2, 0, 3, 2, 3, 4, 2];

#[test]
fn simplify_update() -> Result<()> {
    let mut vb = UPDATE_VB;
    let mut ib = UPDATE_IB;

    let attr_weight = [1.0];

    assert_eq!(
        simplify_with_update_interleaved(
            &mut ib,
            15,
            vb.as_flattened_mut(),
            5,
            4 * 4,
            3,
            4 * 4,
            &attr_weight,
            1,
            None,
            9,
            1.0,
            0,
            None
        )?,
        9
    );

    let expected = [0, 1, 3, 3, 1, 2, 0, 3, 2];

    assert_eq!(ib[..9], expected);

    // border vertices haven't moved but may have small floating point drift
    for i in 0..3 {
        assert!(vb[i][3].abs() < 1e-6);
    }

    // center vertex got updated
    assert!((vb[3][0] - 0.88).abs() < 1e-2);
    assert!((vb[3][1] - 0.19).abs() < 1e-2);
    assert!((vb[3][2] - 0.11).abs() < 1e-2);
    assert!((vb[3][3] - 0.18).abs() < 1e-2);
    Ok(())
}

fn simplify_update_locked(options: u32) -> Result<()> {
    let mut vb = UPDATE_VB;
    let mut ib = UPDATE_IB;

    let attr_weight = [1.0];

    let vertex_lock = [0u8, 0, 0, 1, 0];

    assert_eq!(
        simplify_with_update_interleaved(
            &mut ib,
            15,
            vb.as_flattened_mut(),
            5,
            4 * 4,
            3,
            4 * 4,
            &attr_weight,
            1,
            Some(&vertex_lock),
            9,
            1.0,
            options,
            None
        )?,
        9
    );

    let expected = [0, 1, 3, 3, 1, 2, 0, 3, 2];

    assert_eq!(ib[..9], expected);

    for i in 0..3 {
        assert!(vb[i][3].abs() < 1e-6);
    }

    // locking guarantees exact result
    assert_eq!(vb[3], [0.9, 0.2, 0.1, 0.2]);
    Ok(())
}

#[test]
fn simplify_update_locked_default() -> Result<()> {
    simplify_update_locked(0)
}

#[test]
fn simplify_update_locked_sparse() -> Result<()> {
    simplify_update_locked(SIMPLIFY_SPARSE)
}

#[test]
fn simplify_folds() -> Result<()> {
    let vb = [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0,
    ];

    // 0 1 2
    // 3 4 5 + flipped versions
    #[rustfmt::skip]
    let ib = [
        0, 1, 3, 3, 1, 4, 1, 2, 4, 4, 2, 5,
        0, 3, 1, 3, 4, 1, 1, 4, 2, 4, 5, 2,
    ];

    let mut result = [0u32; 24];

    // without fold preservation, the errors for all collapses are close to zero because there are no open edges to prevent this
    assert_eq!(simplify(&mut result, &ib, 24, &vb, 6, 12, 0, 1e-3, 0, None)?, 0);
    assert_eq!(
        simplify(&mut result, &ib, 24, &vb, 6, 12, 0, 1e-3, SIMPLIFY_PRESERVE_FOLDS, None)?,
        12
    );

    let expected = [0, 2, 3, 3, 2, 5, 0, 3, 2, 3, 5, 2];

    assert_eq!(result[..12], expected);
    Ok(())
}
