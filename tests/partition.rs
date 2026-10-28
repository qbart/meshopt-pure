//! Port of cluster partition tests from `demo/tests.cpp`.

use meshoptimizer::*;

#[test]
fn partition_basic() -> Result<()> {
    // 0   1   2
    //     3
    // 4 5 6 7 8
    //     9
    // 10 11  12
    #[rustfmt::skip]
    let ci = [
        0u32, 1, 3, 4, 5, 6,
        1, 2, 3, 6, 7, 8,
        4, 5, 6, 9, 10, 11,
        6, 7, 8, 9, 11, 12,
    ];

    let cc = [6, 6, 6, 6];
    let mut part = [0u32; 4];

    assert_eq!(partition_clusters(&mut part, &ci, ci.len(), &cc, 4, None, 13, 0, 1)?, 4);
    assert_eq!(part, [0, 1, 2, 3]);

    assert_eq!(partition_clusters(&mut part, &ci, ci.len(), &cc, 4, None, 13, 0, 2)?, 2);
    assert_eq!(part, [0, 0, 1, 1]);

    assert_eq!(partition_clusters(&mut part, &ci, ci.len(), &cc, 4, None, 13, 0, 4)?, 1);
    assert_eq!(part, [0, 0, 0, 0]);
    Ok(())
}

#[test]
fn partition_spatial() -> Result<()> {
    #[rustfmt::skip]
    let ci = [
        0u32, 1, 2,
        0, 3, 4,
        0, 5, 6,
    ];

    #[rustfmt::skip]
    let vb = [
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 2.0, 0.0, 2.0, 0.0, 0.0,
        -1.0, 0.0, 0.0, 0.0, -1.0, 0.0,
    ];

    let cc = [3, 3, 3];
    let mut part = [0u32; 3];

    assert_eq!(partition_clusters(&mut part, &ci, ci.len(), &cc, 3, None, 7, 0, 2)?, 2);
    assert_eq!(part, [0, 0, 1]);

    assert_eq!(partition_clusters(&mut part, &ci, ci.len(), &cc, 3, Some(&vb), 7, 4 * 3, 2)?, 2);
    assert_eq!(part, [0, 1, 0]);
    Ok(())
}

#[test]
fn partition_spatial_merge() -> Result<()> {
    #[rustfmt::skip]
    let ci = [
        0u32, 1, 2,
        3, 4, 5,
        6, 7, 8,
    ];

    #[rustfmt::skip]
    let vb = [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0,
        10.0, 0.0, 0.0, 10.0, 1.0, 0.0, 10.0, 2.0, 0.0,
    ];

    let cc = [3, 3, 3];
    let mut part = [0u32; 3];

    assert_eq!(partition_clusters(&mut part, &ci, ci.len(), &cc, 3, None, 9, 0, 2)?, 3);
    assert_eq!(part, [0, 1, 2]);

    assert_eq!(partition_clusters(&mut part, &ci, ci.len(), &cc, 3, Some(&vb), 9, 4 * 3, 2)?, 2);
    assert_eq!(part, [0, 0, 1]);
    Ok(())
}
