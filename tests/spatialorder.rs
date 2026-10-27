//! Sanity tests for spatial ordering (no 1:1 counterpart in `demo/tests.cpp`).

use meshoptimizer::*;

#[test]
fn spatial_sort_is_permutation() -> Result<()> {
    // points on a line in reverse order
    let vb: Vec<f32> = (0..8).rev().flat_map(|i| [i as f32, 0.0, 0.0]).collect();

    let mut remap = [0u32; 8];
    spatial_sort_remap(&mut remap, &vb, 8, 12)?;
    assert_eq!(remap, [7, 6, 5, 4, 3, 2, 1, 0]);

    let mut clusters = [0u32; 8];
    spatial_cluster_points(&mut clusters, &vb, 8, 12, 4)?;
    let mut first: Vec<u32> = clusters[..4].to_vec();
    first.sort();
    assert!(first == [0, 1, 2, 3] || first == [4, 5, 6, 7]);

    assert!(spatial_cluster_points(&mut clusters, &vb, 8, 12, 0).is_err());
    Ok(())
}

#[test]
fn spatial_sort_triangles_in_place_matches() -> Result<()> {
    let vb = [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 9.0, 9.0, 9.0, 10.0, 9.0, 9.0, 9.0, 10.0, 9.0,
    ];
    let ib = [3, 4, 5, 0, 1, 2];

    let mut out = [0u32; 6];
    spatial_sort_triangles(&mut out, &ib, 6, &vb, 6, 12)?;
    assert_eq!(out, [0, 1, 2, 3, 4, 5]);

    let mut inplace = ib;
    spatial_sort_triangles_in_place(&mut inplace, 6, &vb, 6, 12)?;
    assert_eq!(inplace, out);
    Ok(())
}
