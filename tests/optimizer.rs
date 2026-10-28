//! Port of optimizer tests from `demo/tests.cpp` (`emptyMesh`) plus sanity checks.

use meshoptimizer::*;

#[test]
fn empty_mesh() -> Result<()> {
    optimize_vertex_cache::<u32>(&mut [], &[], 0, 0)?;
    optimize_vertex_cache_fifo::<u32>(&mut [], &[], 0, 0, 16)?;
    optimize_overdraw::<u32>(&mut [], &[], 0, &[], 0, 12, 1.0)?;
    Ok(())
}

#[test]
fn vertex_cache_reorders_for_locality() -> Result<()> {
    // 3x1 strip of quads with triangles in scrambled order
    let ib = [2, 6, 3, 0, 4, 1, 3, 6, 7, 1, 4, 5, 1, 5, 2, 2, 5, 6];
    let before = analyze_vertex_cache(&ib, 18, 8, 3, 0, 0)?;

    for optimize in [optimize_vertex_cache, optimize_vertex_cache_strip] {
        let mut out = [0u32; 18];
        optimize(&mut out, &ib, 18, 8)?;
        assert!(analyze_vertex_cache(&out, 18, 8, 3, 0, 0)?.vertices_transformed <= before.vertices_transformed);
    }

    let mut inplace = ib;
    optimize_vertex_cache_in_place(&mut inplace, 18, 8)?;
    let mut fresh = [0u32; 18];
    optimize_vertex_cache(&mut fresh, &ib, 18, 8)?;
    assert_eq!(inplace, fresh);

    let mut out = [0u32; 18];
    optimize_vertex_cache_fifo(&mut out, &ib, 18, 8, 3)?;
    assert!(analyze_vertex_cache(&out, 18, 8, 3, 0, 0)?.vertices_transformed <= before.vertices_transformed);

    assert_eq!(optimize_vertex_cache(&mut out, &ib, 18, 7), Err(Error::IndexOutOfRange));
    assert!(optimize_vertex_cache_fifo(&mut out, &ib, 18, 8, 2).is_err());
    Ok(())
}

#[test]
fn stripify_roundtrip() -> Result<()> {
    // 2x1 quad grid
    let ib = [0, 1, 3, 3, 1, 4, 1, 2, 4, 4, 2, 5];

    for restart in [0u32, !0u32] {
        let mut strip = vec![0u32; stripify_bound(12)?];
        let n = stripify(&mut strip, &ib, 12, 6, restart)?;
        assert!(n < 12);

        let mut list = vec![0u32; unstripify_bound(n)?];
        let m = unstripify(&mut list, &strip, n, restart)?;
        assert_eq!(m, 12);

        let mut sorted: Vec<[u32; 3]> = list[..m].as_chunks::<3>().0.iter().map(|t| canonical(*t)).collect();
        let mut expected: Vec<[u32; 3]> = ib.as_chunks::<3>().0.iter().map(|t| canonical(*t)).collect();
        sorted.sort();
        expected.sort();
        assert_eq!(sorted, expected);
    }

    assert!(stripify_bound(4).is_err());
    assert!(unstripify_bound(2).is_err());
    Ok(())
}

/// Rotates a triangle so the smallest index comes first, preserving winding.
fn canonical(t: [u32; 3]) -> [u32; 3] {
    let m = (0..3).min_by_key(|&i| t[i]).unwrap_or(0);
    [t[m], t[(m + 1) % 3], t[(m + 2) % 3]]
}
