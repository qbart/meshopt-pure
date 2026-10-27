//! Port of optimizer tests from `demo/tests.cpp` (`emptyMesh`) plus sanity checks.

use meshoptimizer::*;

#[test]
fn empty_mesh() -> Result<()> {
    optimize_vertex_cache(&mut [], &[], 0, 0)?;
    optimize_vertex_cache_fifo(&mut [], &[], 0, 0, 16)?;
    optimize_overdraw(&mut [], &[], 0, &[], 0, 12, 1.0)?;
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
