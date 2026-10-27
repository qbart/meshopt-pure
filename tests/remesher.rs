//! Sanity tests for the remesher (no 1:1 counterpart in `demo/tests.cpp`).

use meshoptimizer::*;

/// Unit cube made of 12 triangles.
fn cube() -> (Vec<f32>, Vec<u32>) {
    let vb: Vec<f32> = (0..8)
        .flat_map(|i| [(i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32])
        .collect();
    #[rustfmt::skip]
    let ib = vec![
        0, 2, 1, 1, 2, 3, // z=0
        4, 5, 6, 5, 7, 6, // z=1
        0, 1, 4, 1, 5, 4, // y=0
        2, 6, 3, 3, 6, 7, // y=1
        0, 4, 2, 2, 4, 6, // x=0
        1, 3, 5, 3, 7, 5, // x=1
    ];
    (vb, ib)
}

#[test]
fn remesh_cube() -> Result<()> {
    let (vb, ib) = cube();

    for options in [0, REMESH_SHELL, REMESH_SOLVE] {
        let bound = remesh(None, 0, &ib, ib.len(), &vb, 8, 12, 16, options)?;
        assert!(bound > 0);

        let mut out = vec![0.0f32; bound * 9];
        let count = remesh(Some(&mut out), bound, &ib, ib.len(), &vb, 8, 12, 16, options)?;
        assert!(count <= bound);

        // all output vertices stay close to the unit cube
        for &v in &out[..count * 9] {
            assert!((-0.2..=1.2).contains(&v));
        }
    }

    assert!(remesh(None, 0, &ib, ib.len(), &vb, 8, 12, 3, 0).is_err());
    assert!(remesh(None, 1, &ib, ib.len(), &vb, 8, 12, 16, 0).is_err());
    Ok(())
}
