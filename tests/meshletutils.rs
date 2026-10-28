//! Port of meshlet utility tests from `demo/tests.cpp`.

use meshoptimizer::*;

#[test]
fn cluster_bounds_degenerate() -> Result<()> {
    let vbd = [0.0f32; 9];
    let ibd = [0u32, 0, 0];
    let ib1 = [0u32, 1, 2];

    // all of the bounds below are degenerate as they use 0 triangles, one topology-degenerate triangle and one position-degenerate triangle respectively
    let bounds0 = compute_cluster_bounds::<u32>(&[], 0, &[], 0, 12)?;
    let boundsd = compute_cluster_bounds(&ibd, 3, &vbd, 3, 12)?;
    let bounds1 = compute_cluster_bounds(&ib1, 3, &vbd, 3, 12)?;

    for b in [bounds0, boundsd, bounds1] {
        assert!(b.center == [0.0; 3] && b.radius == 0.0);
    }

    let vb1 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let ib2 = [0u32, 1, 2, 0, 2, 1];

    // these bounds have a degenerate cone since the cluster has two triangles with opposite normals
    let bounds2 = compute_cluster_bounds(&ib2, 6, &vb1, 3, 12)?;

    assert_eq!(bounds2.cone_apex, [0.0; 3]);
    assert_eq!(bounds2.cone_axis, [0.0; 3]);
    assert_eq!(bounds2.cone_cutoff, 1.0);
    assert_eq!(bounds2.cone_axis_s8, [0; 3]);
    assert_eq!(bounds2.cone_cutoff_s8, 127);

    // however, the bounding sphere needs to be in tact (here we only check bbox for simplicity)
    for k in 0..3 {
        assert!(bounds2.center[k] - bounds2.radius <= 0.0 && bounds2.center[k] + bounds2.radius >= 1.0);
    }
    Ok(())
}

#[test]
fn sphere_bounds() -> Result<()> {
    #[rustfmt::skip]
    let vbr: [f32; 16] = [
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 2.0,
        1.0, 0.0, 1.0, 3.0,
    ];

    // without the radius, the center is inside the tetrahedron
    let bounds = compute_sphere_bounds(&vbr, 4, 4 * 4, None, 0)?;
    assert!((bounds.center[0] - 0.5).abs() < 1e-2);
    assert!((bounds.center[1] - 0.5).abs() < 1e-2);
    assert!((bounds.center[2] - 0.5).abs() < 1e-2);
    assert!(bounds.radius < 0.87);

    // when using the radius, the last sphere envelops the entire set
    let boundsr = compute_sphere_bounds(&vbr, 4, 4 * 4, Some(&vbr[3..]), 4 * 4)?;
    assert!((boundsr.center[0] - 1.0).abs() < 1e-2);
    assert!((boundsr.center[1] - 0.0).abs() < 1e-2);
    assert!((boundsr.center[2] - 1.0).abs() < 1e-2);
    assert!((boundsr.radius - 3.0).abs() < 1e-2);
    Ok(())
}

#[test]
fn extract_meshlet() -> Result<()> {
    // 15 and 1039 collide in low 10 bits
    let indices = [0, 7, 15, 1039, 7, 15];

    let mut vertices = [0u32; 4];
    let mut triangles = [0u8; 6];
    let unique = extract_meshlet_indices(&mut vertices, &mut triangles, &indices, 6)?;
    assert_eq!(unique, 4);

    // verify the invariant: vertices[triangles[i]] == indices[i]
    for i in 0..6 {
        assert_eq!(vertices[usize::from(triangles[i])], indices[i]);
    }

    assert_eq!(vertices, [0, 7, 15, 1039]);
    assert_eq!(triangles, [0, 1, 2, 3, 1, 2]);
    Ok(())
}

#[test]
fn meshlets_max() -> Result<()> {
    let mut vb = [0.0f32; 16 * 16 * 3];
    let mut ib = [0u32; 15 * 15 * 2 * 3];

    // 16x16 grid of vertices, 15x15 grid of triangles
    for y in 0..16 {
        for x in 0..16 {
            vb[(y * 16 + x) * 3] = x as f32;
            vb[(y * 16 + x) * 3 + 1] = y as f32;
            vb[(y * 16 + x) * 3 + 2] = 0.0;
        }
    }

    for y in 0..15u32 {
        for x in 0..15u32 {
            let o = ((y * 15 + x) * 2 * 3) as usize;
            ib[o..o + 6].copy_from_slice(&[
                y * 16 + x,
                y * 16 + (x + 1),
                (y + 1) * 16 + x,
                (y + 1) * 16 + x,
                y * 16 + (x + 1),
                (y + 1) * 16 + (x + 1),
            ]);
        }
    }

    let mut ml = [Meshlet::default(); 1];
    let mut mv = [0u32; 16 * 16];
    let mut mt = [0u8; 15 * 15 * 2 * 3 + 3];

    let mc = build_meshlets(&mut ml, &mut mv, &mut mt, &ib, ib.len(), &vb, 16 * 16, 4 * 3, 256, 512, 0.0)?;
    assert_eq!(mc, 1);
    assert_eq!(ml[0].triangle_count, 450);
    assert_eq!(ml[0].vertex_count, 256);

    optimize_meshlet(&mut mv, &mut mt, ml[0].triangle_count as usize, ml[0].vertex_count as usize)?;

    // check sequential ordering of remapped indices
    let mut vmax = -1i32;

    for &t in &mt[..450 * 3] {
        assert!(i32::from(t) <= vmax + 1);
        vmax = vmax.max(i32::from(t));
    }
    Ok(())
}
