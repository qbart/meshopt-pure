//! `tools/clusterfuzz.cpp`: meshlet builders and cluster partitioning on random topology over a 100 vertex grid.

use meshoptimizer::{Meshlet, build_meshlets, build_meshlets_flex, build_meshlets_scan, build_meshlets_spatial, partition_clusters};

use crate::simplifyfuzz::{make_indices, make_vertices};

pub fn test_one_input(buffer: &[u8]) {
    if buffer.is_empty() {
        return;
    }

    let vb = make_vertices(buffer[0]);
    let (ib, indices) = make_indices(buffer);

    let mut ml = [Meshlet::default(); 333];
    let mut mv = [0u32; 333 * 3];
    let mut mt = [0u8; 333 * 4];

    let _ = build_meshlets(
        &mut ml, &mut mv, &mut mt, &ib, indices, &vb, 100, 16, /* max_vertices= */ 4, /* max_triangles= */ 4, 0.0,
    );

    let _ = build_meshlets_scan(
        &mut ml, &mut mv, &mut mt, &ib, indices, 100, /* max_vertices= */ 4, /* max_triangles= */ 4,
    );
    let _ = build_meshlets_scan(
        &mut ml, &mut mv, &mut mt, &ib, indices, 100, /* max_vertices= */ 4, /* max_triangles= */ 8,
    );

    // (max_vertices, min_triangles, max_triangles)
    for (v, tmin, tmax) in [(4, 4, 8), (8, 8, 16), (16, 8, 32), (16, 16, 32)] {
        let _ = build_meshlets_flex(&mut ml, &mut mv, &mut mt, &ib, indices, &vb, 100, 16, v, tmin, tmax, 0.0, 2.0);
    }

    for (v, tmin, tmax) in [(4, 4, 8), (8, 4, 32), (8, 8, 32), (8, 12, 32), (16, 16, 32), (16, 32, 32)] {
        let _ = build_meshlets_spatial(&mut ml, &mut mv, &mut mt, &ib, indices, &vb, 100, 16, v, tmin, tmax, 0.0);
    }

    let mut part = [0u32; 333];
    let mut clsize = [0u32; 333];

    // treat each triangle as a cluster
    for i in 0..indices {
        clsize[i / 3] = 3;
    }

    let _ = partition_clusters(
        &mut part,
        &ib,
        indices,
        &clsize,
        indices / 3,
        Some(&vb),
        100,
        16,
        /* target_group_size= */ 12,
    );
}
