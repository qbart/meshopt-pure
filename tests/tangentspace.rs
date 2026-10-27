//! Port of tangent/normal generation tests from `demo/tests.cpp`.
#![allow(clippy::approx_constant, clippy::excessive_precision)]

use meshoptimizer::*;

/// `struct Vertex { float px, py, pz; float nx, ny, nz; float tx, ty; }`
const VERTEX_SIZE: usize = 8 * 4;

#[test]
fn tangents_basic() -> Result<()> {
    // unindexed quad with matching corner data for shared diagonal
    #[rustfmt::skip]
    let vertices: [f32; 6 * 8] = [
        0.0, 0.0, 0.0, -0.28, 0.0, 0.96, 0.0, 0.0, // diag 1
        1.0, 0.0, 0.0, 0.28, 0.0, 0.96, 1.0, 0.0,
        1.0, 1.0, 0.0, 0.28, 0.0, 0.96, 1.0, 1.0, // diag 2
        0.0, 0.0, 0.0, -0.28, 0.0, 0.96, 0.0, 0.0, // diag 1
        1.0, 1.0, 0.0, 0.28, 0.0, 0.96, 1.0, 1.0, // diag 2
        0.0, 1.0, 0.0, -0.28, 0.0, 0.96, 0.0, 1.0,
    ];

    // unindexed input: indices == None, vertex_count == index_count
    let mut tangents = [0.0f32; 6 * 4];
    generate_tangents(
        &mut tangents,
        None,
        6,
        &vertices,
        6,
        VERTEX_SIZE,
        &vertices[3..],
        VERTEX_SIZE,
        &vertices[6..],
        VERTEX_SIZE,
        0,
    )?;

    // (1, 0, 0) reprojected onto tilted normals
    let left = [0.96, 0.0, 0.28, 1.0];
    let right = [0.96, 0.0, -0.28, 1.0];

    let expected = [left, right, right, left, right, left];

    for i in 0..6 {
        for k in 0..4 {
            assert!((tangents[i * 4 + k] - expected[i][k]).abs() < 1e-3);
        }
    }

    // shared vertices get the same tangent vector
    for k in 0..4 {
        assert_eq!(tangents[k], tangents[3 * 4 + k]); // diag 1
        assert_eq!(tangents[2 * 4 + k], tangents[4 * 4 + k]); // diag 2
    }
    Ok(())
}

#[test]
fn tangent_degenerate() -> Result<()> {
    #[rustfmt::skip]
    let vertices: [f32; 6 * 8] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0,
        -1.0, -2.0, 1.0, 0.0, 0.0, 1.0, 0.0, -2.0,
        0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
        -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
    ];

    #[rustfmt::skip]
    let indices = [
        // outer, positive
        0, 3, 1,
        // inner, degenerate UVs
        0, 1, 2,
        // outer, positive
        1, 4, 2,
        // outer, negative
        2, 5, 0,
    ];

    let mut tangents = [0.0f32; 12 * 4];
    generate_tangents(
        &mut tangents,
        Some(&indices),
        12,
        &vertices,
        6,
        VERTEX_SIZE,
        &vertices[3..],
        VERTEX_SIZE,
        &vertices[6..],
        VERTEX_SIZE,
        TANGENT_COMPATIBLE,
    )?;

    let vx = 0.0836;
    let vy = 0.9965;
    #[rustfmt::skip]
    let expected: [[f32; 4]; 12] = [
        // outer, positive
        [1.0, 0.0, 0.0, 1.0], // 0
        [1.0, 0.0, 0.0, 1.0], // 3
        [vx, vy, 0.0, 1.0],   // 1
        // inner, degenerate UVs
        [1.0, 0.0, 0.0, 1.0], // 0
        [vx, vy, 0.0, 1.0],   // 1
        [0.0, 1.0, 0.0, 1.0], // 2
        // outer, positive
        [vx, vy, 0.0, 1.0],   // 1
        [0.0, 1.0, 0.0, 1.0], // 4
        [0.0, 1.0, 0.0, 1.0], // 2
        // outer, negative
        [-1.0, 0.0, 0.0, -1.0], // 2 (split)
        [-1.0, 0.0, 0.0, -1.0], // 5
        [-1.0, 0.0, 0.0, -1.0], // 0 (split)
    ];

    for i in 0..12 {
        for k in 0..4 {
            assert!((tangents[i * 4 + k] - expected[i][k]).abs() < 1e-3);
        }
    }
    Ok(())
}

#[test]
fn normals_basic() -> Result<()> {
    #[rustfmt::skip]
    let vertices: [f32; 4 * 3] = [
        -1.0, -0.57735, 0.0,
        1.0, -0.57735, 0.0,
        0.0, 1.15470, 0.0,
        0.0, 0.0, 0.38,
    ];

    // flattened tetrahedron (apex is closer to the base so that we can get soft side edges)
    let indices = [0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3];

    let mut normals = [0.0f32; 12 * 3];
    generate_normals(&mut normals, Some(&indices), 12, &vertices, 4, 12, 3.14159265 / 3.0, 0.0)?;

    let base = [0.0, 0.0, -1.0];
    let side0 = [-0.2707, -0.1563, 0.9499];
    let side1 = [0.2707, -0.1563, 0.9499];
    let side2 = [0.0, 0.3126, 0.9499];
    let apex = [0.0, 0.0, 1.0];

    let expected = [base, base, base, side0, side1, apex, side1, side2, apex, side2, side0, apex];

    for i in 0..12 {
        for k in 0..3 {
            assert!((normals[i * 3 + k] - expected[i][k]).abs() < 1e-3);
        }
    }

    // shared vertices around soft side edges get the same normal vector
    for k in 0..3 {
        assert_eq!(normals[3 * 3 + k], normals[10 * 3 + k]); // side0
        assert_eq!(normals[4 * 3 + k], normals[6 * 3 + k]); // side1
        assert_eq!(normals[7 * 3 + k], normals[9 * 3 + k]); // side2
        assert_eq!(normals[5 * 3 + k], normals[8 * 3 + k]); // apex
        assert_eq!(normals[5 * 3 + k], normals[11 * 3 + k]); // apex
    }
    Ok(())
}

#[test]
fn normals_degenerate() -> Result<()> {
    #[rustfmt::skip]
    let vertices: [f32; 5 * 3] = [
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        1.0, 0.0, 0.0,
        2.0, 0.0, 0.0,
        0.0, 0.5, 0.0,
    ];

    // degenerate triangle connects the triangles across a hard edge; corners inherit normals from adjacent triangles
    let indices = [0, 1, 2, 0, 2, 3, 0, 3, 4];

    let mut normals = [0.0f32; 9 * 3];
    generate_normals(&mut normals, Some(&indices), 9, &vertices, 5, 12, 3.14159265 * 3.0 / 4.0, 0.0)?;

    let negative = [0.0, 0.0, -1.0];
    let positive = [0.0, 0.0, 1.0];
    let expected = [
        negative, negative, negative, negative, negative, positive, positive, positive, positive,
    ];

    for i in 0..9 {
        for k in 0..3 {
            assert!((normals[i * 3 + k] - expected[i][k]).abs() < 1e-3);
        }
    }
    Ok(())
}
