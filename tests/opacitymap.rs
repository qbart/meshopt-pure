//! Port of opacity micromap tests from `demo/tests.cpp`.
#![allow(clippy::needless_range_loop)]

use meshoptimizer::*;

const TEXTURE_SIZE: u32 = 32;

fn circle_texture() -> Vec<u8> {
    let mut texture = vec![0u8; (TEXTURE_SIZE * TEXTURE_SIZE) as usize];

    let center = TEXTURE_SIZE as f32 * 0.5;
    let radius = 10.0f32;

    for y in 0..TEXTURE_SIZE {
        for x in 0..TEXTURE_SIZE {
            let dx = x as f32 + 0.5 - center;
            let dy = y as f32 + 0.5 - center;
            let dc = radius - (dx * dx + dy * dy).sqrt();

            texture[(y * TEXTURE_SIZE + x) as usize] = quantize_unorm(dc + 0.5, 8) as u8;
        }
    }

    texture
}

#[test]
fn opacity_map() -> Result<()> {
    const TRIANGLE_COUNT: usize = 6;
    const VERTEX_COUNT: usize = 8;

    #[rustfmt::skip]
    let indices: [u32; TRIANGLE_COUNT * 3] = [
        // 4 corner triangles
        0, 4, 7,
        1, 5, 4,
        2, 6, 5,
        3, 7, 6,

        // 2 center triangles (2x larger)
        4, 6, 5, // note: this triangle is flipped from its correct orientation to produce the same OMM to test compaction
        4, 6, 7,
    ];

    #[rustfmt::skip]
    let uvs: [f32; VERTEX_COUNT * 2] = [
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
        0.5, 0.0,
        1.0, 0.5,
        0.5, 1.0,
        0.0, 0.5,
    ];

    let texture = circle_texture();

    // subdivision parameters
    let target_edge = 2.5;
    let max_level = 4;

    // state histogram for testing
    let mut histogram = [[0i32; 4]; 2];

    for k in 0..2 {
        let states = 2 << k;

        let mut levels = [0u8; TRIANGLE_COUNT];
        let mut sources = [0u32; TRIANGLE_COUNT];
        let mut omm_indices = [0i32; TRIANGLE_COUNT];

        // compute level and source triangle per OMM; note that this can also deduplicate OMMs based on UVs but in our case the UVs are unique
        let omm_count = opacity_map_measure(
            &mut levels,
            &mut sources,
            &mut omm_indices,
            &indices,
            TRIANGLE_COUNT * 3,
            &uvs,
            VERTEX_COUNT,
            4 * 2,
            TEXTURE_SIZE,
            TEXTURE_SIZE,
            max_level,
            target_edge,
        )?;
        assert!(omm_count <= TRIANGLE_COUNT);

        // validate expected levels/special indices based on underlying test data; depends on implementation specifics, might change
        assert_eq!(levels, [2, 2, 2, 2, 3, 3]);

        // layout OMM data
        let mut offsets = vec![0u32; omm_count];
        let mut data_size = 0;

        for i in 0..omm_count {
            offsets[i] = data_size as u32;
            data_size += opacity_map_entry_size(i32::from(levels[i]), states)?;
        }

        let mut data = vec![0u8; data_size];

        for i in 0..omm_count {
            let tri = sources[i] as usize;
            assert!(tri < TRIANGLE_COUNT);

            let uv = |c: usize| {
                let v = indices[tri * 3 + c] as usize;
                [uvs[v * 2], uvs[v * 2 + 1]]
            };

            let offset = offsets[i] as usize;
            opacity_map_rasterize(
                &mut data[offset..],
                i32::from(levels[i]),
                states,
                &uv(0),
                &uv(1),
                &uv(2),
                &texture,
                1,
                TEXTURE_SIZE as usize,
                TEXTURE_SIZE,
                TEXTURE_SIZE,
            )?;

            let micro_triangle_count = 1usize << (levels[i] * 2);

            for j in 0..micro_triangle_count {
                let byte = data[offset + (j >> if states == 2 { 3 } else { 2 })];
                let state = (byte >> if states == 2 { j & 7 } else { (j & 3) * 2 }) & (states - 1) as u8;
                histogram[k][usize::from(state)] += 1;
            }
        }

        // compact OMMs; note, this can be done per mesh but for optimal reuse this should be done for all meshes in model/scene at once
        let size = data.len();
        let compact_count = opacity_map_compact(
            &mut data,
            size,
            &mut levels,
            &mut offsets,
            omm_count,
            &mut omm_indices,
            TRIANGLE_COUNT,
            states,
        )?;
        assert!(compact_count <= omm_count);
        let trimmed = if compact_count == 0 {
            0
        } else {
            offsets[compact_count - 1] as usize + opacity_map_entry_size(i32::from(levels[compact_count - 1]), states)?
        };
        data.truncate(trimmed);

        // after compaction, some OMM indices may be replaced with a special index (-4..-1)
        for &i in &omm_indices {
            assert!(i < 0 || (i as usize) < compact_count);
        }

        // validate expected levels/special indices based on underlying test data; depends on implementation specifics, might change
        assert!(levels[0] == 3 && levels[1] == 3); // note: OMM data got compacted so we only have 3-level OMMs left
        assert_eq!(omm_indices[..4], [-1, -1, -1, -1]);
        assert!(compact_count == 1 && omm_indices[4] == 0 && omm_indices[5] == 0); // we force the two center triangles to use opposite orientations to make sure their data matches
    }

    // validate expected histogram based on underlying test data; depends on rasterization specifics, might change
    assert_eq!(histogram[0][0], histogram[1][0] + histogram[1][2]);
    assert_eq!(histogram[0][1], histogram[1][1] + histogram[1][3]);

    let opaque = histogram[0][1] as f32 / (histogram[0][0] + histogram[0][1]) as f32;
    let known = (histogram[1][0] + histogram[1][1]) as f32 / (histogram[1][0] + histogram[1][1] + histogram[1][2] + histogram[1][3]) as f32;

    assert!((opaque - 0.36).abs() < 1e-2);
    assert!((known - 0.76).abs() < 1e-2);
    Ok(())
}

#[test]
fn opacity_map_rasterize0() -> Result<()> {
    let texture = circle_texture();

    let uv0 = [0.0, 0.0];
    let uv1 = [1.0, 0.0];
    let uv2 = [0.0, 1.0];
    let uv3 = [0.5, 0.0];
    let uv4 = [0.0, 0.5];
    let uv5 = [0.5, 0.5];

    let size = TEXTURE_SIZE as usize;
    let (mut r0, mut r1, mut r2) = ([0u8], [0u8], [0u8]);
    opacity_map_rasterize(&mut r0, 0, 4, &uv0, &uv1, &uv2, &texture, 1, size, TEXTURE_SIZE, TEXTURE_SIZE)?;
    opacity_map_rasterize(&mut r1, 0, 4, &uv0, &uv3, &uv4, &texture, 1, size, TEXTURE_SIZE, TEXTURE_SIZE)?;
    opacity_map_rasterize(&mut r2, 0, 4, &uv5, &uv3, &uv4, &texture, 1, size, TEXTURE_SIZE, TEXTURE_SIZE)?;

    assert_eq!(r0[0], 2); // unknown-transparent
    assert_eq!(r1[0], 0); // transparent
    assert_eq!(r2[0], 3); // unknown-opaque
    Ok(())
}

fn opacity_map_special_index(level: i32, states: i32, value: u8) -> Result<i32> {
    let mut data = vec![value; opacity_map_entry_size(level, states)?];
    let mut levels = [level as u8];
    let mut offsets = [0u32];
    let mut omm_indices = [0i32];

    let size = data.len();
    let compact_count = opacity_map_compact(&mut data, size, &mut levels, &mut offsets, 1, &mut omm_indices, 1, states)?;
    assert_eq!(compact_count, 0);
    Ok(omm_indices[0])
}

#[test]
fn opacity_map_special() -> Result<()> {
    for level in 0..=2 {
        assert_eq!(opacity_map_special_index(level, 2, 0x00)?, -1);
        assert_eq!(opacity_map_special_index(level, 2, 0xff)?, -2);

        assert_eq!(opacity_map_special_index(level, 4, 0x00)?, -1);
        assert_eq!(opacity_map_special_index(level, 4, 0x55)?, -2);
        assert_eq!(opacity_map_special_index(level, 4, 0xaa)?, -3);
        assert_eq!(opacity_map_special_index(level, 4, 0xff)?, -4);
    }
    Ok(())
}
