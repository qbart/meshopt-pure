//! Golden tests: clod_build + clod_local_indices + clod_build_hierarchy outputs are serialized and hashed (FNV-1a 64);
//! the expected hashes come from the C++ `demo/clusterlod.h` run on the same meshes and configurations
//! (see docs/spec/35-clusterlod.md). Set `CLOD_DUMP=dir` to write the serialized outputs for diffing.

use clusterlod::*;

const VERTEX_FLOATS: usize = 8; // px py pz nx ny nz tx ty

struct Mesh {
    vertices: Vec<f32>,
    indices: Vec<u32>,
}

fn load(name: &str) -> Mesh {
    let path = format!("{}/tests/data/{name}.bin", env!("CARGO_MANIFEST_DIR"));
    let data = std::fs::read(&path).unwrap_or_default();
    let word = |i: usize| {
        data.get(i * 4..i * 4 + 4)
            .map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };

    let vertex_count = word(0) as usize;
    let index_count = word(1) as usize;
    let vertices = (0..vertex_count * VERTEX_FLOATS).map(|i| f32::from_bits(word(2 + i))).collect();
    let indices = (0..index_count).map(|i| word(2 + vertex_count * VERTEX_FLOATS + i)).collect();

    Mesh { vertices, indices }
}

fn make_config(c: u32) -> ClodConfig {
    let config = |t| clod_default_config(t).unwrap_or_default();
    let config_rt = |t| clod_default_config_rt(t).unwrap_or_default();

    let mut config_c = config(128);
    match c {
        1 => config_c = config_rt(128),
        2 => {
            config_c = config(64);
            config_c.partition_sort = true;
        }
        3 => {
            config_c.simplify_permissive = false;
            config_c.simplify_fallback_permissive = true;
        }
        4 => {
            config_c.simplify_regularize = true;
            config_c.simplify_preserve_folds = true;
        }
        5 => config_c.simplify_dilate_borders = true,
        6 => {
            config_c.simplify_error_edge_limit = 1.0;
            config_c.simplify_error_merge_additive = 0.5;
            config_c.simplify_error_merge_previous = 1.5;
        }
        7 => {
            config_c.optimize_bounds = true;
            config_c.optimize_clusters = false;
        }
        8 => {
            config_c = config_rt(64);
            config_c.cluster_fill_weight = 0.25;
            config_c.partition_spatial = false;
        }
        9 => config_c.simplify_permissive = false,
        10 => {
            config_c = config(32);
            config_c.simplify_fallback_sloppy = false;
            config_c.optimize_clusters_level = 3;
        }
        11 => {
            config_c.simplify_error_clamped = false;
            config_c.simplify_ratio = 0.6;
            config_c.simplify_threshold = 0.9;
            config_c.simplify_error_factor_sloppy = 3.0;
            config_c.partition_size = 8;
        }
        12 => {
            config_c = config(256);
            config_c.simplify_dilate_borders = true;
            config_c.partition_sort = true;
            config_c.optimize_bounds = true;
        }
        _ => {}
    }
    config_c
}

#[derive(Default)]
struct Dump(Vec<u8>);

impl Dump {
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn bounds(&mut self, b: &ClodBounds) {
        for v in [b.center[0], b.center[1], b.center[2], b.radius, b.error] {
            self.f32(v);
        }
    }
    fn hash(&self) -> u64 {
        self.0
            .iter()
            .fold(14695981039346656037u64, |h, &b| (h ^ b as u64).wrapping_mul(1099511628211))
    }
}

fn run(mesh: &Mesh, c: u32) -> Result<(Dump, usize), meshoptimizer::Error> {
    let config = make_config(c);
    let attribute_weights = [0.5f32; 3];
    let lock: Vec<u8> = (0..mesh.vertices.len() / VERTEX_FLOATS)
        .map(|i| if i % 7 == 0 { meshoptimizer::SIMPLIFY_VERTEX_LOCK } else { 0 })
        .collect();

    let mut clod_mesh = ClodMesh {
        indices: &mesh.indices,
        index_count: mesh.indices.len(),
        vertex_count: mesh.vertices.len() / VERTEX_FLOATS,
        vertex_positions: &mesh.vertices,
        vertex_positions_stride: VERTEX_FLOATS * 4,
        ..ClodMesh::default()
    };
    if c != 9 {
        clod_mesh.vertex_attributes = mesh.vertices.get(3..);
        clod_mesh.vertex_attributes_stride = VERTEX_FLOATS * 4;
        clod_mesh.attribute_weights = Some(&attribute_weights);
        clod_mesh.attribute_count = 3;
        clod_mesh.attribute_protect_mask = (1 << 3) | (1 << 4);
    }
    if c == 10 {
        clod_mesh.vertex_lock = Some(&lock);
    }

    let mut out = Dump::default();
    let mut groups = Vec::new();
    let mut local_error = None;

    clod_build(config, clod_mesh, |group, clusters, positions| {
        out.i32(group.depth);
        out.bounds(&group.simplified);
        out.u32(clusters.len() as u32);

        for cl in clusters {
            out.i32(cl.refined);
            out.bounds(&cl.bounds);
            out.u32(cl.index_count as u32);
            out.u32(cl.vertex_count as u32);
            for &v in cl.indices {
                out.u32(v);
            }
            if config.simplify_dilate_borders {
                for &v in cl.indices {
                    for k in 0..3 {
                        out.f32(positions[v as usize * VERTEX_FLOATS + k]);
                    }
                }
            }

            let mut lv = vec![0u32; cl.index_count];
            let mut lt = vec![0u8; cl.index_count];
            match clod_local_indices(&mut lv, &mut lt, cl.indices, cl.index_count) {
                Ok(uv) => {
                    out.u32(uv as u32);
                    for &v in &lv[..uv] {
                        out.u32(v);
                    }
                    out.0.extend_from_slice(&lt);
                }
                Err(e) => local_error = Some(e),
            }
        }

        groups.push(group);
        groups.len() as i32 - 1
    })?;

    if let Some(e) = local_error {
        return Err(e);
    }

    if let Some(last) = groups.last() {
        for width in [8, 4, 2] {
            let levels = last.depth as usize + 1;
            let mut nodes = vec![ClodNode::default(); clod_build_hierarchy_bound(groups.len(), width, levels)?];
            out.u32(nodes.len() as u32);
            let count = clod_build_hierarchy(&mut nodes, &groups, groups.len(), width, levels)?;
            out.u32(count as u32);
            for n in &nodes[..count] {
                out.bounds(&n.bounds);
                out.i32(n.group);
                out.u32(n.child_offset);
                out.u32(n.child_count);
            }
        }
    }

    Ok((out, groups.len()))
}

// generated by the C++ oracle (demo/clusterlod.h at ref f0a684f)
const EXPECTED: &[(&str, u32, u64)] = &[
    ("pirate", 0, 0x215b6a96296552c0),
    ("pirate", 1, 0x38a823101c4b4bec),
    ("pirate", 2, 0x29cf387be06d37f3),
    ("pirate", 3, 0x6c9ce5eeaac3234f),
    ("pirate", 4, 0xd9dbb284ee7f324b),
    ("pirate", 5, 0x71f353362bc11b39),
    ("pirate", 6, 0x83838369f8682326),
    ("pirate", 7, 0x6d3e01a12bbc166a),
    ("pirate", 8, 0x95b2ab672b708fce),
    ("pirate", 9, 0xceaecd3952df682d),
    ("pirate", 10, 0xb4b47ba27cd55e7f),
    ("pirate", 11, 0x8572970a591add83),
    ("pirate", 12, 0x8516eb8f5d6bf6b1),
    ("grid", 0, 0x930d51941f6eb8b3),
    ("grid", 1, 0x3d7126e03c06be6e),
    ("grid", 2, 0x847a0749869aa7ad),
    ("grid", 3, 0x930d51941f6eb8b3),
    ("grid", 4, 0xbabd5488cdd072ce),
    ("grid", 5, 0x433e3c38d03415af),
    ("grid", 6, 0xbfd73b965d016496),
    ("grid", 7, 0x63b4c6f3ff166084),
    ("grid", 8, 0xc90796dd60891e8a),
    ("grid", 9, 0x930d51941f6eb8b3),
    ("grid", 10, 0xadf85e45d25e38af),
    ("grid", 11, 0xffa10e529e40606b),
    ("grid", 12, 0x761c3d325634a3ff),
];

#[test]
fn golden_outputs_match_reference() {
    let dump_dir = std::env::var("CLOD_DUMP").ok();
    let mut failures = Vec::new();

    for &(name, c, expected) in EXPECTED {
        let mesh = load(name);
        assert!(!mesh.indices.is_empty(), "missing test mesh {name}");

        match run(&mesh, c) {
            Ok((dump, _)) => {
                if let Some(dir) = &dump_dir {
                    let _ = std::fs::write(format!("{dir}/dump_{name}_{c}.rust.bin"), &dump.0);
                }
                if dump.hash() != expected {
                    failures.push(format!("{name} config {c}: hash {:016x}, expected {expected:016x}", dump.hash()));
                }
            }
            Err(e) => failures.push(format!("{name} config {c}: error {e}")),
        }
    }

    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn dilation_moves_border_vertices() {
    // the grid has straight borders that simplification preserves exactly, pirate has open borders that shrink
    let mesh = load("pirate");
    let mut config = make_config(5);
    config.simplify_dilate_borders = true;

    let clod_mesh = ClodMesh {
        indices: &mesh.indices,
        index_count: mesh.indices.len(),
        vertex_count: mesh.vertices.len() / VERTEX_FLOATS,
        vertex_positions: &mesh.vertices,
        vertex_positions_stride: VERTEX_FLOATS * 4,
        ..ClodMesh::default()
    };

    let mut moved = false;
    let result = clod_build(config, clod_mesh, |_, _, positions| {
        moved |= positions != mesh.vertices.as_slice();
        -1
    });

    assert!(result.is_ok());
    assert!(moved);
}

#[test]
fn invalid_arguments() {
    assert!(clod_default_config(3).is_err());
    assert!(clod_default_config(257).is_err());
    assert!(clod_default_config_rt(4).is_ok());
    assert!(clod_build_hierarchy_bound(10, 1, 1).is_err());

    let mesh = load("pirate");
    let config = make_config(0);
    let base = ClodMesh {
        indices: &mesh.indices,
        index_count: mesh.indices.len(),
        vertex_count: mesh.vertices.len() / VERTEX_FLOATS,
        vertex_positions: &mesh.vertices,
        vertex_positions_stride: VERTEX_FLOATS * 4,
        ..ClodMesh::default()
    };

    let mut called = false;
    let build = |mesh: ClodMesh, called: &mut bool| {
        clod_build(config, mesh, |_, _, _| {
            *called = true;
            -1
        })
    };

    // attribute stride must be a multiple of 4 and fit attribute_count and the protect mask
    assert!(
        build(
            ClodMesh {
                vertex_attributes_stride: 6,
                ..base
            },
            &mut called
        )
        .is_err()
    );
    assert!(
        build(
            ClodMesh {
                attribute_count: 3,
                vertex_attributes_stride: 8,
                ..base
            },
            &mut called
        )
        .is_err()
    );
    assert!(
        build(
            ClodMesh {
                attribute_protect_mask: 1 << 2,
                vertex_attributes_stride: 8,
                ..base
            },
            &mut called
        )
        .is_err()
    );
    // protect mask needs attribute data
    assert!(
        build(
            ClodMesh {
                attribute_protect_mask: 1,
                vertex_attributes_stride: 32,
                ..base
            },
            &mut called
        )
        .is_err()
    );
    // out of range indices and short buffers
    assert!(build(ClodMesh { vertex_count: 10, ..base }, &mut called).is_err());
    assert!(
        build(
            ClodMesh {
                index_count: mesh.indices.len() + 3,
                ..base
            },
            &mut called
        )
        .is_err()
    );
    assert!(
        build(
            ClodMesh {
                vertex_lock: Some(&[0u8; 4]),
                ..base
            },
            &mut called
        )
        .is_err()
    );
    assert!(!called);

    // empty mesh produces no groups
    assert!(build(ClodMesh { index_count: 0, ..base }, &mut called).is_ok());
    assert!(!called);

    // hierarchy needs groups for every level and enough output space
    let groups = [
        ClodGroup {
            depth: 0,
            ..ClodGroup::default()
        },
        ClodGroup {
            depth: 2,
            ..ClodGroup::default()
        },
    ];
    let mut nodes = vec![ClodNode::default(); 16];
    assert!(clod_build_hierarchy(&mut nodes, &groups, 2, 8, 3).is_err());
    assert!(clod_build_hierarchy(&mut nodes[..1], &groups, 2, 8, 1).is_err());
    assert_eq!(clod_build_hierarchy(&mut nodes, &groups, 2, 8, 1), Ok(1));
}
