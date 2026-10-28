//! Mesh transformation, merging, deduplication, filtering and optimization (`mesh.cpp`).

use std::collections::HashMap;

use crate::cgltf::{self, AttributeType, Data, PrimitiveType};
use crate::cxx::{d2usize, fmax, fmin, sort_by};
use crate::node::decompose_transform;
use crate::parsegltf::are_extras_equal;
use crate::stream::attr_floats;
use crate::types::*;
use meshoptimizer as mo;

fn inverse_transpose(result: &mut [f32; 16], transform: &[f32; 16]) -> f32 {
    let m = |r: usize, c: usize| transform[r * 4 + c];

    let det = m(0, 0) * (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
        + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));

    let invdet = if det == 0.0 { 0.0 } else { 1.0 / det };

    let mut r = [[0.0f32; 4]; 4];

    r[0][0] = (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) * invdet;
    r[1][0] = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * invdet;
    r[2][0] = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * invdet;
    r[0][1] = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) * invdet;
    r[1][1] = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * invdet;
    r[2][1] = (m(1, 0) * m(0, 2) - m(0, 0) * m(1, 2)) * invdet;
    r[0][2] = (m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1)) * invdet;
    r[1][2] = (m(2, 0) * m(0, 1) - m(0, 0) * m(2, 1)) * invdet;
    r[2][2] = (m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1)) * invdet;

    r[3][3] = 1.0;

    *result = *r.as_flattened().first_chunk::<16>().unwrap_or(&[0.0; 16]);

    det
}

fn transform_position(res: &mut Attr, ptr: &Attr, transform: &[f32; 16]) {
    let x = ptr[0] * transform[0] + ptr[1] * transform[4] + ptr[2] * transform[8] + transform[12];
    let y = ptr[0] * transform[1] + ptr[1] * transform[5] + ptr[2] * transform[9] + transform[13];
    let z = ptr[0] * transform[2] + ptr[1] * transform[6] + ptr[2] * transform[10] + transform[14];

    res[0] = x;
    res[1] = y;
    res[2] = z;
}

fn transform_normal(res: &mut Attr, ptr: &Attr, transform: &[f32; 16]) {
    let x = ptr[0] * transform[0] + ptr[1] * transform[4] + ptr[2] * transform[8];
    let y = ptr[0] * transform[1] + ptr[1] * transform[5] + ptr[2] * transform[9];
    let z = ptr[0] * transform[2] + ptr[1] * transform[6] + ptr[2] * transform[10];

    let l = (x * x + y * y + z * z).sqrt();
    let s = if l == 0.0 { 0.0 } else { 1.0 / l };

    res[0] = x * s;
    res[1] = y * s;
    res[2] = z * s;
}

/// Index of the first stream with the given type and index (`getStream`).
pub(crate) fn get_stream(mesh: &Mesh, type_: AttributeType, index: i32) -> Option<usize> {
    mesh.streams.iter().position(|s| s.type_ == type_ && s.index == index)
}

/// `transformMesh(target, mesh, node)`; assumes mesh & target are structurally identical.
fn transform_mesh(target: &mut Mesh, mesh: &Mesh, transform: &[f32; 16]) {
    let mut transforminvt = [0.0f32; 16];
    let det = inverse_transpose(&mut transforminvt, transform);

    for (stream, source) in target.streams.iter_mut().zip(&mesh.streams) {
        match stream.type_ {
            AttributeType::Position => {
                for (d, s) in stream.data.iter_mut().zip(&source.data) {
                    transform_position(d, s, transform);
                }
            }
            AttributeType::Normal => {
                for (d, s) in stream.data.iter_mut().zip(&source.data) {
                    transform_normal(d, s, &transforminvt);
                }
            }
            AttributeType::Tangent => {
                for (d, s) in stream.data.iter_mut().zip(&source.data) {
                    transform_normal(d, s, transform);
                }

                if det < 0.0 {
                    // negative scale means we need to flip tangent frame handedness
                    for d in stream.data.iter_mut() {
                        d[3] = -d[3];
                    }
                }
            }
            _ => {}
        }
    }

    // copy indices so that we can modify them below
    target.indices.clone_from(&mesh.indices);

    if det < 0.0 && mesh.type_ == PrimitiveType::Triangles {
        // negative scale means we need to flip face winding
        for tri in target.indices.as_chunks_mut::<3>().0 {
            tri.swap(0, 1);
        }
    }
}

pub fn compare_mesh_targets(lhs: &Mesh, rhs: &Mesh) -> bool {
    lhs.targets == rhs.targets && lhs.target_weights == rhs.target_weights && lhs.target_names == rhs.target_names
}

pub fn compare_mesh_variants(lhs: &Mesh, rhs: &Mesh) -> bool {
    lhs.variants.len() == rhs.variants.len()
        && lhs
            .variants
            .iter()
            .zip(&rhs.variants)
            .all(|(l, r)| l.variant == r.variant && l.material == r.material)
}

pub fn compare_mesh_nodes(lhs: &Mesh, rhs: &Mesh) -> bool {
    lhs.nodes == rhs.nodes
}

fn compare_instances(lhs: &Instance, rhs: &Instance) -> bool {
    crate::material::bits_eq(&lhs.transform, &rhs.transform) && crate::material::bits_eq(&lhs.color, &rhs.color)
}

fn can_merge_mesh_nodes(data: &Data, lhs: usize, rhs: usize, settings: &Settings) -> bool {
    if lhs == rhs {
        return true;
    }

    let (l, r) = (&data.nodes[lhs], &data.nodes[rhs]);

    if l.parent != r.parent {
        return false;
    }

    let lhs_transform = l.has_translation | l.has_rotation | l.has_scale | l.has_matrix | l.weights.is_some();
    let rhs_transform = r.has_translation | r.has_rotation | r.has_scale | r.has_matrix | r.weights.is_some();

    if lhs_transform || rhs_transform {
        return false;
    }

    if settings.keep_nodes {
        let named = |n: &cgltf::Node| n.name.as_deref().is_some_and(|s| !s.is_empty());
        if named(l) || named(r) {
            return false;
        }
    }

    // we can merge nodes that don't have transforms of their own and have the same parent
    // this is helpful when instead of splitting mesh into primitives, DCCs split mesh into mesh nodes
    true
}

fn can_merge_meshes(data: &Data, lhs: &Mesh, rhs: &Mesh, settings: &Settings) -> bool {
    if lhs.scene != rhs.scene {
        return false;
    }

    if lhs.nodes.len() != rhs.nodes.len() {
        return false;
    }

    if !lhs
        .nodes
        .iter()
        .zip(&rhs.nodes)
        .all(|(&l, &r)| can_merge_mesh_nodes(data, l, r, settings))
    {
        return false;
    }

    if lhs.instances.len() != rhs.instances.len() || !lhs.instances.iter().zip(&rhs.instances).all(|(l, r)| compare_instances(l, r)) {
        return false;
    }

    if lhs.material != rhs.material || lhs.skin != rhs.skin || lhs.type_ != rhs.type_ {
        return false;
    }

    if !compare_mesh_targets(lhs, rhs) || !compare_mesh_variants(lhs, rhs) {
        return false;
    }

    if lhs.indices.is_empty() != rhs.indices.is_empty() {
        return false;
    }

    if lhs.streams.len() != rhs.streams.len() {
        return false;
    }

    if settings.keep_extras && !are_extras_equal(&lhs.extras, &rhs.extras) {
        return false;
    }

    lhs.streams
        .iter()
        .zip(&rhs.streams)
        .all(|(l, r)| l.type_ == r.type_ && l.index == r.index && l.target == r.target)
}

fn merge_mesh_data(target: &mut Mesh, mesh: &Mesh) {
    let vertex_offset = target.streams.first().map_or(0, |s| s.data.len());

    for (t, s) in target.streams.iter_mut().zip(&mesh.streams) {
        t.data.extend_from_slice(&s.data);
    }

    target
        .indices
        .extend(mesh.indices.iter().map(|&i| (vertex_offset + i as usize) as u32));
}

const C1: u64 = 0x87c3_7b91_1142_53d5;
const C2: u64 = 0x4cf5_ad43_2745_937f;

fn hash_block(h1: &mut u64, h2: &mut u64, mut k1: u64, mut k2: u64) {
    k1 = k1.wrapping_mul(C1);
    k1 = k1.rotate_left(31);
    k1 = k1.wrapping_mul(C2);
    *h1 ^= k1;
    *h1 = h1.rotate_left(27);
    *h1 = h1.wrapping_add(*h2);
    *h1 = h1.wrapping_mul(5).wrapping_add(0x52dc_e729);
    k2 = k2.wrapping_mul(C2);
    k2 = k2.rotate_left(33);
    k2 = k2.wrapping_mul(C1);
    *h2 ^= k2;
    *h2 = h2.rotate_left(31);
    *h2 = h2.wrapping_add(*h1);
    *h2 = h2.wrapping_mul(5).wrapping_add(0x3849_5ab5);
}

/// `hashUpdate`: MurMurHash3 128-bit style update over `data`.
fn hash_update(hash: &mut [u64; 2], data: &[u8]) {
    let (mut h1, mut h2) = (hash[0], hash[1]);

    let (blocks, tail) = data.as_chunks::<16>();

    // body
    for b in blocks {
        let k1 = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
        let k2 = u64::from_le_bytes([b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]]);
        hash_block(&mut h1, &mut h2, k1, k2);
    }

    // tail
    if !tail.is_empty() {
        let mut t = [0u8; 16];
        t[..tail.len()].copy_from_slice(tail);

        let mut k1 = u64::from_le_bytes([t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]]);
        let mut k2 = u64::from_le_bytes([t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15]]);

        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(31);
        k1 = k1.wrapping_mul(C2);
        h1 ^= k1;
        k2 = k2.wrapping_mul(C2);
        k2 = k2.rotate_left(33);
        k2 = k2.wrapping_mul(C1);
        h2 ^= k2;
    }

    h1 ^= data.len() as u64;
    h2 ^= data.len() as u64;

    *hash = [h1, h2];
}

/// `hashUpdate` over `Attr` data without materializing bytes (each Attr is exactly one 16-byte block).
fn hash_update_attrs(hash: &mut [u64; 2], data: &[Attr]) {
    let (mut h1, mut h2) = (hash[0], hash[1]);

    for a in data {
        let k1 = u64::from(a[0].to_bits()) | (u64::from(a[1].to_bits()) << 32);
        let k2 = u64::from(a[2].to_bits()) | (u64::from(a[3].to_bits()) << 32);
        hash_block(&mut h1, &mut h2, k1, k2);
    }

    let size = (data.len() * 16) as u64;
    h1 ^= size;
    h2 ^= size;

    *hash = [h1, h2];
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn hash_mesh(mesh: &mut Mesh) {
    let mut hash = [41u64, 41u64];

    for stream in &mesh.streams {
        let meta = [stream.type_ as i32, stream.index, stream.target];
        hash_update(&mut hash, &i32_bytes(&meta));

        if let Some(name) = &stream.custom_name {
            hash_update(&mut hash, name.split('\0').next().unwrap_or("").as_bytes());
        }

        hash_update_attrs(&mut hash, &stream.data);
    }

    if !mesh.indices.is_empty() {
        let bytes: Vec<u8> = mesh.indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        hash_update(&mut hash, &bytes);
    }

    let meta = [
        mesh.streams.len() as i32,
        mesh.streams.first().map_or(0, |s| s.data.len() as i32),
        mesh.indices.len() as i32,
        mesh.type_ as i32,
    ];
    hash_update(&mut hash, &i32_bytes(&meta));

    mesh.geometry_hash = hash;
}

fn can_dedup_mesh(mesh: &Mesh, settings: &Settings) -> bool {
    // empty mesh
    if mesh.streams.is_empty() {
        return false;
    }

    // world-space mesh
    if mesh.nodes.is_empty() && mesh.instances.is_empty() {
        return false;
    }

    // has extras
    if settings.keep_extras && mesh.extras.data.is_some() {
        return false;
    }

    // to simplify dedup we ignore complex target setups for now
    mesh.target_weights.is_empty() && mesh.target_names.is_empty() && mesh.variants.is_empty()
}

pub fn dedup_meshes(meshes: &mut [Mesh], settings: &Settings) {
    let mut hashes: HashMap<u64, i32> = HashMap::new();

    for mesh in meshes.iter_mut() {
        hash_mesh(mesh);
        *hashes.entry(mesh.geometry_hash[0] ^ mesh.geometry_hash[1]).or_insert(0) += 1;
    }

    for i in 0..meshes.len() {
        if !can_dedup_mesh(&meshes[i], settings) {
            continue;
        }

        let key = meshes[i].geometry_hash[0] ^ meshes[i].geometry_hash[1];
        if hashes.get(&key).copied().unwrap_or(0) <= 1 {
            continue;
        }

        for j in i + 1..meshes.len() {
            let (head, tail) = meshes.split_at_mut(j);
            let target = &mut head[i];
            let mesh = &mut tail[0];

            if mesh.geometry_hash != target.geometry_hash {
                continue;
            }

            if !can_dedup_mesh(mesh, settings) {
                continue;
            }

            if mesh.scene != target.scene || mesh.material != target.material || mesh.skin != target.skin {
                // mark both meshes as having duplicate geometry; we don't use this in dedupMeshes but it's useful later in the pipeline
                target.geometry_duplicate = true;
                mesh.geometry_duplicate = true;
                continue;
            }

            target.nodes.extend_from_slice(&mesh.nodes);
            target.instances.extend_from_slice(&mesh.instances);

            mesh.streams.clear();
            mesh.indices.clear();
            mesh.nodes.clear();
            mesh.instances.clear();
        }
    }

    for target in meshes.iter_mut() {
        if target.nodes.len() <= 1 {
            continue;
        }

        target.nodes.sort_unstable();
        target.nodes.dedup();
    }
}

pub fn merge_mesh_instances(mesh: &mut Mesh, data: &Data) {
    if mesh.nodes.is_empty() {
        return;
    }

    // fast-path: for single instance meshes we transform in-place
    if mesh.nodes.len() == 1 {
        let transform = cgltf::node_transform_world(&data.nodes, &data.nodes[mesh.nodes[0]]);
        let base = mesh.clone();
        transform_mesh(mesh, &base, &transform);
        mesh.nodes.clear();
        return;
    }

    let base = mesh.clone();
    let mut transformed = base.clone();

    for stream in mesh.streams.iter_mut() {
        stream.data.clear();
    }

    mesh.indices.clear();

    for &node in &base.nodes {
        let transform = cgltf::node_transform_world(&data.nodes, &data.nodes[node]);
        transform_mesh(&mut transformed, &base, &transform);
        merge_mesh_data(mesh, &transformed);
    }

    mesh.nodes.clear();
}

pub fn merge_meshes(meshes: &mut [Mesh], data: &Data, settings: &Settings) {
    for i in 0..meshes.len() {
        if meshes[i].streams.is_empty() {
            continue;
        }

        for j in i + 1..meshes.len() {
            let (head, tail) = meshes.split_at_mut(j);
            let target = &mut head[i];
            let mesh = &mut tail[0];

            if !mesh.streams.is_empty() && can_merge_meshes(data, target, mesh, settings) {
                merge_mesh_data(target, mesh);

                mesh.streams.clear();
                mesh.indices.clear();
                mesh.nodes.clear();
                mesh.instances.clear();
            }
        }
    }
}

pub fn filter_empty_meshes(meshes: &mut Vec<Mesh>) {
    meshes.retain(|mesh| {
        !mesh.streams.is_empty() && !mesh.streams[0].data.is_empty() && (mesh.type_ == PrimitiveType::Points || !mesh.indices.is_empty())
    });
}

fn is_constant(data: &[Attr], value: &Attr, tolerance: f32) -> bool {
    data.iter().all(|a| (0..4).all(|k| (a[k] - value[k]).abs() <= tolerance))
}

pub fn filter_streams(mesh: &mut Mesh, mi: &MaterialInfo) {
    let mut morph_normal = false;
    let mut morph_tangent = false;
    let mut keep_texture_set = -1;

    for stream in &mesh.streams {
        if stream.target != 0 {
            morph_normal = morph_normal || (stream.type_ == AttributeType::Normal && !is_constant(&stream.data, &[0.0; 4], 0.01));
            morph_tangent = morph_tangent || (stream.type_ == AttributeType::Tangent && !is_constant(&stream.data, &[0.0; 4], 0.01));
        }

        if stream.type_ == AttributeType::Texcoord
            && stream.index < 32
            && (mi.texture_set_mask & 1u32.wrapping_shl(stream.index as u32)) != 0
        {
            keep_texture_set = keep_texture_set.max(stream.index);
        }
    }

    let skin = mesh.skin.is_some();
    let points = mesh.type_ == PrimitiveType::Points;

    mesh.streams.retain(|stream| {
        if stream.type_ == AttributeType::Texcoord && stream.index > keep_texture_set {
            return false;
        }

        if stream.type_ == AttributeType::Normal && mi.unlit {
            return false;
        }

        if stream.type_ == AttributeType::Tangent && !mi.needs_tangents {
            return false;
        }

        if (stream.type_ == AttributeType::Joints || stream.type_ == AttributeType::Weights) && !skin {
            return false;
        }

        if stream.type_ == AttributeType::Color && is_constant(&stream.data, &[1.0; 4], 0.01) {
            return false;
        }

        if stream.target != 0 && stream.type_ == AttributeType::Normal && !morph_normal {
            return false;
        }

        if stream.target != 0 && stream.type_ == AttributeType::Tangent && !morph_tangent {
            return false;
        }

        if points && stream.type_ == AttributeType::Normal && !stream.data.is_empty() && is_constant(&stream.data, &stream.data[0], 0.01) {
            return false;
        }

        true
    });
}

fn get_tangent_texcoord(material: &cgltf::Material) -> i32 {
    if material.normal_texture.texture.is_some() {
        return material.normal_texture.texcoord;
    }
    if material.has_clearcoat && material.clearcoat.clearcoat_normal_texture.texture.is_some() {
        return material.clearcoat.clearcoat_normal_texture.texcoord;
    }
    if material.has_anisotropy && material.anisotropy.anisotropy_texture.texture.is_some() {
        return material.anisotropy.anisotropy_texture.texcoord;
    }
    0
}

/// Removes existing tangent streams and appends an empty one; returns its index.
fn prepare_tangent_stream(mesh: &mut Mesh, vertex_count: usize) -> usize {
    // remove existing tangent streams, if any
    // in particular removes morph target streams as we don't support morph tangent generation
    mesh.streams.retain(|s| s.type_ != AttributeType::Tangent);

    mesh.streams.push(Stream {
        type_: AttributeType::Tangent,
        data: vec![[0.0; 4]; vertex_count],
        ..Default::default()
    });

    mesh.streams.len() - 1
}

/// `splitVertices<N>`: assigns per-corner values (`n` floats per corner) to `target`, splitting vertices as needed.
fn split_vertices(mesh: &mut Mesh, target: usize, data: &[f32], n: usize) {
    let vertex_count = mesh.streams[target].data.len();

    // seed each vertex with one of its corner values; the loop below fixes any mismatches
    for (i, &v) in mesh.indices.iter().enumerate() {
        if let Some(d) = mesh.streams[target].data.get_mut(v as usize) {
            d[..n].copy_from_slice(&data[i * n..i * n + n]);
        }
    }

    let mut splits = vec![!0u32; vertex_count];

    for i in 0..mesh.indices.len() {
        let mut a: Attr = [0.0; 4];
        a[..n].copy_from_slice(&data[i * n..i * n + n]);

        let v = mesh.indices[i];
        let mut sv = v;

        // walk the chain of split copies looking for a vertex whose attribute matches
        while sv != !0u32 {
            let t = mesh.streams[target].data[sv as usize];
            if t[0] == a[0] && t[1] == a[1] && t[2] == a[2] && t[3] == a[3] {
                break;
            }
            sv = splits[sv as usize];
        }

        // no match in chain: append a new split copy with the target attribute and chain it
        if sv == !0u32 {
            sv = mesh.streams[target].data.len() as u32;

            for stream in mesh.streams.iter_mut() {
                let value = stream.data[v as usize];
                stream.data.push(value);
            }

            mesh.streams[target].data[sv as usize] = a;

            splits.push(splits[v as usize]);
            splits[v as usize] = sv;
        }

        mesh.indices[i] = sv;
    }
}

pub fn generate_tangents(mesh: &mut Mesh, data: &Data) {
    let Some(material) = mesh.material else {
        return;
    };

    if mesh.type_ != PrimitiveType::Triangles || mesh.indices.is_empty() {
        return;
    }

    let texcoord = get_tangent_texcoord(&data.materials[material]);
    let positions = get_stream(mesh, AttributeType::Position, 0);
    let normals = get_stream(mesh, AttributeType::Normal, 0);
    let uvs = get_stream(mesh, AttributeType::Texcoord, texcoord);

    let (Some(positions), Some(normals), Some(uvs)) = (positions, normals, uvs) else {
        return;
    };

    let vertex_count = mesh.streams[positions].data.len();

    let mut tangents = vec![0.0f32; mesh.indices.len() * 4];
    let ok = mo::generate_tangents(
        &mut tangents,
        Some(mesh.indices.as_slice()),
        mesh.indices.len(),
        attr_floats(&mesh.streams[positions].data),
        vertex_count,
        16,
        attr_floats(&mesh.streams[normals].data),
        16,
        attr_floats(&mesh.streams[uvs].data),
        16,
        0,
    );

    if ok.is_err() {
        return;
    }

    let tangent = prepare_tangent_stream(mesh, vertex_count);

    split_vertices(mesh, tangent, &tangents, 4);
}

pub fn generate_normals(mesh: &mut Mesh, crease_angle: f32) {
    if mesh.type_ != PrimitiveType::Triangles
        || mesh.indices.is_empty()
        || mesh.targets != 0
        || get_stream(mesh, AttributeType::Normal, 0).is_some()
    {
        return;
    }

    let Some(positions) = get_stream(mesh, AttributeType::Position, 0) else {
        return;
    };

    let vertex_count = mesh.streams[positions].data.len();

    let mut normals = vec![0.0f32; mesh.indices.len() * 3];
    let ok = mo::generate_normals(
        &mut normals,
        Some(mesh.indices.as_slice()),
        mesh.indices.len(),
        attr_floats(&mesh.streams[positions].data),
        vertex_count,
        16,
        crease_angle * (std::f32::consts::PI / 180.0), // C++: 3.14159265f, the same float
        1.5,
    );

    if ok.is_err() {
        return;
    }

    mesh.streams.push(Stream {
        type_: AttributeType::Normal,
        data: vec![[0.0; 4]; vertex_count],
        ..Default::default()
    });

    let normal = mesh.streams.len() - 1;

    split_vertices(mesh, normal, &normals, 3);
}

fn reindex_tolerance(type_: AttributeType, settings: &Settings) -> f32 {
    let pow = |bits: i32| (1i32.wrapping_shl(bits as u32)) as f32;

    match type_ {
        AttributeType::Normal | AttributeType::Tangent => fmax(2e-4, if settings.quantize { 1.0 / pow(settings.nrm_bits) } else { 0.0 }),
        AttributeType::Texcoord => fmax(5e-5, if settings.quantize { 0.5 / pow(settings.tex_bits) } else { 0.0 }),
        AttributeType::Color => 0.5 / pow(settings.col_bits),
        AttributeType::Weights => 0.5 / 256.0,
        _ => 0.0,
    }
}

/// `meshopt_remapVertexBuffer` (in place) for `Attr` streams.
fn remap_attrs(data: &mut [Attr], vertex_count: usize, remap: &[u32]) {
    let source: Vec<Attr> = data[..vertex_count.min(data.len())].to_vec();

    for (i, &r) in remap.iter().enumerate().take(source.len()) {
        if r != !0u32
            && let Some(d) = data.get_mut(r as usize)
        {
            *d = source[i];
        }
    }
}

fn reindex_mesh(mesh: &mut Mesh, settings: &Settings) {
    let total_vertices = mesh.streams[0].data.len();
    let total_indices = mesh.indices.len();

    let Some(positions) = get_stream(mesh, AttributeType::Position, 0) else {
        return;
    };

    let streams: Vec<(&[Attr], f32)> = mesh
        .streams
        .iter()
        .filter(|s| s.type_ != AttributeType::Position && s.target == 0)
        .map(|s| (&s.data[..], reindex_tolerance(s.type_, settings)))
        .collect();

    let mut remap = vec![0u32; total_vertices];

    let mut callback = |lhs: u32, rhs: u32| -> bool {
        for (data, tolerance) in &streams {
            let at = |i: u32| data.get(i as usize).copied().unwrap_or_default();
            let (l, r) = (at(lhs), at(rhs));
            let dx = (l[0] - r[0]).abs();
            let dy = (l[1] - r[1]).abs();
            let dz = (l[2] - r[2]).abs();
            let dw = (l[3] - r[3]).abs();

            if dx > *tolerance || dy > *tolerance || dz > *tolerance || dw > *tolerance {
                return false;
            }
        }

        true
    };

    let unique_vertices = mo::generate_vertex_remap_custom(
        &mut remap,
        Some(mesh.indices.as_slice()),
        total_indices,
        attr_floats(&mesh.streams[positions].data),
        total_vertices,
        16,
        Some(&mut callback),
    );

    drop(streams);

    let Ok(unique_vertices) = unique_vertices else {
        return;
    };

    if mo::remap_index_buffer_in_place(&mut mesh.indices, total_indices, &remap).is_err() {
        return;
    }

    for stream in mesh.streams.iter_mut() {
        remap_attrs(&mut stream.data, total_vertices, &remap);
        stream.data.truncate(unique_vertices);
    }
}

fn filter_triangles(mesh: &mut Mesh) {
    let mut write = 0;

    for i in (0..mesh.indices.len() / 3 * 3).step_by(3) {
        let (a, b, c) = (mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]);

        if a != b && a != c && b != c {
            mesh.indices[write] = a;
            mesh.indices[write + 1] = b;
            mesh.indices[write + 2] = c;
            write += 3;
        }
    }

    mesh.indices.truncate(write);
}

fn simplify_attributes(attrs: &mut Vec<f32>, attrw: &mut [f32; 8], stride: usize, mesh: &Mesh, uv_weight: f32) {
    let vertex_count = mesh.streams[0].data.len();

    attrs.resize(vertex_count * stride, 0.0);

    if let Some(s) = get_stream(mesh, AttributeType::Normal, 0) {
        for (i, a) in mesh.streams[s].data.iter().enumerate().take(vertex_count) {
            attrs[i * stride..i * stride + 3].copy_from_slice(&a[..3]);
        }

        attrw[0] = 0.5;
        attrw[1] = 0.5;
        attrw[2] = 0.5;
    }

    if let Some(s) = get_stream(mesh, AttributeType::Color, 0) {
        for (i, a) in mesh.streams[s].data.iter().enumerate().take(vertex_count) {
            attrs[i * stride + 3] = a[0] * a[3];
            attrs[i * stride + 4] = a[1] * a[3];
            attrs[i * stride + 5] = a[2] * a[3];
        }

        attrw[3] = 1.0;
        attrw[4] = 1.0;
        attrw[5] = 1.0;
    }

    if let Some(s) = get_stream(mesh, AttributeType::Texcoord, 0) {
        for (i, a) in mesh.streams[s].data.iter().enumerate().take(vertex_count) {
            attrs[i * stride + 6] = a[0];
            attrs[i * stride + 7] = a[1];
        }
        attrw[6] = uv_weight;
        attrw[7] = uv_weight;
    }
}

fn simplify_update(attrs: &[f32], stride: usize, mesh: &mut Mesh) {
    let vertex_count = mesh.streams[0].data.len();

    if let Some(s) = get_stream(mesh, AttributeType::Normal, 0) {
        for (i, a) in mesh.streams[s].data.iter_mut().enumerate().take(vertex_count) {
            let (nx, ny, nz) = (attrs[i * stride], attrs[i * stride + 1], attrs[i * stride + 2]);
            let nl = (nx * nx + ny * ny + nz * nz).sqrt();
            let ns = if nl > 0.0 { 1.0 / nl } else { 0.0 };
            a[0] = nx * ns;
            a[1] = ny * ns;
            a[2] = nz * ns;
        }
    }

    if let Some(s) = get_stream(mesh, AttributeType::Color, 0) {
        for (i, a) in mesh.streams[s].data.iter_mut().enumerate().take(vertex_count) {
            let cs = if a[3] > 0.0 { 1.0 / a[3] } else { 0.0 };
            a[0] = fmax(0.0, fmin(1.0, attrs[i * stride + 3] * cs));
            a[1] = fmax(0.0, fmin(1.0, attrs[i * stride + 4] * cs));
            a[2] = fmax(0.0, fmin(1.0, attrs[i * stride + 5] * cs));
        }
    }

    if let Some(s) = get_stream(mesh, AttributeType::Texcoord, 0) {
        for (i, a) in mesh.streams[s].data.iter_mut().enumerate().take(vertex_count) {
            a[0] = attrs[i * stride + 6];
            a[1] = attrs[i * stride + 7];
        }
    }
}

fn simplify_protect(locks: &mut Vec<u8>, mesh: &Mesh, presplit_vertices: usize) {
    let Some(positions) = get_stream(mesh, AttributeType::Position, 0) else {
        return;
    };

    let vertex_count = mesh.streams[positions].data.len();

    locks.resize(vertex_count, 0);

    let mut remap = vec![0u32; vertex_count];
    let _ = mo::generate_position_remap(&mut remap, attr_floats(&mesh.streams[positions].data), vertex_count, 16);

    // protect UV discontinuities
    if let Some(s) = get_stream(mesh, AttributeType::Texcoord, 0) {
        let a = &mesh.streams[s].data;

        for i in 0..vertex_count {
            let r = remap[i] as usize;

            if r != i && (a[i][0] != a[r][0] || a[i][1] != a[r][1]) {
                locks[i] |= mo::SIMPLIFY_VERTEX_PROTECT;
            }
        }
    }

    // protect all vertices that were artificially split on the UV mirror edges
    for lock in locks.iter_mut().take(vertex_count).skip(presplit_vertices) {
        *lock |= mo::SIMPLIFY_VERTEX_PROTECT;
    }
}

fn simplify_uv_split(mesh: &mut Mesh, remap: &mut Vec<u32>) {
    let Some(uv) = get_stream(mesh, AttributeType::Texcoord, 0) else {
        return;
    };

    let vertex_count = mesh.streams[uv].data.len();

    let mut uvsign = vec![0u8; mesh.indices.len() / 3];
    let mut flipseam = vec![0u8; vertex_count];

    {
        let uvd = &mesh.streams[uv].data;

        for (t, tri) in mesh.indices.as_chunks::<3>().0.iter().enumerate() {
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);

            let (va, vb, vc) = (uvd[a], uvd[b], uvd[c]);

            let uvarea = (vb[0] - va[0]) * (vc[1] - va[1]) - (vc[0] - va[0]) * (vb[1] - va[1]);
            let flag = if uvarea > 0.0 {
                1
            } else if uvarea < 0.0 {
                2
            } else {
                0
            };

            uvsign[t] = flag;
            flipseam[a] |= flag;
            flipseam[b] |= flag;
            flipseam[c] |= flag;
        }
    }

    let mut split = vec![0u32; vertex_count];
    let mut splits = 0usize;

    remap.resize(vertex_count, 0);

    for i in 0..vertex_count {
        remap[i] = i as u32;

        if flipseam[i] == 3 {
            remap.push(i as u32);

            split[i] = (vertex_count + splits) as u32;
            splits += 1;

            for stream in mesh.streams.iter_mut() {
                let value = stream.data[i];
                stream.data.push(value);
            }
        }
    }

    for (t, tri) in mesh.indices.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let sign = uvsign[t];

        for v in tri.iter_mut() {
            let x = *v as usize;
            if flipseam[x] == 3 && sign == 2 {
                *v = split[x];
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn simplify_mesh(mesh: &mut Mesh, threshold: f32, error: f32, aggressive: bool, lock_borders: bool, permissive: bool, update: bool) {
    if mesh.indices.is_empty() {
        return;
    }

    let Some(positions) = get_stream(mesh, AttributeType::Position, 0) else {
        return;
    };

    let presplit_vertices = mesh.streams[positions].data.len();

    let mut uvremap = Vec::new();
    simplify_uv_split(mesh, &mut uvremap);

    let vertex_count = mesh.streams[positions].data.len();

    let target_index_count = d2usize((mesh.indices.len() / 3) as f64 * f64::from(threshold)) * 3;
    let target_error = error;
    let target_error_aggressive = 1e-1;

    let mut options = 0;
    if lock_borders {
        options |= mo::SIMPLIFY_LOCK_BORDER;
    } else {
        options |= mo::SIMPLIFY_PRUNE;
    }

    if permissive {
        options |= mo::SIMPLIFY_PERMISSIVE;
    }

    if mesh.targets != 0 || get_stream(mesh, AttributeType::Weights, 0).is_some() {
        options |= mo::SIMPLIFY_REGULARIZE;
    }

    let mut indices = vec![0u32; mesh.indices.len()];

    let mut attrw = [0.0f32; 8];
    let mut attrs = Vec::new();
    simplify_attributes(&mut attrs, &mut attrw, 8, mesh, if update { 1.0 } else { 0.0 });

    let mut locks = Vec::new();
    if permissive {
        simplify_protect(&mut locks, mesh, presplit_vertices);
    }
    let lock = if permissive { Some(&locks[..]) } else { None };

    // for now we disable simplify-with-update if the mesh has a second texture coordinate since we don't currently update it and moving vertices may create UV distortion
    if update && mesh.targets == 0 && get_stream(mesh, AttributeType::Texcoord, 1).is_none() {
        indices.clone_from(&mesh.indices);
        let count = mo::simplify_with_update(
            &mut indices,
            mesh.indices.len(),
            mesh.streams[positions].data.as_flattened_mut(),
            vertex_count,
            16,
            &mut attrs,
            32,
            &attrw,
            8,
            lock,
            target_index_count,
            target_error,
            options,
            None,
        )
        .unwrap_or(indices.len());
        indices.truncate(count);
        simplify_update(&attrs, 8, mesh);
    } else {
        let count = mo::simplify_with_attributes(
            &mut indices,
            &mesh.indices,
            mesh.indices.len(),
            attr_floats(&mesh.streams[positions].data),
            vertex_count,
            16,
            &attrs,
            32,
            &attrw,
            8,
            lock,
            target_index_count,
            target_error,
            options,
            None,
        )
        .unwrap_or(0);
        indices.truncate(count);
    }

    std::mem::swap(&mut mesh.indices, &mut indices);

    // Note: if the simplifier got stuck, we can try to reindex without normals/tangents and retry
    // For now we simply fall back to aggressive simplifier instead

    // if the precise simplifier got "stuck", we'll try to simplify using the sloppy simplifier; this is only used when aggressive simplification is enabled as it breaks attribute discontinuities
    if aggressive && mesh.indices.len() > target_index_count {
        indices.resize(indices.len().max(mesh.indices.len()), 0);
        let count = mo::simplify_sloppy(
            &mut indices,
            &mesh.indices,
            mesh.indices.len(),
            attr_floats(&mesh.streams[positions].data),
            vertex_count,
            16,
            None,
            target_index_count,
            target_error_aggressive,
            None,
        )
        .unwrap_or(0);
        indices.truncate(count);
        std::mem::swap(&mut mesh.indices, &mut indices);
    }

    if !uvremap.is_empty() && !mesh.indices.is_empty() {
        let count = mesh.indices.len();
        let _ = mo::remap_index_buffer_in_place(&mut mesh.indices, count, &uvremap);
    }
}

fn optimize_mesh(mesh: &mut Mesh, compressmore: bool) {
    if mesh.indices.is_empty() {
        return;
    }

    let vertex_count = mesh.streams[0].data.len();
    let index_count = mesh.indices.len();

    let ok = if compressmore {
        mo::optimize_vertex_cache_strip_in_place(&mut mesh.indices, index_count, vertex_count)
    } else {
        mo::optimize_vertex_cache_in_place(&mut mesh.indices, index_count, vertex_count)
    };

    if ok.is_err() {
        return;
    }

    let mut remap = vec![0u32; vertex_count];
    let Ok(unique_vertices) = mo::optimize_vertex_fetch_remap(&mut remap, &mesh.indices, index_count, vertex_count) else {
        return;
    };

    if mo::remap_index_buffer_in_place(&mut mesh.indices, index_count, &remap).is_err() {
        return;
    }

    for stream in mesh.streams.iter_mut() {
        remap_attrs(&mut stream.data, vertex_count, &remap);
        stream.data.truncate(unique_vertices);
    }
}

#[derive(Clone, Copy, Default)]
struct BoneInfluence {
    i: f32,
    w: f32,
}

fn filter_bones(mesh: &mut Mesh) {
    const MAX_GROUPS: usize = 8;

    // gather all joint/weight groups; each group contains 4 bone influences
    let mut groups: Vec<(usize, usize)> = Vec::new();

    for i in 0..MAX_GROUPS {
        let jg = get_stream(mesh, AttributeType::Joints, i as i32);
        let wg = get_stream(mesh, AttributeType::Weights, i as i32);

        let (Some(jg), Some(wg)) = (jg, wg) else {
            break;
        };

        groups.push((jg, wg));
    }

    if groups.is_empty() {
        return;
    }

    // weights below cutoff can't be represented in quantized 8-bit storage
    let weight_cutoff = 0.5 / 255.0;

    let vertex_count = mesh.streams[0].data.len();

    let mut inf = [BoneInfluence::default(); MAX_GROUPS * 4];

    for i in 0..vertex_count {
        let mut count = 0;

        // gather all bone influences for this vertex
        for &(jg, wg) in &groups {
            let ja = mesh.streams[jg].data.get(i).copied().unwrap_or_default();
            let wa = mesh.streams[wg].data.get(i).copied().unwrap_or_default();

            for k in 0..4 {
                if wa[k] > weight_cutoff {
                    inf[count] = BoneInfluence { i: ja[k], w: wa[k] };
                    count += 1;
                }
            }
        }

        // pick top 4 influences; this also sorts resulting influences by weight which helps renderers that use influence subset in shader LODs
        sort_by(&mut inf[..count], |lhs, rhs| lhs.w > rhs.w);

        // copy the top 4 influences back into stream 0 - we will remove other streams at the end
        let (jg, wg) = groups[0];

        for k in 0..4 {
            let (j, w) = if k < count { (inf[k].i, inf[k].w) } else { (0.0, 0.0) };

            if let Some(ja) = mesh.streams[jg].data.get_mut(i) {
                ja[k] = j;
            }
            if let Some(wa) = mesh.streams[wg].data.get_mut(i) {
                wa[k] = w;
            }
        }
    }

    // remove redundant weight/joint streams
    mesh.streams
        .retain(|s| !((s.type_ == AttributeType::Joints || s.type_ == AttributeType::Weights) && s.index > 0));
}

fn simplify_point_mesh(mesh: &mut Mesh, threshold: f32) {
    if threshold >= 1.0 {
        return;
    }

    let Some(positions) = get_stream(mesh, AttributeType::Position, 0) else {
        return;
    };

    let colors = get_stream(mesh, AttributeType::Color, 0);

    let vertex_count = mesh.streams[0].data.len();

    let target_vertex_count = d2usize(vertex_count as f64 * f64::from(threshold));

    let color_weight = 1.0;

    let mut indices = vec![0u32; target_vertex_count];
    if target_vertex_count != 0 {
        let count = mo::simplify_points(
            &mut indices,
            attr_floats(&mesh.streams[positions].data),
            vertex_count,
            16,
            colors.map(|c| attr_floats(&mesh.streams[c].data)),
            16,
            color_weight,
            target_vertex_count,
        )
        .unwrap_or(0);
        indices.truncate(count);
    }

    for stream in mesh.streams.iter_mut() {
        let scratch: Vec<Attr> = indices
            .iter()
            .map(|&j| stream.data.get(j as usize).copied().unwrap_or_default())
            .collect();
        stream.data = scratch;
    }
}

fn sort_point_mesh(mesh: &mut Mesh) {
    let Some(positions) = get_stream(mesh, AttributeType::Position, 0) else {
        return;
    };

    // skip spatial sort in presence of custom attributes, since when they refer to mesh features their order is more important to preserve for compression efficiency
    if get_stream(mesh, AttributeType::Custom, 0).is_some() {
        return;
    }

    let vertex_count = mesh.streams[0].data.len();

    let mut remap = vec![0u32; vertex_count];
    if mo::spatial_sort_remap(&mut remap, attr_floats(&mesh.streams[positions].data), vertex_count, 16).is_err() {
        return;
    }

    for stream in mesh.streams.iter_mut() {
        remap_attrs(&mut stream.data, vertex_count, &remap);
    }
}

pub fn process_mesh(mesh: &mut Mesh, settings: &Settings) {
    match mesh.type_ {
        PrimitiveType::Points => {
            simplify_point_mesh(mesh, settings.simplify_ratio);
            sort_point_mesh(mesh);
        }
        PrimitiveType::Lines => {}
        PrimitiveType::Triangles => {
            filter_bones(mesh);
            reindex_mesh(mesh, settings);
            filter_triangles(mesh);

            if settings.simplify_ratio < 1.0 {
                simplify_mesh(
                    mesh,
                    settings.simplify_ratio,
                    settings.simplify_error / mesh.quality,
                    settings.simplify_aggressive,
                    settings.simplify_lock_borders,
                    settings.simplify_permissive,
                    settings.simplify_update,
                );
            }

            optimize_mesh(mesh, settings.compressmore);
        }
        _ => {}
    }
}

fn get_scale(transform: &[f32; 16]) -> f32 {
    let mut translation = [0.0; 3];
    let mut rotation = [0.0; 4];
    let mut scale = [0.0; 3];
    decompose_transform(&mut translation, &mut rotation, &mut scale, transform);

    let (sx, sy, sz) = (scale[0].abs(), scale[1].abs(), scale[2].abs());
    fmax(fmax(sx, sy), sz)
}

pub fn compute_mesh_quality(meshes: &mut [Mesh], data: &Data) {
    let mut scales = vec![1.0f32; meshes.len()];
    let mut maxscale = 0.0f32;

    for (i, mesh) in meshes.iter().enumerate() {
        let Some(positions) = get_stream(mesh, AttributeType::Position, 0) else {
            continue;
        };

        let positions = &mesh.streams[positions].data;
        let geometry_scale = mo::simplify_scale(attr_floats(positions), positions.len(), 16).unwrap_or(0.0);
        let mut node_maxscale = 0.0f32;

        for &node in &mesh.nodes {
            let transform = cgltf::node_transform_world(&data.nodes, &data.nodes[node]);

            node_maxscale = fmax(node_maxscale, get_scale(&transform));
        }

        for xf in &mesh.instances {
            node_maxscale = fmax(node_maxscale, get_scale(&xf.transform));
        }

        scales[i] = if node_maxscale == 0.0 {
            geometry_scale
        } else {
            node_maxscale * geometry_scale
        };
        maxscale = fmax(maxscale, scales[i]);
    }

    for (mesh, &scale) in meshes.iter_mut().zip(&scales) {
        mesh.quality = if scale == 0.0 || maxscale == 0.0 { 1.0 } else { scale / maxscale };
    }
}

pub fn has_vertex_alpha(mesh: &Mesh) -> bool {
    match get_stream(mesh, AttributeType::Color, 0) {
        Some(c) => mesh.streams[c].data.iter().any(|a| a[3] < 1.0),
        None => false,
    }
}

pub fn has_instance_alpha(instances: &[Instance]) -> bool {
    instances.iter().any(|i| i.color[3] < 1.0)
}
