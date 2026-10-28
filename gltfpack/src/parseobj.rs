//! OBJ input (`parseobj.cpp`): fast_obj mesh → cgltf scene + gltfpack meshes.

use crate::cgltf::{self, AlphaMode, AttributeType, Data, PrimitiveType};
use crate::fast_obj::{self, FastObjMesh};
use crate::types::*;
use meshoptimizer as mo;

fn texture_index(textures: &[u32], name: u32) -> Option<usize> {
    textures.iter().position(|&t| t == name)
}

/// `strdup` of a fast_obj byte string (stops at the first NUL).
fn c_string(s: &[u8]) -> String {
    let end = s.iter().position(|&c| c == 0).unwrap_or(s.len());
    String::from_utf8_lossy(&s[..end]).into_owned()
}

fn texture_name(obj: &FastObjMesh, id: u32) -> String {
    obj.textures
        .get(id as usize)
        .and_then(|t| t.name.as_deref())
        .map_or_else(String::new, c_string)
}

fn parse_materials_obj(obj: &FastObjMesh, data: &mut Data) {
    // every texture in obj has a unique id (1+); we convert it to a 0-based index
    // this effectively extracts only used textures out of the obj, as we don't remove unused textures later in the processing
    let mut textures: Vec<u32> = Vec::new();

    for om in &obj.materials {
        if om.map_Kd != 0 && texture_index(&textures, om.map_Kd).is_none() {
            textures.push(om.map_Kd);
        }
    }

    data.images = textures
        .iter()
        .map(|&id| cgltf::Image {
            // Some .obj paths come with back slashes, that are invalid as URI separators and won't open on macOS/Linux when embedding textures
            uri: Some(texture_name(obj, id).replace('\\', "/")),
            ..Default::default()
        })
        .collect();

    data.textures = (0..textures.len())
        .map(|i| cgltf::Texture {
            image: Some(i),
            ..Default::default()
        })
        .collect();

    data.materials = obj
        .materials
        .iter()
        .map(|om| {
            let mut gm = cgltf::Material {
                name: om.name.as_deref().map(c_string),
                has_pbr_metallic_roughness: true,
                alpha_cutoff: 0.5,
                ..Default::default()
            };

            gm.pbr_metallic_roughness.base_color_factor = [1.0; 4];
            gm.pbr_metallic_roughness.metallic_factor = 0.0;
            gm.pbr_metallic_roughness.roughness_factor = 1.0;

            if om.map_Kd != 0 {
                gm.pbr_metallic_roughness.base_color_texture.texture = texture_index(&textures, om.map_Kd);
                gm.pbr_metallic_roughness.base_color_texture.scale = 1.0;

                gm.alpha_mode = if matches!(om.illum, 4 | 6 | 7 | 9) {
                    AlphaMode::Mask
                } else {
                    AlphaMode::Opaque
                };
            } else {
                gm.pbr_metallic_roughness.base_color_factor[..3].copy_from_slice(&om.Kd);
            }

            if om.map_d != 0 {
                if om.map_Kd != 0 && texture_name(obj, om.map_Kd) != texture_name(obj, om.map_d) {
                    eprintln!(
                        "Warning: material has different diffuse and alpha textures (Kd: {}, d: {}) and might not render correctly",
                        texture_name(obj, om.map_Kd),
                        texture_name(obj, om.map_d)
                    );
                }

                gm.alpha_mode = AlphaMode::Blend;
            } else if om.d < 1.0 {
                gm.pbr_metallic_roughness.base_color_factor[3] = om.d;

                gm.alpha_mode = AlphaMode::Blend;
            }

            gm
        })
        .collect();
}

fn parse_nodes_obj(obj: &FastObjMesh, data: &mut Data) {
    data.nodes = obj
        .objects
        .iter()
        .map(|og| {
            let mut node = cgltf::Node {
                name: og.name.as_deref().map(c_string),
                ..Default::default()
            };

            node.rotation[3] = 1.0;
            node.scale = [1.0; 3];
            node
        })
        .collect();

    data.scenes = vec![cgltf::Scene {
        nodes: (0..obj.objects.len()).collect(),
        ..Default::default()
    }];

    data.scene = Some(0);
}

fn index_bytes(indices: &[fast_obj::FastObjIndex]) -> Vec<u8> {
    indices
        .iter()
        .flat_map(|i| [i.p.to_le_bytes(), i.t.to_le_bytes(), i.n.to_le_bytes()].concat())
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn parse_mesh_obj(
    obj: &FastObjMesh,
    face_offset: u32,
    face_vertex_offset: u32,
    face_count: u32,
    face_vertex_count: u32,
    index_count: u32,
    mesh: &mut Mesh,
) {
    let fvo = face_vertex_offset as usize;
    let fvc = face_vertex_count as usize;

    let corners = obj.indices.get(fvo..fvo + fvc).unwrap_or(&[]);

    let mut remap = vec![0u32; fvc];
    let unique_vertices = mo::generate_vertex_remap::<u32>(&mut remap, None, fvc, &index_bytes(corners), fvc, 12).unwrap_or(0);

    let pos_stream = 0;
    let nrm_stream: i32 = if obj.normal_count > 1 { 1 } else { -1 };
    let tex_stream: i32 = if obj.texcoord_count > 1 {
        1 + i32::from(nrm_stream >= 0)
    } else {
        -1
    };
    let col_stream: i32 = if obj.color_count > 1 {
        1 + i32::from(nrm_stream >= 0) + i32::from(tex_stream >= 0)
    } else {
        -1
    };

    let stream_count = 1 + usize::from(nrm_stream >= 0) + usize::from(tex_stream >= 0) + usize::from(col_stream >= 0);
    mesh.streams = vec![Stream::default(); stream_count];

    mesh.streams[pos_stream].type_ = AttributeType::Position;
    mesh.streams[pos_stream].data = vec![[0.0; 4]; unique_vertices];

    for (s, type_) in [
        (nrm_stream, AttributeType::Normal),
        (tex_stream, AttributeType::Texcoord),
        (col_stream, AttributeType::Color),
    ] {
        if s >= 0 {
            mesh.streams[s as usize].type_ = type_;
            mesh.streams[s as usize].data = vec![[0.0; 4]; unique_vertices];
        }
    }

    mesh.indices = vec![0; index_count as usize];

    // out of range OBJ indices (C++ undefined behavior) read as zero
    let pos = |i: usize| obj.positions.get(i).copied().unwrap_or(0.0);
    let nrm = |i: usize| obj.normals.get(i).copied().unwrap_or(0.0);
    let tex = |i: usize| obj.texcoords.get(i).copied().unwrap_or(0.0);
    let col = |i: usize| obj.colors.get(i).copied().unwrap_or(0.0);

    for (vi, ii) in corners.iter().enumerate() {
        let target = remap[vi] as usize;
        // TODO: this fills every target vertex multiple times

        let (p, t, n) = (ii.p as usize, ii.t as usize, ii.n as usize);

        if let Some(d) = mesh.streams[pos_stream].data.get_mut(target) {
            *d = [pos(p * 3), pos(p * 3 + 1), pos(p * 3 + 2), 0.0];
        }

        if nrm_stream >= 0
            && let Some(d) = mesh.streams[nrm_stream as usize].data.get_mut(target)
        {
            *d = [nrm(n * 3), nrm(n * 3 + 1), nrm(n * 3 + 2), 0.0];
        }

        if tex_stream >= 0
            && let Some(d) = mesh.streams[tex_stream as usize].data.get_mut(target)
        {
            *d = [tex(t * 2), 1.0 - tex(t * 2 + 1), 0.0, 0.0];
        }

        if col_stream >= 0
            && let Some(d) = mesh.streams[col_stream as usize].data.get_mut(target)
        {
            *d = [col(p * 3), col(p * 3 + 1), col(p * 3 + 2), 0.0];
        }
    }

    let mut vertex_offset = 0usize;
    let mut index_offset = 0usize;

    for fi in 0..face_count as usize {
        let face_vertices = obj.face_vertices.get(face_offset as usize + fi).copied().unwrap_or(0) as usize;

        let mut set = |to: usize, v: usize| {
            if let (Some(d), Some(&r)) = (mesh.indices.get_mut(to), remap.get(v)) {
                *d = r;
            }
        };

        if mesh.type_ == PrimitiveType::Lines {
            for vi in 1..face_vertices {
                let to = index_offset + (vi - 1) * 2;

                set(to, vertex_offset + vi - 1);
                set(to + 1, vertex_offset + vi);
            }

            vertex_offset += face_vertices;
            index_offset = index_offset.wrapping_add(face_vertices.wrapping_sub(1).wrapping_mul(2));
        } else {
            for vi in 2..face_vertices {
                let to = index_offset + (vi - 2) * 3;

                set(to, vertex_offset);
                set(to + 1, vertex_offset + vi - 1);
                set(to + 2, vertex_offset + vi);
            }

            vertex_offset += face_vertices;
            index_offset = index_offset.wrapping_add(face_vertices.wrapping_sub(2).wrapping_mul(3));
        }
    }
}

fn parse_mesh_group_obj(obj: &FastObjMesh, og: &fast_obj::FastObjGroup, data: &Data, node: Option<usize>, meshes: &mut Vec<Mesh>) {
    let mut face_vertex_offset = og.index_offset;
    let face_end_offset = og.face_offset + og.face_count;

    let face_material = |f: u32| obj.face_materials.get(f as usize).copied().unwrap_or(0);
    let face_line = |f: u32| {
        if obj.face_lines.is_empty() {
            0
        } else {
            obj.face_lines.get(f as usize).copied().unwrap_or(0)
        }
    };
    let face_vertices = |f: u32| obj.face_vertices.get(f as usize).copied().unwrap_or(0);

    let mut face_offset = og.face_offset;
    while face_offset < face_end_offset {
        let mi = face_material(face_offset);
        let isl = face_line(face_offset);

        let mut face_count: u32 = 0;
        let mut face_vertex_count: u32 = 0;
        let mut index_count: u32 = 0;

        let mut fj = face_offset;
        while fj < face_end_offset && face_material(fj) == mi && (obj.face_lines.is_empty() || face_line(fj) == isl) {
            face_count += 1;
            face_vertex_count = face_vertex_count.wrapping_add(face_vertices(fj));
            index_count = index_count.wrapping_add(if isl != 0 {
                face_vertices(fj).wrapping_sub(1).wrapping_mul(2)
            } else {
                face_vertices(fj).wrapping_sub(2).wrapping_mul(3)
            });
            fj += 1;
        }

        // faces with too few vertices make the C++ index count wrap around (a multi-GB allocation that aborts);
        // skip such runs instead
        if u64::from(index_count) > 3 * u64::from(face_vertex_count) {
            eprintln!("Warning: ignoring faces with too few vertices");
            face_offset += face_count;
            face_vertex_offset = face_vertex_offset.wrapping_add(face_vertex_count);
            continue;
        }

        let mut mesh = Mesh {
            material: if data.materials.is_empty() { None } else { Some(mi as usize) },
            type_: if isl != 0 { PrimitiveType::Lines } else { PrimitiveType::Triangles },
            targets: 0,
            ..Default::default()
        };

        if let Some(node) = node {
            mesh.nodes.push(node);
        }

        parse_mesh_obj(
            obj,
            face_offset,
            face_vertex_offset,
            face_count,
            face_vertex_count,
            index_count,
            &mut mesh,
        );

        meshes.push(mesh);

        face_offset += face_count;
        face_vertex_offset = face_vertex_offset.wrapping_add(face_vertex_count);
    }
}

/// `parseObj`: returns the scene or an error message.
pub fn parse_obj(path: &str, meshes: &mut Vec<Mesh>) -> Result<Data, &'static str> {
    let obj = fast_obj::fast_obj_read(path).ok_or("file not found")?;

    let fallback_materials: i32 = obj.materials.iter().map(|m| m.fallback).sum();

    if fallback_materials != 0 {
        eprintln!(
            "Warning: {fallback_materials}/{} materials could not be loaded from mtllib",
            obj.material_count
        );
    }

    let mut data = Data::default();

    parse_materials_obj(&obj, &mut data);

    parse_nodes_obj(&obj, &mut data);

    for (oi, og) in obj.objects.iter().enumerate() {
        parse_mesh_group_obj(&obj, og, &data, Some(oi), meshes);
    }

    fast_obj::fast_obj_destroy(obj);

    Ok(data)
}
