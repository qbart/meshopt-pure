//! gltfpack pipeline and command line interface (`gltfpack.cpp`).

use std::collections::HashMap;

use crate::animation::process_animation;
use crate::cgltf::{self, AlphaMode, Data, PrimitiveType};
use crate::cstd::{atof, atoi, format_f};
use crate::fileio::get_extension;
use crate::json::*;
use crate::material::*;
use crate::mesh::*;
use crate::node::*;
use crate::parsegltf::parse_gltf;
use crate::parseobj::parse_obj;
use crate::splats::{parse_splats, process_splat_mesh, splat_attribute_type};
use crate::stream::*;
use crate::types::*;
use crate::write::*;
use meshoptimizer as mo;

pub fn get_version() -> String {
    format!("{}.{}", mo::MESHOPTIMIZER_VERSION / 1000, (mo::MESHOPTIMIZER_VERSION % 1000) / 10)
}

fn finalize_buffer_views(
    json: &mut String,
    views: &mut [BufferView],
    bin: &mut Vec<u8>,
    mut fallback: Option<&mut Vec<u8>>,
    fallback_size: &mut usize,
    meshopt_ext: &str,
    attribute_level: i32,
) {
    for view in views.iter_mut() {
        let bin_offset = bin.len();
        let fallback_offset = *fallback_size;

        let count = view.data.len().checked_div(view.stride).unwrap_or(0);

        if view.compression == Compression::None {
            bin.extend_from_slice(&view.data);
        } else {
            match view.compression {
                Compression::Attribute => compress_vertex_stream(bin, &view.data, count, view.stride, attribute_level),
                Compression::Index => compress_index_stream(bin, &view.data, count, view.stride),
                Compression::IndexSequence => compress_index_sequence(bin, &view.data, count, view.stride),
                Compression::None => {}
            }

            if let Some(fb) = fallback.as_deref_mut() {
                fb.extend_from_slice(&view.data);
            }
            *fallback_size += view.data.len();
        }

        let raw_offset = if view.compression != Compression::None {
            fallback_offset
        } else {
            bin_offset
        };

        comma(json);
        write_buffer_view(
            json,
            view.kind,
            view.filter,
            count,
            view.stride,
            raw_offset,
            view.data.len(),
            view.compression,
            bin_offset,
            bin.len() - bin_offset,
            meshopt_ext,
        );

        // record written bytes for statistics
        view.bytes = bin.len() - bin_offset;

        // align each bufferView by 4 bytes
        bin.resize((bin.len() + 3) & !3, 0);
        if let Some(fb) = fallback.as_deref_mut() {
            fb.resize((fb.len() + 3) & !3, 0);
        }
        *fallback_size = (*fallback_size + 3) & !3;
    }
}

struct DrawStats {
    triangles: usize,
    instances: usize,
    draws: usize,
}

fn draw_stats(meshes: &[Mesh]) -> DrawStats {
    let mut stats = DrawStats {
        triangles: 0,
        instances: 0,
        draws: 0,
    };

    for mesh in meshes {
        let triangles = if mesh.type_ == PrimitiveType::Triangles {
            mesh.indices.len() / 3
        } else {
            0
        };
        let instances = (mesh.nodes.len() + mesh.instances.len()).max(1);

        stats.triangles += triangles * instances;
        stats.instances += instances;
        stats.draws += mesh.nodes.len().max(1);
    }

    stats
}

fn print_mesh_stats(meshes: &[Mesh], name: &str) {
    let mut mesh_triangles = 0usize;
    let mut mesh_vertices = 0usize;

    for mesh in meshes {
        mesh_triangles += if mesh.type_ == PrimitiveType::Triangles {
            mesh.indices.len() / 3
        } else {
            0
        };
        mesh_vertices += mesh.streams.first().map_or(0, |s| s.data.len());
    }

    let stats = draw_stats(meshes);

    println!(
        "{name}: {} mesh primitives ({} triangles, {} vertices); {} draw calls ({} instances, {} triangles)",
        meshes.len() as i32,
        mesh_triangles as i32,
        mesh_vertices as i32,
        stats.draws as i32,
        stats.instances as i32,
        stats.triangles as i64
    );
}

fn bytes_by_kind(views: &[BufferView]) -> [usize; BUFFER_VIEW_KIND_COUNT] {
    let mut bytes = [0usize; BUFFER_VIEW_KIND_COUNT];

    for view in views {
        bytes[view.kind as usize] += view.bytes;
    }

    bytes
}

fn print_scene_stats(
    views: &[BufferView],
    meshes: &[Mesh],
    node_offset: usize,
    mesh_offset: usize,
    material_offset: usize,
    json_size: usize,
    bin_size: usize,
) {
    let bytes = bytes_by_kind(views);

    println!(
        "output: {} nodes, {} meshes ({} primitives), {} materials",
        node_offset as i32,
        mesh_offset as i32,
        meshes.len() as i32,
        material_offset as i32
    );
    println!("output: JSON {} bytes, buffers {} bytes", json_size as i32, bin_size as i32);
    println!(
        "output: buffers: vertex {} bytes, index {} bytes, skin {} bytes, time {} bytes, keyframe {} bytes, instance {} bytes, image {} bytes, splat {} bytes",
        bytes[BufferViewKind::Vertex as usize] as i32,
        bytes[BufferViewKind::Index as usize] as i32,
        bytes[BufferViewKind::Skin as usize] as i32,
        bytes[BufferViewKind::Time as usize] as i32,
        bytes[BufferViewKind::Keyframe as usize] as i32,
        bytes[BufferViewKind::Instance as usize] as i32,
        bytes[BufferViewKind::Image as usize] as i32,
        bytes[BufferViewKind::Splat as usize] as i32
    );
}

fn variant_name(kind: BufferViewKind, variant: i32) -> &'static str {
    match kind {
        BufferViewKind::Vertex => attribute_type(match variant {
            1 => cgltf::AttributeType::Position,
            2 => cgltf::AttributeType::Normal,
            3 => cgltf::AttributeType::Tangent,
            4 => cgltf::AttributeType::Texcoord,
            5 => cgltf::AttributeType::Color,
            6 => cgltf::AttributeType::Joints,
            7 => cgltf::AttributeType::Weights,
            8 => cgltf::AttributeType::Custom,
            _ => cgltf::AttributeType::Invalid,
        }),
        BufferViewKind::Index => "index",
        BufferViewKind::Keyframe | BufferViewKind::Instance => animation_path(match variant {
            1 => cgltf::AnimationPathType::Translation,
            2 => cgltf::AnimationPathType::Rotation,
            3 => cgltf::AnimationPathType::Scale,
            4 => cgltf::AnimationPathType::Weights,
            _ => cgltf::AnimationPathType::Invalid,
        }),
        BufferViewKind::Splat => splat_attribute_type(variant),
        _ => "unknown",
    }
}

fn print_attribute_stats(views: &[BufferView], kind: BufferViewKind, name: &str) {
    for view in views.iter().filter(|v| v.kind == kind) {
        let variant = variant_name(kind, view.variant);

        let mut count = view.data.len().checked_div(view.stride).unwrap_or(0);

        if kind == BufferViewKind::Splat
            && let Some(degree) = variant.strip_prefix("SH_DEGREE_")
        {
            count /= (1 + atoi(degree.as_bytes()) * 2) as usize; // 1, 3, 5, 7 vectors per degree
        }

        let bits = format_f(view.bytes as f64 / count as f64 * 8.0, 1);

        if view.compression == Compression::None {
            println!("stats: {name} {variant}: {} bytes ({bits} bits)", view.bytes as i32);
        } else {
            println!(
                "stats: {name} {variant}: compressed {} bytes ({bits} bits), raw {} bytes ({} bits)",
                view.bytes as i32,
                view.data.len() as i32,
                (view.data.len() * 8).checked_div(count).unwrap_or(0) as i32
            );
        }
    }
}

fn print_image_stats(views: &[BufferView], kind: TextureKind, name: &str) {
    let mut bytes = 0usize;
    let mut count = 0usize;

    for view in views {
        if view.kind != BufferViewKind::Image || view.variant != -1 - kind as i32 {
            continue;
        }

        count += 1;
        bytes += view.data.len();
    }

    if count != 0 {
        println!("stats: image {name}: {} bytes in {} images", bytes as i32, count as i32);
    }
}

#[allow(clippy::too_many_arguments)]
fn print_report(
    path: &str,
    views: &[BufferView],
    meshes: &[Mesh],
    node_count: usize,
    mesh_count: usize,
    texture_count: usize,
    material_count: usize,
    animation_count: usize,
    json_size: usize,
    bin_size: usize,
) -> bool {
    let bytes = bytes_by_kind(views);
    let stats = draw_stats(meshes);

    let b = |k: BufferViewKind| bytes[k as usize];

    let mut out = String::new();
    out += "{\n";
    out += &format!("\t\"generator\": \"gltfpack {}\",\n", get_version());
    out += "\t\"scene\": {\n";
    out += &format!("\t\t\"nodeCount\": {},\n", node_count as i32);
    out += &format!("\t\t\"meshCount\": {},\n", mesh_count as i32);
    out += &format!("\t\t\"materialCount\": {},\n", material_count as i32);
    out += &format!("\t\t\"textureCount\": {},\n", texture_count as i32);
    out += &format!("\t\t\"animationCount\": {}\n", animation_count as i32);
    out += "\t},\n";
    out += "\t\"render\": {\n";
    out += &format!("\t\t\"drawCount\": {},\n", stats.draws as i32);
    out += &format!("\t\t\"instanceCount\": {},\n", stats.instances as i32);
    out += &format!("\t\t\"triangleCount\": {}\n", stats.triangles as i64);
    out += "\t},\n";
    out += "\t\"data\": {\n";
    out += &format!("\t\t\"json\": {},\n", json_size as i32);
    out += &format!("\t\t\"binary\": {},\n", bin_size as i32);
    out += "\t\t\"buffers\": {\n";
    out += &format!("\t\t\t\"vertex\": {},\n", b(BufferViewKind::Vertex) as i32);
    out += &format!("\t\t\t\"index\": {},\n", b(BufferViewKind::Index) as i32);
    out += &format!(
        "\t\t\t\"animation\": {},\n",
        (b(BufferViewKind::Time) + b(BufferViewKind::Keyframe)) as i32
    );
    out += &format!(
        "\t\t\t\"transform\": {},\n",
        (b(BufferViewKind::Skin) + b(BufferViewKind::Instance)) as i32
    );
    out += &format!("\t\t\t\"image\": {}\n", b(BufferViewKind::Image) as i32);
    out += "\t\t}\n";
    out += "\t}\n";
    out += "}\n";

    std::fs::write(path, out).is_ok()
}

fn can_transform_mesh(data: &Data, mesh: &Mesh) -> bool {
    // volume thickness is specified in mesh coordinate space; to avoid modifying materials we prohibit transforming meshes with volume materials
    match mesh.material.map(|m| &data.materials[m]) {
        Some(material) => !(material.has_volume && material.volume.thickness_factor > 0.0),
        None => true,
    }
}

fn detach_mesh(mesh: &mut Mesh, data: &Data, nodes: &[NodeInfo], settings: &Settings) {
    // mesh is already instanced, skip
    if !mesh.instances.is_empty() {
        return;
    }

    // mesh is already world space, skip
    if mesh.nodes.is_empty() {
        return;
    }

    // note: when -kn is specified, we keep mesh-node attachment so that named nodes can be transformed
    if settings.keep_nodes {
        return;
    }

    // we keep skinned meshes or meshes with morph targets as is
    // in theory we could transform both, but in practice transforming morph target meshes is more involved,
    // and reparenting skinned meshes leads to incorrect bounding box generated in three.js
    if mesh.skin.is_some() || mesh.targets != 0 {
        return;
    }

    // animated meshes will be anchored to the same node that they used to be in to retain the animation
    if mesh.nodes.iter().any(|&n| nodes[n].animated) {
        return;
    }

    let scene = nodes[mesh.nodes[0]].scene;
    let any_other_scene = mesh.nodes.iter().any(|&n| scene != nodes[n].scene);

    // we only merge instances when all nodes have a single consistent scene
    if scene < 0 || any_other_scene {
        return;
    }

    // we only merge multiple instances together if requested
    // this often makes the scenes faster to render by reducing the draw call count, but can result in larger files
    if mesh.nodes.len() > 1 && !settings.mesh_merge && !settings.mesh_instancing {
        return;
    }

    // mesh has duplicate geometry; detaching it would increase the size due to unique world-space transforms
    if mesh.nodes.len() == 1 && mesh.geometry_duplicate && !settings.mesh_merge {
        return;
    }

    // prefer instancing if possible, use merging otherwise
    if mesh.nodes.len() > 1 && settings.mesh_instancing {
        mesh.instances = mesh
            .nodes
            .iter()
            .map(|&n| Instance {
                transform: cgltf::node_transform_world(&data.nodes, &data.nodes[n]),
                color: [1.0; 4],
            })
            .collect();

        mesh.nodes.clear();
        mesh.scene = scene;
    } else if can_transform_mesh(data, mesh) {
        merge_mesh_instances(mesh, data);

        mesh.scene = scene;
    }
}

fn is_extension_supported(extensions: &[ExtensionInfo], name: &str) -> bool {
    extensions.iter().any(|e| e.name == name)
}

#[derive(Clone, Copy, Default)]
struct PrimitiveCacheEntry {
    offset: usize,
    size: usize,
    qt: QuantizationTexture,
}

fn same_quantization(lhs: &QuantizationTexture, rhs: &QuantizationTexture, settings: &Settings) -> bool {
    if !settings.quantize || settings.tex_float {
        return true;
    }

    lhs.offset[0] == rhs.offset[0]
        && lhs.offset[1] == rhs.offset[1]
        && lhs.scale[0] == rhs.scale[0]
        && lhs.scale[1] == rhs.scale[1]
        && lhs.bits == rhs.bits
        && lhs.normalized == rhs.normalized
}

/// Output of `process`: the JSON (without the `buffers` array), binary buffers and the insertion point for `buffers`.
pub struct ProcessOutput {
    pub json: String,
    pub bin: Vec<u8>,
    pub fallback: Vec<u8>,
    pub fallback_size: usize,
    pub bufferspec_pos: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn process(
    data: &mut Data,
    input_path: Option<&str>,
    output_path: Option<&str>,
    report_path: Option<&str>,
    meshes: &mut Vec<Mesh>,
    animations: &mut [Animation],
    settings: &Settings,
    json: &mut String,
    bin: &mut Vec<u8>,
    fallback: &mut Vec<u8>,
    fallback_size: &mut usize,
    meshopt_ext: &str,
) -> usize {
    if settings.verbose != 0 {
        println!(
            "input: {} nodes, {} meshes ({} primitives), {} materials, {} skins, {} animations, {} images",
            data.nodes.len() as i32,
            data.meshes.len() as i32,
            meshes.len() as i32,
            data.materials.len() as i32,
            data.skins.len() as i32,
            animations.len() as i32,
            data.images.len() as i32
        );
        print_mesh_stats(meshes, "input");
    }

    for animation in animations.iter_mut() {
        process_animation(animation, data, settings);
    }

    let mut nodes = vec![NodeInfo::default(); data.nodes.len()];

    mark_scenes(data, &mut nodes);
    mark_animated(data, &mut nodes, animations);

    merge_mesh_materials(data, meshes, settings);
    if settings.mesh_dedup {
        dedup_meshes(meshes, settings);
    }

    for mesh in meshes.iter_mut() {
        detach_mesh(mesh, data, &nodes, settings);
    }

    // material information is required for mesh and image processing
    let mut materials = vec![MaterialInfo::default(); data.materials.len()];
    let mut textures = vec![TextureInfo::default(); data.textures.len()];
    let mut images = vec![ImageInfo::default(); data.images.len()];

    // mark materials that need to keep blending due to mesh vertex colors
    for mesh in meshes.iter() {
        // skip hasAlpha check unless it's required
        let blended = mesh.material.is_some_and(|m| data.materials[m].alpha_mode != AlphaMode::Opaque) || !mesh.variants.is_empty();

        if (blended && has_vertex_alpha(mesh)) || has_instance_alpha(&mesh.instances) {
            if let Some(m) = mesh.material {
                materials[m].mesh_alpha = true;
            }

            for variant in &mesh.variants {
                materials[variant.material].mesh_alpha = true;
            }
        }
    }

    analyze_materials(data, &mut materials, &mut textures, &mut images);

    merge_textures(data, &mut textures);

    optimize_materials(data, &materials, &mut images, input_path);

    // streams need to be filtered before mesh merging (or processing) to make sure we can merge meshes with redundant streams
    for mesh in meshes.iter_mut() {
        let mut mi = mesh.material.map_or_else(MaterialInfo::default, |m| materials[m]);

        // merge material requirements across all variants
        for variant in &mesh.variants {
            let vi = materials[variant.material];

            mi.needs_tangents |= vi.needs_tangents;
            mi.texture_set_mask |= vi.texture_set_mask;
            mi.unlit &= vi.unlit;
        }

        if !settings.keep_attributes {
            filter_streams(mesh, &mi);
        }

        if settings.mesh_normals && (!mi.unlit || settings.keep_attributes) {
            generate_normals(mesh, settings.normals_crease);
        }

        if settings.mesh_tangents && (mi.needs_tangents || settings.keep_attributes) {
            generate_tangents(mesh, data);
        }
    }

    merge_meshes(meshes, data, settings);
    filter_empty_meshes(meshes);

    mark_needed_nodes(data, &mut nodes, meshes, animations, settings);
    mark_needed_materials(data, &mut materials, meshes, settings);

    if settings.simplify_ratio < 1.0 {
        compute_mesh_quality(meshes, data);
    }

    for mesh in meshes.iter_mut() {
        if mesh.splats {
            process_splat_mesh(mesh, settings);
        } else {
            process_mesh(mesh, settings);
        }

        if mesh.geometry_duplicate {
            hash_mesh(mesh);
        }
    }

    filter_empty_meshes(meshes); // some meshes may become empty after processing

    let qp = prepare_quantization_position(meshes, settings);

    let mut qt_materials = vec![QuantizationTexture::default(); materials.len()];
    let mut qt_meshes = vec![usize::MAX; meshes.len()];
    prepare_quantization_texture(data, &mut qt_materials, &mut qt_meshes, meshes, settings);

    let qt_dummy = QuantizationTexture {
        bits: settings.tex_bits,
        ..Default::default()
    };

    let mut json_images = String::new();
    let mut json_samplers = String::new();
    let mut json_textures = String::new();
    let mut json_materials = String::new();
    let mut json_accessors = String::new();
    let mut json_meshes = String::new();
    let mut json_nodes = String::new();
    let mut json_skins = String::new();
    let mut json_roots = vec![String::new(); data.scenes.len()];
    let mut json_animations = String::new();
    let mut json_cameras = String::new();
    let mut json_extensions = String::new();

    let mut views: Vec<BufferView> = Vec::new();

    let mut ext_pbr_specular_glossiness = false;
    let mut ext_clearcoat = false;
    let mut ext_transmission = false;
    let mut ext_ior = false;
    let mut ext_specular = false;
    let mut ext_sheen = false;
    let mut ext_volume = false;
    let mut ext_emissive_strength = false;
    let mut ext_iridescence = false;
    let mut ext_anisotropy = false;
    let mut ext_dispersion = false;
    let mut ext_diffuse_transmission = false;
    let mut ext_unlit = false;
    let mut ext_instancing = false;
    let mut ext_texture_transform = false;
    let mut ext_texture_basisu = false;
    let mut ext_texture_webp = false;
    let mut ext_splatting = false;
    let mut ext_quantization = false;

    let mut accr_offset = 0usize;
    let mut node_offset = 0usize;
    let mut mesh_offset = 0usize;
    let mut texture_offset = 0usize;
    let mut material_offset = 0usize;

    for sampler in &data.samplers {
        comma(&mut json_samplers);
        append(&mut json_samplers, "{");
        write_sampler(&mut json_samplers, sampler);
        append(&mut json_samplers, "}");
    }

    let mut encoded_images: Vec<Vec<u8>> = vec![Vec::new(); data.images.len()];

    if !data.images.is_empty() && settings.texture_ktx2 {
        crate::encodebasis::encode_images_basis(&mut encoded_images, data, &images, input_path, settings);
    }

    for (i, image) in data.images.iter().enumerate() {
        let encoded = if encoded_images[i].is_empty() {
            None
        } else {
            Some(&encoded_images[i][..])
        };

        comma(&mut json_images);
        append(&mut json_images, "{");
        write_image(
            &mut json_images,
            &mut views,
            data,
            image,
            &images[i],
            encoded,
            i,
            input_path,
            output_path,
            settings,
        );
        append(&mut json_images, "}");

        if encoded.is_some() {
            encoded_images[i] = Vec::new(); // reclaim memory early
        }
    }

    for (i, texture) in data.textures.iter().enumerate() {
        if !textures[i].keep {
            continue;
        }

        comma(&mut json_textures);
        append(&mut json_textures, "{");
        write_texture(&mut json_textures, texture, texture.image.map(|im| &images[im]), settings);
        append(&mut json_textures, "}");

        texture_offset += 1;
        ext_texture_basisu = ext_texture_basisu || texture.has_basisu;
        ext_texture_webp = ext_texture_webp || texture.has_webp;
    }

    for (i, material) in data.materials.iter().enumerate() {
        let mi = &mut materials[i];

        if !mi.keep {
            continue;
        }

        comma(&mut json_materials);
        append(&mut json_materials, "{");
        write_material(
            &mut json_materials,
            material,
            if settings.quantize && !settings.pos_float {
                Some(&qp)
            } else {
                None
            },
            if settings.quantize && !settings.tex_float {
                Some(&qt_materials[i])
            } else {
                None
            },
            &textures,
        );
        if settings.keep_extras {
            write_extras(&mut json_materials, &material.extras);
        }
        append(&mut json_materials, "}");

        mi.remap = material_offset as i32;
        material_offset += 1;

        ext_pbr_specular_glossiness = ext_pbr_specular_glossiness || material.has_pbr_specular_glossiness;
        ext_clearcoat = ext_clearcoat || material.has_clearcoat;
        ext_transmission = ext_transmission || material.has_transmission;
        ext_ior = ext_ior || material.has_ior;
        ext_specular = ext_specular || material.has_specular;
        ext_sheen = ext_sheen || material.has_sheen;
        ext_volume = ext_volume || material.has_volume;
        ext_emissive_strength = ext_emissive_strength || material.has_emissive_strength;
        ext_iridescence = ext_iridescence || material.has_iridescence;
        ext_diffuse_transmission = ext_diffuse_transmission || material.has_diffuse_transmission;
        ext_anisotropy = ext_anisotropy || material.has_anisotropy;
        ext_dispersion = ext_dispersion || material.has_dispersion;
        ext_unlit = ext_unlit || material.unlit;
        ext_texture_transform = ext_texture_transform || mi.uses_texture_transform;
    }

    let mut primitive_cache: HashMap<[u64; 2], PrimitiveCacheEntry> = HashMap::new();

    let mut i = 0;
    while i < meshes.len() {
        let mesh = &meshes[i];

        comma(&mut json_meshes);
        append(&mut json_meshes, "{\"primitives\":[");

        let mut pi = i;
        while pi < meshes.len() {
            let prim = &meshes[pi];

            if prim.scene != mesh.scene || prim.skin != mesh.skin || prim.targets != mesh.targets {
                break;
            }

            if pi > i && (!mesh.instances.is_empty() || !prim.instances.is_empty()) {
                break;
            }

            if !compare_mesh_nodes(mesh, prim) {
                break;
            }

            if !compare_mesh_targets(mesh, prim) {
                break;
            }

            let qt = if qt_meshes[pi] == usize::MAX {
                &qt_dummy
            } else {
                &qt_materials[qt_meshes[pi]]
            };

            comma(&mut json_meshes);

            if prim.splats {
                write_mesh_splats(
                    &mut json_meshes,
                    &mut views,
                    &mut json_accessors,
                    &mut accr_offset,
                    prim,
                    &qp,
                    settings,
                );
                ext_splatting = true;
            } else if prim.geometry_duplicate {
                let entry = primitive_cache.entry(prim.geometry_hash).or_default();

                if entry.size != 0 && same_quantization(&entry.qt, qt, settings) {
                    // reuse previously written accessors
                    let reused = json_meshes[entry.offset..entry.offset + entry.size].to_string();
                    json_meshes.push_str(&reused);
                } else {
                    entry.offset = json_meshes.len();
                    write_mesh_geometry(
                        &mut json_meshes,
                        &mut views,
                        &mut json_accessors,
                        &mut accr_offset,
                        prim,
                        &qp,
                        qt,
                        settings,
                    );
                    ext_quantization = true;
                    entry.size = json_meshes.len() - entry.offset;
                    entry.qt = *qt;
                }
            } else {
                write_mesh_geometry(
                    &mut json_meshes,
                    &mut views,
                    &mut json_accessors,
                    &mut accr_offset,
                    prim,
                    &qp,
                    qt,
                    settings,
                );
                ext_quantization = true;
            }

            if let Some(m) = prim.material {
                append(&mut json_meshes, ",\"material\":");
                append_size(&mut json_meshes, materials[m].remap as isize as usize);
            }

            if !prim.variants.is_empty() {
                append(&mut json_meshes, ",\"extensions\":{\"KHR_materials_variants\":{\"mappings\":[");

                for variant in &prim.variants {
                    comma(&mut json_meshes);
                    append(&mut json_meshes, "{\"material\":");
                    append_size(&mut json_meshes, materials[variant.material].remap as isize as usize);
                    append(&mut json_meshes, ",\"variants\":[");
                    append_size(&mut json_meshes, variant.variant);
                    append(&mut json_meshes, "]}");
                }

                append(&mut json_meshes, "]}}");
            }

            if settings.keep_extras {
                write_extras(&mut json_meshes, &prim.extras);
            }

            append(&mut json_meshes, "}");

            pi += 1;
        }

        append(&mut json_meshes, "]");

        if !mesh.target_weights.is_empty() {
            append(&mut json_meshes, ",\"weights\":");
            append_floats(&mut json_meshes, &mesh.target_weights);
        }

        if !mesh.target_names.is_empty() {
            append(&mut json_meshes, ",\"extras\":{\"targetNames\":[");
            for name in &mesh.target_names {
                comma(&mut json_meshes);
                append(&mut json_meshes, "\"");
                append(&mut json_meshes, name);
                append(&mut json_meshes, "\"");
            }
            append(&mut json_meshes, "]}");
        }

        append(&mut json_meshes, "}");

        for &node in &mesh.nodes {
            let ni = &mut nodes[node];

            // if we don't use position quantization, prefer attaching the mesh to its node directly
            if !ni.has_mesh
                && (!settings.quantize
                    || settings.pos_float
                    || (qp.offset[0] == 0.0 && qp.offset[1] == 0.0 && qp.offset[2] == 0.0 && qp.node_scale == 1.0))
            {
                ni.has_mesh = true;
                ni.mesh_index = mesh_offset;
                ni.mesh_skin = mesh.skin;
            } else {
                ni.mesh_nodes.push(node_offset);

                write_mesh_node(
                    &mut json_nodes,
                    mesh_offset,
                    Some(&data.nodes[node]),
                    mesh.skin,
                    if settings.quantize && !settings.pos_float {
                        Some(&qp)
                    } else {
                        None
                    },
                );

                node_offset += 1;
            }
        }

        if !mesh.instances.is_empty() {
            let scene = mesh.scene.max(0) as usize;
            if let Some(roots) = json_roots.get_mut(scene) {
                comma(roots);
                append_size(roots, node_offset);
            }

            let has_color = mesh.instances.iter().any(|instance| instance.color != [1.0; 4]);

            let instance_accr = write_instances(
                &mut views,
                &mut json_accessors,
                &mut accr_offset,
                &mesh.instances,
                &qp,
                has_color,
                settings,
            );

            write_mesh_node_instanced(&mut json_nodes, mesh_offset, instance_accr, has_color);

            node_offset += 1;
        }

        if mesh.nodes.is_empty() && mesh.instances.is_empty() {
            let scene = mesh.scene.max(0) as usize;
            if let Some(roots) = json_roots.get_mut(scene) {
                comma(roots);
                append_size(roots, node_offset);
            }

            write_mesh_node(
                &mut json_nodes,
                mesh_offset,
                None,
                mesh.skin,
                if settings.quantize && !settings.pos_float {
                    Some(&qp)
                } else {
                    None
                },
            );

            node_offset += 1;
        }

        mesh_offset += 1;
        ext_instancing = ext_instancing || !mesh.instances.is_empty();

        // skip all meshes that we've written in this iteration
        i = pi.max(i + 1);
    }

    remap_nodes(data, &mut nodes, &mut node_offset);

    for (i, node) in data.nodes.iter().enumerate() {
        if !nodes[i].keep {
            continue;
        }

        comma(&mut json_nodes);
        append(&mut json_nodes, "{");
        write_node(&mut json_nodes, node, i, &nodes);
        if settings.keep_extras {
            write_extras(&mut json_nodes, &node.extras);
        }
        append(&mut json_nodes, "}");
    }

    for (i, scene) in data.scenes.iter().enumerate() {
        for &node in &scene.nodes {
            let ni = &nodes[node];

            if ni.keep {
                comma(&mut json_roots[i]);
                append_size(&mut json_roots[i], ni.remap as isize as usize);
            }
        }
    }

    for skin in &data.skins {
        let matrix_accr = write_joint_bind_matrices(&mut views, &mut json_accessors, &mut accr_offset, data, skin, &qp, settings);

        write_skin(&mut json_skins, skin, matrix_accr, &nodes);
    }

    for (i, animation) in animations.iter().enumerate() {
        write_animation(
            &mut json_animations,
            &mut views,
            &mut json_accessors,
            &mut accr_offset,
            animation,
            i,
            &nodes,
            settings,
        );
    }

    for camera in &data.cameras {
        write_camera(&mut json_cameras, camera);
    }

    if !data.lights.is_empty() {
        comma(&mut json_extensions);
        append(&mut json_extensions, "\"KHR_lights_punctual\":{\"lights\":[");

        for light in &data.lights {
            write_light(&mut json_extensions, light);
        }

        append(&mut json_extensions, "]}");
    }

    if !data.variants.is_empty() {
        comma(&mut json_extensions);
        append(&mut json_extensions, "\"KHR_materials_variants\":{\"variants\":[");

        for variant in &data.variants {
            comma(&mut json_extensions);
            append(&mut json_extensions, "{\"name\":\"");
            append(&mut json_extensions, variant.name.as_deref().unwrap_or(""));
            append(&mut json_extensions, "\"}");
        }

        append(&mut json_extensions, "]}");
    }

    append(json, "\"asset\":{");
    append(json, "\"version\":\"2.0\",\"generator\":\"gltfpack ");
    append(json, &get_version());
    append(json, "\"");
    write_extras(json, &data.asset.extras);
    append(json, "}");

    let extensions = [
        ExtensionInfo {
            name: "KHR_mesh_quantization",
            used: settings.quantize && ext_quantization,
            required: true,
        },
        ExtensionInfo {
            name: meshopt_ext,
            used: settings.compress,
            required: !settings.fallback,
        },
        ExtensionInfo {
            name: "KHR_texture_transform",
            used: (settings.quantize && !settings.tex_float && !json_textures.is_empty()) || ext_texture_transform,
            required: false,
        },
        ExtensionInfo {
            name: "KHR_materials_pbrSpecularGlossiness",
            used: ext_pbr_specular_glossiness,
            required: false,
        },
        ExtensionInfo {
            name: "KHR_materials_clearcoat",
            used: ext_clearcoat,
            required: false,
        },
        ExtensionInfo {
            name: "KHR_materials_transmission",
            used: ext_transmission,
            required: false,
        },
        ExtensionInfo {
            name: "KHR_materials_ior",
            used: ext_ior,
            required: false,
        },
        ExtensionInfo {
            name: "KHR_materials_specular",
            used: ext_specular,
            required: false,
        },
        ExtensionInfo {
            name: "KHR_materials_sheen",
            used: ext_sheen,
            required: false,
        },
        ExtensionInfo {
            name: "KHR_materials_volume",
            used: ext_volume,
            required: false,
        },
        ExtensionInfo {
            name: "KHR_materials_emissive_strength",
            used: ext_emissive_strength,
            required: false,
        },
        ExtensionInfo {
            name: "KHR_materials_iridescence",
            used: ext_iridescence,
            required: false,
        },
        ExtensionInfo {
            name: "KHR_materials_anisotropy",
            used: ext_anisotropy,
            required: false,
        },
        ExtensionInfo {
            name: "KHR_materials_dispersion",
            used: ext_dispersion,
            required: false,
        },
        ExtensionInfo {
            name: "KHR_materials_diffuse_transmission",
            used: ext_diffuse_transmission,
            required: false,
        },
        ExtensionInfo {
            name: "KHR_materials_unlit",
            used: ext_unlit,
            required: false,
        },
        ExtensionInfo {
            name: "KHR_materials_variants",
            used: !data.variants.is_empty(),
            required: false,
        },
        ExtensionInfo {
            name: "KHR_lights_punctual",
            used: !data.lights.is_empty(),
            required: false,
        },
        ExtensionInfo {
            name: "KHR_texture_basisu",
            used: (!json_textures.is_empty() && settings.texture_ktx2) || ext_texture_basisu,
            required: true,
        },
        ExtensionInfo {
            name: "KHR_gaussian_splatting",
            used: ext_splatting,
            required: true,
        },
        ExtensionInfo {
            name: "EXT_texture_webp",
            used: (!json_textures.is_empty() && settings.texture_webp) || ext_texture_webp,
            required: true,
        },
        ExtensionInfo {
            name: "EXT_mesh_gpu_instancing",
            used: ext_instancing,
            required: true,
        },
    ];

    for ext in &data.extensions_required {
        if !is_extension_supported(&extensions, ext) && !ext.contains("_meshopt_compression") {
            eprintln!("Warning: required extension {ext} is not supported and will be skipped");
        }
    }

    write_extensions(json, &extensions);

    // buffers[] array to be inserted by the caller
    let bufferspec_pos = json.len();

    let mut json_views = String::new();
    let attribute_level = if settings.compresskhr {
        if settings.compressmore { 3 } else { 2 }
    } else {
        0
    };
    finalize_buffer_views(
        &mut json_views,
        &mut views,
        bin,
        if settings.fallback { Some(fallback) } else { None },
        fallback_size,
        meshopt_ext,
        attribute_level,
    );

    write_array(json, "bufferViews", &json_views);
    write_array(json, "accessors", &json_accessors);
    write_array(json, "samplers", &json_samplers);
    write_array(json, "images", &json_images);
    write_array(json, "textures", &json_textures);
    write_array(json, "materials", &json_materials);
    write_array(json, "meshes", &json_meshes);
    write_array(json, "skins", &json_skins);
    write_array(json, "animations", &json_animations);
    write_array(json, "nodes", &json_nodes);

    if !json_roots.is_empty() {
        append(json, ",\"scenes\":[");

        for (scene, roots) in data.scenes.iter().zip(&json_roots) {
            write_scene(json, scene, roots, settings);
        }

        append(json, "]");
    }

    write_array(json, "cameras", &json_cameras);

    if let Some(scene) = data.scene {
        append(json, ",\"scene\":");
        append_size(json, scene);
    }

    if !json_extensions.is_empty() {
        append(json, ",\"extensions\":{");
        append(json, &json_extensions);
        append(json, "}");
    }

    if settings.verbose != 0 {
        print_mesh_stats(meshes, "output");
        print_scene_stats(&views, meshes, node_offset, mesh_offset, material_offset, json.len(), bin.len());
    }

    if settings.verbose > 1 {
        print_attribute_stats(&views, BufferViewKind::Vertex, "vertex");
        print_attribute_stats(&views, BufferViewKind::Index, "index");
        print_attribute_stats(&views, BufferViewKind::Keyframe, "keyframe");
        print_attribute_stats(&views, BufferViewKind::Instance, "instance");
        print_attribute_stats(&views, BufferViewKind::Splat, "splat");

        print_image_stats(&views, TextureKind::Generic, "generic");
        print_image_stats(&views, TextureKind::Color, "color");
        print_image_stats(&views, TextureKind::Normal, "normal");
        print_image_stats(&views, TextureKind::Attrib, "attrib");
    }

    if let Some(report_path) = report_path
        && !print_report(
            report_path,
            &views,
            meshes,
            node_offset,
            mesh_offset,
            texture_offset,
            material_offset,
            animations.len(),
            json.len(),
            bin.len(),
        )
    {
        eprintln!("Warning: cannot save report to {report_path}");
    }

    bufferspec_pos
}

fn get_base_name(path: &str) -> &str {
    let rs = path.rfind('/').map_or(0, |i| i + 1);
    let bs = path.rfind('\\').map_or(0, |i| i + 1);

    &path[rs.max(bs)..]
}

fn get_buffer_spec(
    bin_path: Option<&str>,
    bin_size: usize,
    fallback_path: Option<&str>,
    fallback_size: usize,
    fallback_ref: bool,
    meshopt_ext: &str,
) -> String {
    let mut json = String::new();
    append(&mut json, "\"buffers\":[");
    append(&mut json, "{");
    if let Some(bin_path) = bin_path {
        append(&mut json, "\"uri\":\"");
        append(&mut json, bin_path);
        append(&mut json, "\"");
    }
    comma(&mut json);
    append(&mut json, "\"byteLength\":");
    append_size(&mut json, bin_size);
    append(&mut json, "}");
    if fallback_ref {
        comma(&mut json);
        append(&mut json, "{");
        if let Some(fallback_path) = fallback_path {
            append(&mut json, "\"uri\":\"");
            append(&mut json, fallback_path);
            append(&mut json, "\"");
        }
        comma(&mut json);
        append(&mut json, "\"byteLength\":");
        append_size(&mut json, fallback_size);
        append(&mut json, ",\"extensions\":{");
        append(&mut json, "\"");
        append(&mut json, meshopt_ext);
        append(&mut json, "\":{");
        append(&mut json, "\"fallback\":true");
        append(&mut json, "}}");
        append(&mut json, "}");
    }
    append(&mut json, "]");

    json
}

/// Replaces the last `n` bytes of `path` with `suffix` (C++ `std::string::replace(size - n, n, suffix)`).
fn replace_tail(path: &str, n: usize, suffix: &str) -> String {
    let cut = path.len().saturating_sub(n);
    let cut = (0..=cut).rev().find(|&c| path.is_char_boundary(c)).unwrap_or(0);
    format!("{}{suffix}", &path[..cut])
}

/// Writes all files, reporting failure like the C++ `fopen`/`fclose` checks.
fn write_outputs(files: &[(&str, &[u8])]) -> bool {
    files.iter().all(|(path, data)| std::fs::write(path, data).is_ok())
}

/// `gltfpack(input, output, report, settings)`: returns the process exit code.
pub fn gltfpack(input: &str, output: Option<&str>, report: Option<&str>, mut settings: Settings) -> i32 {
    let mut meshes: Vec<Mesh> = Vec::new();
    let mut animations: Vec<Animation> = Vec::new();

    let iext = get_extension(input);
    let oext = output.map(get_extension).unwrap_or_default();

    if output.is_some() && oext != ".gltf" && oext != ".glb" {
        eprintln!("Error: unsupported output extension '{oext}' (expected .gltf or .glb)");
        return 4;
    }

    let parsed = if iext == ".gltf" || iext == ".glb" {
        parse_gltf(input, &mut meshes, &mut animations)
    } else if iext == ".obj" {
        parse_obj(input, &mut meshes)
    } else if iext == ".ply" {
        if settings.fallback {
            eprintln!("Option -cf can not be used together with Gaussian Splats");
            return 2;
        }

        let result = parse_splats(input, &mut meshes);

        if result.is_ok() {
            eprintln!("Warning: Gaussian Splatting support is experimental; output structure, compression and quality may change");

            // override position storage to -vpf; we can't support any other form of position quantization as position & scale have the same transform
            settings.pos_float = true;

            // override compression to KHR_meshopt_compression for smaller files
            settings.compresskhr = true;
        }

        result
    } else {
        eprintln!("Error loading {input}: unknown extension (expected .gltf, .glb, .obj or .ply)");
        return 2;
    };

    let mut data = match parsed {
        Ok(data) => data,
        Err(error) => {
            eprintln!("Error loading {input}: {error}");
            return 2;
        }
    };

    if !crate::encodebasis::SUPPORTED && !data.images.is_empty() && settings.texture_ktx2 {
        eprintln!("Error: gltfpack was built without BasisU support, texture compression is not available");
        return 3;
    }

    if !data.images.is_empty() && settings.texture_webp {
        eprintln!("Error: gltfpack was built without WebP support, texture compression is not available");
        return 3;
    }

    if oext == ".glb" {
        settings.texture_embed = true;
    }

    if !data.images.is_empty() && !settings.texture_ref && !settings.texture_embed {
        let external = |uri: &Option<String>| uri.as_deref().filter(|u| !u.starts_with("data:")).map(str::to_string);

        for i in 0..data.images.len() {
            let Some(uri) = external(&data.images[i].uri) else {
                continue;
            };

            for j in 0..i {
                let Some(urj) = external(&data.images[j].uri) else {
                    continue;
                };

                if uri != urj && get_base_name(&uri) == get_base_name(&urj) {
                    eprintln!("Warning: images {uri} and {urj} share the same base name and will overwrite each other");
                    break;
                }
            }
        }
    }

    let meshopt_ext = if settings.compresskhr {
        "KHR_meshopt_compression"
    } else {
        "EXT_meshopt_compression"
    };

    let mut json = String::new();
    let mut bin = Vec::new();
    let mut fallback = Vec::new();
    let mut fallback_size = 0usize;

    json.push('{');
    let bufferspec_pos = process(
        &mut data,
        Some(input),
        output,
        report,
        &mut meshes,
        &mut animations,
        &settings,
        &mut json,
        &mut bin,
        &mut fallback,
        &mut fallback_size,
        meshopt_ext,
    );
    json.push('}');

    cgltf::free(data);

    let Some(output) = output else {
        return 0;
    };

    if oext == ".gltf" {
        let binpath = replace_tail(output, 5, ".bin");
        let fbpath = replace_tail(output, 5, ".fallback.bin");

        let bufferspec = get_buffer_spec(
            Some(get_base_name(&binpath)),
            bin.len(),
            if settings.fallback { Some(get_base_name(&fbpath)) } else { None },
            fallback_size,
            settings.compress,
            meshopt_ext,
        );
        json.insert_str(bufferspec_pos, &format!(",{bufferspec}"));

        let mut files: Vec<(&str, &[u8])> = vec![(output, json.as_bytes()), (&binpath, &bin)];
        if settings.fallback {
            files.push((&fbpath, &fallback));
        }

        if !write_outputs(&files) {
            eprintln!("Error saving {output}");
            return 4;
        }
    } else if oext == ".glb" {
        let fbpath = replace_tail(output, 4, ".fallback.bin");

        let bufferspec = get_buffer_spec(
            None,
            bin.len(),
            if settings.fallback { Some(get_base_name(&fbpath)) } else { None },
            fallback_size,
            settings.compress,
            meshopt_ext,
        );
        json.insert_str(bufferspec_pos, &format!(",{bufferspec}"));

        while !json.len().is_multiple_of(4) {
            json.push(' ');
        }

        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }

        // GLB header and two chunks with a chunk header
        let size = 12 + 8 + json.len() + 8 + bin.len();

        if size > u32::MAX as usize {
            eprintln!("Error: GLB output cannot exceed 4 GB in size");
            return 4;
        }

        let mut glb = Vec::with_capacity(size);
        for v in [0x4654_6C67u32, 2, size as u32, json.len() as u32, 0x4E4F_534A] {
            glb.extend_from_slice(&v.to_le_bytes());
        }
        glb.extend_from_slice(json.as_bytes());
        for v in [bin.len() as u32, 0x004E_4942] {
            glb.extend_from_slice(&v.to_le_bytes());
        }
        glb.extend_from_slice(&bin);

        let mut files: Vec<(&str, &[u8])> = vec![(output, &glb)];
        if settings.fallback {
            files.push((&fbpath, &fallback));
        }

        if !write_outputs(&files) {
            eprintln!("Error saving {output}");
            return 4;
        }
    } else {
        eprintln!("Error saving {output}: unknown extension (expected .gltf or .glb)");
        return 4;
    }

    0
}

/// Default settings (`defaults()`).
pub fn defaults() -> Settings {
    Settings {
        quantize: true,
        pos_bits: 14,
        tex_bits: 12,
        nrm_bits: 8,
        col_bits: 8,
        trn_bits: 16,
        rot_bits: 12,
        scl_bits: 16,
        anim_freq: 30,
        mesh_dedup: true,
        simplify_ratio: 1.0,
        simplify_error: 1e-2,
        texture_mode: [TextureMode::Raw; TEXTURE_KIND_COUNT],
        texture_scale: [1.0; TEXTURE_KIND_COUNT],
        texture_quality: [8; TEXTURE_KIND_COUNT],
        splats_dim: 2,
        ..Default::default()
    }
}

fn clamp<T: PartialOrd>(v: T, min: T, max: T) -> T {
    if v < min {
        min
    } else if v > max {
        max
    } else {
        v
    }
}

fn texture_mask(arg: &str) -> u32 {
    let mut result = 0;

    for seg in arg.split(',') {
        match seg {
            "color" => result |= 1 << TextureKind::Color as u32,
            "normal" => result |= 1 << TextureKind::Normal as u32,
            "attrib" => result |= 1 << TextureKind::Attrib as u32,
            _ => eprintln!("Warning: unrecognized texture class {seg}"),
        }
    }

    result
}

fn apply_setting<T: Copy>(data: &mut [T; TEXTURE_KIND_COUNT], value: T, mask: u32) {
    for (kind, d) in data.iter_mut().enumerate() {
        if mask & (1 << kind) != 0 {
            *d = value;
        }
    }
}

fn first_byte(s: &str) -> u8 {
    s.as_bytes().first().copied().unwrap_or(0)
}

fn print_usage(help: bool) {
    let mut s = String::new();
    s += &format!("gltfpack {}\n", get_version());
    s += "Usage: gltfpack [options] -i input -o output\n";

    if help {
        s += concat!(
            "\nBasics:\n",
            "\t-i file: input file to process, .gltf/.glb/.obj/.ply\n",
            "\t-o file: output file path, .gltf/.glb\n",
            "\t-c: produce compressed gltf/glb files (-cc/-cz for higher compression ratio)\n",
            "\nTextures:\n",
            "\t-tc: convert all textures to KTX2 with BasisU supercompression\n",
            "\t-tu: use UASTC when encoding textures (much higher quality and much larger size)\n",
            "\t-tw: convert all textures to WebP\n",
            "\t-tq N: set texture encoding quality (default: 8; N should be between 1 and 10)\n",
            "\t-ts R: scale texture dimensions by the ratio R (default: 1; R should be between 0 and 1)\n",
            "\t-tl N: limit texture dimensions to N pixels (default: 0 = no limit)\n",
            "\t-tp: resize textures to nearest power of 2 to conform to WebGL1 restrictions\n",
            "\t-tfy: flip textures along Y axis during BasisU supercompression\n",
            "\t-tj N: use N threads when compressing textures\n",
            "\t-tr: keep referring to original texture paths instead of copying/embedding images\n",
            "\tTexture classes:\n",
            "\t-tc C: use ETC1S when encoding textures of class C\n",
            "\t-tu C: use UASTC when encoding textures of class C\n",
            "\t-tw C: use WebP when encoding textures of class C\n",
            "\t-tq C N: set texture encoding quality for class C\n",
            "\t-ts C R: scale texture dimensions for class C\n",
            "\t-tl C N: limit texture dimensions for class C\n",
            "\t... where C is a comma-separated list (no spaces) with valid values color,normal,attrib\n",
            "\nSimplification:\n",
            "\t-si R: simplify meshes targeting triangle/point count ratio R (default: 1; R should be between 0 and 1)\n",
            "\t-se E: limit simplification error to E (default: 0.01 = 1% deviation; E should be between 0 and 1)\n",
            "\t-sv: simplify with vertex position and attribute optimization\n",
            "\t-sp: use permissive simplification mode to allow simplification across attribute discontinuities\n",
            "\t-sa: aggressively simplify to the target ratio disregarding quality\n",
            "\t-slb: lock border vertices during simplification to avoid gaps on connected meshes\n",
            "\nVertex precision:\n",
            "\t-vp N: use N-bit quantization for positions (default: 14; N should be between 1 and 16)\n",
            "\t-vt N: use N-bit quantization for texture coordinates (default: 12; N should be between 1 and 16)\n",
            "\t-vn N: use N-bit quantization for normals (default: 8; N should be between 1 and 16) and tangents (up to 8-bit)\n",
            "\t-vc N: use N-bit quantization for colors (default: 8; N should be between 1 and 16)\n",
            "\nVertex positions:\n",
            "\t-vpi: use integer attributes for positions (default)\n",
            "\t-vpn: use normalized attributes for positions\n",
            "\t-vpf: use floating point attributes for positions\n",
            "\nVertex attributes:\n",
            "\t-vtf: use floating point attributes for texture coordinates\n",
            "\t-vnf: use floating point attributes for normals\n",
            "\t-vi: use interleaved vertex attributes (reduces compression efficiency)\n",
            "\t-gt: generate tangent frames when needed, replacing existing tangents\n",
            "\t-gn A: generate normals when absent, using crease angle A (degrees)\n",
            "\t-kv: keep source vertex attributes even if they aren't used\n",
            "\nAnimations:\n",
            "\t-at N: use N-bit quantization for translations (default: 16; N should be between 1 and 24)\n",
            "\t-ar N: use N-bit quantization for rotations (default: 12; N should be between 4 and 16)\n",
            "\t-as N: use N-bit quantization for scale (default: 16; N should be between 1 and 24)\n",
            "\t-af N: resample animations at N Hz (default: 30; use 0 to disable)\n",
            "\t-ac: keep constant animation tracks even if they don't modify the node transform\n",
            "\nGaussian Splats:\n",
            "\t-gsd N: keep spherical harmonics up to degree N (default: 2; N should be between 0 and 3)\n",
            "\nScene:\n",
            "\t-kn: keep named nodes and meshes attached to named nodes so that named nodes can be transformed externally\n",
            "\t-km: keep named materials and disable named material merging\n",
            "\t-ke: keep extras data\n",
            "\t-mm: merge instances of the same mesh together when possible\n",
            "\t-mi: use EXT_mesh_gpu_instancing when serializing multiple mesh instances\n",
            "\nMiscellaneous:\n",
            "\t-cf: produce compressed gltf/glb files with fallback for loaders that don't support compression\n",
            "\t-ce ext|khr: use EXT or KHR version of meshopt compression extension for compression\n",
            "\t-noq: disable quantization; produces much larger glTF files with no extensions\n",
            "\t-v: verbose output (when used with other options)\n",
            "\t-v: print version (when used without other options)\n",
            "\t-r file: output a JSON report to file\n",
            "\t-h: display this help and exit\n",
        );
    } else {
        s += concat!(
            "\nBasics:\n",
            "\t-i file: input file to process, .gltf/.glb/.obj/.ply\n",
            "\t-o file: output file path, .gltf/.glb\n",
            "\t-c: produce compressed gltf/glb files (-cc/-cz for higher compression ratio)\n",
            "\t-tc: convert all textures to KTX2 with BasisU supercompression\n",
            "\t-tw: convert all textures to WebP\n",
            "\t-si R: simplify meshes targeting triangle/point count ratio R (between 0 and 1)\n",
            "\nRun gltfpack -h to display a full list of options\n",
        );
    }

    eprint!("{s}");
}

/// `main(argc, argv)`: `args` excludes the program name. Returns the exit code.
pub fn run(args: &[String]) -> i32 {
    let _ = mo::encode_vertex_version(0);
    let _ = mo::encode_index_version(1);

    let mut settings = defaults();

    let mut input: Option<&str> = None;
    let mut output: Option<&str> = None;
    let mut report: Option<&str> = None;
    let mut help = false;
    let mut test = false;
    let mut require_texc = false;

    let mut testinputs: Vec<&str> = Vec::new();

    let argc = args.len() + 1;
    let arg_at = |i: usize| args.get(i).map(String::as_str).unwrap_or("");
    let is_digit = |i: usize| i < args.len() && first_byte(arg_at(i)).is_ascii_digit();
    let is_alpha = |i: usize| i < args.len() && first_byte(arg_at(i)).is_ascii_alphabetic();

    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();

        let int_arg = |i: &mut usize| {
            *i += 1;
            atoi(arg_at(*i).as_bytes())
        };
        let float_arg = |i: &mut usize| {
            *i += 1;
            atof(arg_at(*i).as_bytes()) as f32
        };

        match arg {
            "-vp" if is_digit(i + 1) => settings.pos_bits = clamp(int_arg(&mut i), 1, 16),
            "-vt" if is_digit(i + 1) => settings.tex_bits = clamp(int_arg(&mut i), 1, 16),
            "-vn" if is_digit(i + 1) => settings.nrm_bits = clamp(int_arg(&mut i), 1, 16),
            "-vc" if is_digit(i + 1) => settings.col_bits = clamp(int_arg(&mut i), 1, 16),
            "-vpi" => {
                settings.pos_float = false;
                settings.pos_normalized = false;
            }
            "-vpn" => {
                settings.pos_float = false;
                settings.pos_normalized = true;
            }
            "-vpf" => settings.pos_float = true,
            "-vtf" => settings.tex_float = true,
            "-vnf" => settings.nrm_float = true,
            "-vi" => settings.mesh_interleaved = true,
            "-gt" => settings.mesh_tangents = true,
            "-gn" if is_digit(i + 1) => {
                settings.mesh_normals = true;
                settings.normals_crease = clamp(float_arg(&mut i), 0.0, 180.0);
            }
            "-at" if is_digit(i + 1) => settings.trn_bits = clamp(int_arg(&mut i), 1, 24),
            "-ar" if is_digit(i + 1) => settings.rot_bits = clamp(int_arg(&mut i), 4, 16),
            "-as" if is_digit(i + 1) => settings.scl_bits = clamp(int_arg(&mut i), 1, 24),
            "-af" if is_digit(i + 1) => settings.anim_freq = clamp(int_arg(&mut i), 0, 100),
            "-ac" => settings.anim_const = true,
            "-kn" => settings.keep_nodes = true,
            "-km" => settings.keep_materials = true,
            "-ke" => settings.keep_extras = true,
            "-kv" => settings.keep_attributes = true,
            "-mdd" => {
                eprintln!("Warning: option -mdd disables mesh deduplication and is temporary; avoid production usage");
                settings.mesh_dedup = false;
            }
            "-mm" => settings.mesh_merge = true,
            "-mi" => settings.mesh_instancing = true,
            "-si" if is_digit(i + 1) => settings.simplify_ratio = clamp(float_arg(&mut i), 0.0, 1.0),
            "-se" if is_digit(i + 1) => settings.simplify_error = clamp(float_arg(&mut i), 0.0, 1.0),
            "-sa" => settings.simplify_aggressive = true,
            "-slb" => settings.simplify_lock_borders = true,
            "-sv" => settings.simplify_update = true,
            "-sp" => settings.simplify_permissive = true,
            "-tu" | "-tc" | "-tw" => {
                let mode = match arg {
                    "-tu" => TextureMode::Uastc,
                    "-tc" => TextureMode::Etc1s,
                    _ => TextureMode::WebP,
                };

                if mode == TextureMode::WebP {
                    settings.texture_webp = true;
                } else {
                    settings.texture_ktx2 = true;
                }

                let mut mask = !0u32;
                if is_alpha(i + 1) {
                    i += 1;
                    mask = texture_mask(arg_at(i));
                }

                apply_setting(&mut settings.texture_mode, mode, mask);
            }
            "-tq" if is_digit(i + 1) => {
                require_texc = true;

                let quality = clamp(int_arg(&mut i), 1, 10);
                apply_setting(&mut settings.texture_quality, quality, !0);
            }
            "-tq" if i + 2 < args.len() && is_alpha(i + 1) && is_digit(i + 2) => {
                require_texc = true;

                i += 1;
                let mask = texture_mask(arg_at(i));
                let quality = clamp(int_arg(&mut i), 1, 10);
                apply_setting(&mut settings.texture_quality, quality, mask);
            }
            "-ts" if is_digit(i + 1) => {
                require_texc = true;

                let scale = clamp(float_arg(&mut i), 0.0, 1.0);
                apply_setting(&mut settings.texture_scale, scale, !0);
            }
            "-ts" if i + 2 < args.len() && is_alpha(i + 1) && is_digit(i + 2) => {
                require_texc = true;

                i += 1;
                let mask = texture_mask(arg_at(i));
                let scale = clamp(float_arg(&mut i), 0.0, 1.0);
                apply_setting(&mut settings.texture_scale, scale, mask);
            }
            "-tl" if is_digit(i + 1) => {
                require_texc = true;

                let limit = int_arg(&mut i);
                apply_setting(&mut settings.texture_limit, limit, !0);
            }
            "-tl" if i + 2 < args.len() && is_alpha(i + 1) && is_digit(i + 2) => {
                require_texc = true;

                i += 1;
                let mask = texture_mask(arg_at(i));
                let limit = int_arg(&mut i);
                apply_setting(&mut settings.texture_limit, limit, mask);
            }
            "-tp" => {
                require_texc = true;

                settings.texture_pow2 = true;
            }
            "-tfy" => {
                require_texc = true;

                settings.texture_flipy = true;
            }
            "-tr" => settings.texture_ref = true,
            "-tj" if is_digit(i + 1) => settings.texture_jobs = clamp(int_arg(&mut i), 0, 128),
            "-gsd" if is_digit(i + 1) => settings.splats_dim = clamp(int_arg(&mut i), 0, 3),
            "-noq" => {
                // TODO: Warn if -noq is used and suggest -vpf instead; use -noqq to silence
                settings.quantize = false;
            }
            "-i" if i + 1 < args.len() && input.is_none() => {
                i += 1;
                input = Some(arg_at(i));
            }
            "-o" if i + 1 < args.len() && output.is_none() => {
                i += 1;
                output = Some(arg_at(i));
            }
            "-r" if i + 1 < args.len() && report.is_none() => {
                i += 1;
                report = Some(arg_at(i));
            }
            "-c" => settings.compress = true,
            "-cc" => {
                settings.compress = true;
                settings.compressmore = true;
            }
            "-cf" => {
                settings.compress = true;
                settings.fallback = true;
            }
            "-cz" => {
                settings.compress = true;
                settings.compressmore = true;
                settings.compresskhr = true;
            }
            "-ce" if matches!(args.get(i + 1).map(String::as_str), Some("khr" | "ext")) => {
                settings.compress = true;
                i += 1;
                settings.compresskhr = arg_at(i) == "khr";
            }
            "-v" => settings.verbose = 1,
            "-vv" => settings.verbose = 2,
            "-h" => help = true,
            "-test" => test = true,
            _ if arg.starts_with('-') => {
                eprintln!("Unrecognized option {arg}");
                return 1;
            }
            _ if test => testinputs.push(arg),
            _ => {
                eprintln!("Expected option, got {arg} instead");
                return 1;
            }
        }

        i += 1;
    }

    // shortcut for gltfpack -v
    if settings.verbose != 0 && argc == 2 {
        println!("gltfpack {}", get_version());
        return 0;
    }

    if test {
        for path in testinputs {
            println!("{path}");
            gltfpack(path, None, None, settings.clone());
        }

        return 0;
    }

    let (Some(input), Some(output), false) = (input, output, help) else {
        print_usage(help);
        return 1;
    };

    if require_texc && !settings.texture_ktx2 && !settings.texture_webp {
        eprintln!("Texture processing is only supported when texture compression is enabled via -tc/-tu/-tw");
        return 1;
    }

    if settings.texture_ref && (settings.texture_ktx2 || settings.texture_webp) {
        eprintln!("Option -tr currently can not be used together with texture compression");
        return 1;
    }

    if settings.fallback && settings.compressmore {
        eprintln!("Option -cf can not be used together with -cc");
        return 1;
    }

    if settings.fallback && (settings.pos_float || settings.tex_float || settings.nrm_float) {
        eprintln!("Option -cf can not be used together with -vpf, -vtf or -vnf");
        return 1;
    }

    if settings.keep_nodes && (settings.mesh_merge || settings.mesh_instancing) {
        eprintln!("Warning: option -kn disables mesh merge (-mm) and mesh instancing (-mi) optimizations");
    }

    gltfpack(input, Some(output), report, settings)
}
