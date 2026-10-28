//! glTF/GLB input (`parsegltf.cpp`): cgltf parsing, meshopt decompression and conversion into gltfpack meshes/animations.

use crate::cgltf::{self, AnimationPathType, AttributeType, CgltfResult, Data, PrimitiveType, Type};
use crate::types::*;
use meshoptimizer as mo;

const MAX_STREAMS: usize = 16;

fn get_error(result: CgltfResult, has_data: bool) -> &'static str {
    match result {
        CgltfResult::FileNotFound => {
            if has_data {
                "resource not found"
            } else {
                "file not found"
            }
        }
        CgltfResult::IoError => "I/O error",
        CgltfResult::InvalidJson => "invalid JSON",
        CgltfResult::InvalidGltf => "invalid GLTF",
        CgltfResult::OutOfMemory => "out of memory",
        CgltfResult::LegacyGltf => "legacy GLTF",
        CgltfResult::DataTooShort => {
            if has_data {
                "buffer too short"
            } else {
                "not a GLTF file"
            }
        }
        CgltfResult::UnknownFormat => {
            if has_data {
                "unknown resource format"
            } else {
                "not a GLTF file"
            }
        }
        _ => "unknown error",
    }
}

fn read_accessor_scalar(data: &Data, accessor: &cgltf::Accessor) -> Vec<f32> {
    let mut result = vec![0.0f32; accessor.count];
    cgltf::accessor_unpack_floats(data, accessor, Some(&mut result));
    result
}

fn read_accessor(data: &Data, accessor: &cgltf::Accessor) -> Vec<Attr> {
    let components = cgltf::num_components(accessor.type_);

    let mut temp = vec![0.0f32; accessor.count * components];
    cgltf::accessor_unpack_floats(data, accessor, Some(&mut temp));

    let mut result = vec![[0.0f32; 4]; accessor.count];

    for (i, r) in result.iter_mut().enumerate() {
        for k in 0..components.min(4) {
            r[k] = temp[i * components + k];
        }
    }

    result
}

fn read_accessor_sparse(data: &Data, accessor: &cgltf::Accessor, sparse: &[u32]) -> Vec<Attr> {
    let mut result = vec![[0.0f32; 4]; sparse.len()];

    for (r, &index) in result.iter_mut().zip(sparse) {
        cgltf::accessor_read_float(data, accessor, index as usize, r);
    }

    result
}

fn fixup_indices(indices: &mut Vec<u32>, type_: &mut PrimitiveType) {
    match *type_ {
        PrimitiveType::LineLoop => {
            let mut result = Vec::with_capacity(indices.len() * 2 + 2);

            for i in 1..=indices.len() {
                result.push(indices[i - 1]);
                result.push(indices[i % indices.len()]);
            }

            *indices = result;
            *type_ = PrimitiveType::Lines;
        }
        PrimitiveType::LineStrip => {
            let mut result = Vec::with_capacity(indices.len() * 2);

            for i in 1..indices.len() {
                result.push(indices[i - 1]);
                result.push(indices[i]);
            }

            *indices = result;
            *type_ = PrimitiveType::Lines;
        }
        PrimitiveType::TriangleStrip => {
            let mut result = Vec::with_capacity(indices.len() * 3);

            for i in 2..indices.len() {
                let flip = i & 1;

                result.push(indices[i - 2 + flip]);
                result.push(indices[i - 1 - flip]);
                result.push(indices[i]);
            }

            *indices = result;
            *type_ = PrimitiveType::Triangles;
        }
        PrimitiveType::TriangleFan => {
            let mut result = Vec::with_capacity(indices.len() * 3);

            for i in 2..indices.len() {
                result.push(indices[0]);
                result.push(indices[i - 1]);
                result.push(indices[i]);
            }

            *indices = result;
            *type_ = PrimitiveType::Triangles;
        }
        PrimitiveType::Lines => {
            // glTF files don't require that line index count is divisible by 2, but it is obviously critical for scenes to render
            indices.truncate(indices.len() / 2 * 2);
        }
        PrimitiveType::Triangles => {
            // glTF files don't require that triangle index count is divisible by 3, but it is obviously critical for scenes to render
            indices.truncate(indices.len() / 3 * 3);
        }
        _ => {}
    }
}

fn is_id_attribute(name: &str) -> bool {
    name == "_ID" || name == "_BATCHID" || name.starts_with("_FEATURE_ID_")
}

fn attr_name(attr: &cgltf::Attribute) -> &str {
    attr.name.as_deref().unwrap_or("")
}

fn parse_meshes_gltf(data: &Data, meshes: &mut Vec<Mesh>, mesh_remap: &mut Vec<(usize, usize)>) {
    let total_primitives: usize = data.meshes.iter().map(|m| m.primitives.len()).sum();

    meshes.reserve(total_primitives);
    mesh_remap.resize(data.meshes.len(), (0, 0));

    for (mi, mesh) in data.meshes.iter().enumerate() {
        let remap_offset = meshes.len();

        for (pi, primitive) in mesh.primitives.iter().enumerate() {
            if primitive.type_ == PrimitiveType::Points && primitive.indices.is_some() {
                eprintln!("Warning: ignoring primitive {pi} of mesh {mi} because indexed points are not supported");
                continue;
            }

            let Some(positions) = cgltf::find_accessor(primitive, AttributeType::Position, 0) else {
                eprintln!("Warning: ignoring primitive {pi} of mesh {mi} because it has no position data");
                continue;
            };

            let mut result = Mesh {
                scene: -1,
                material: primitive.material,
                extras: primitive.extras.clone(),
                type_: primitive.type_,
                ..Default::default()
            };

            result.streams.reserve(primitive.attributes.len());

            let vertex_count = data.accessors[positions].count;

            if let Some(indices) = primitive.indices {
                let accessor = &data.accessors[indices];
                result.indices = vec![0; accessor.count];
                if !result.indices.is_empty() {
                    cgltf::accessor_unpack_indices(data, accessor, Some(&mut result.indices));
                }
            } else if primitive.type_ != PrimitiveType::Points {
                // note, while we could generate a good index buffer here, mesh will be reindexed during processing
                result.indices = (0..vertex_count as u32).collect();
            }

            // convert line loops and line/triangle strips to lists
            fixup_indices(&mut result.indices, &mut result.type_);

            let mut sparse = Vec::new();

            // if the index data is very sparse, switch to deindexing on the fly to avoid the excessive cost of reading large accessors
            if !result.indices.is_empty() && result.indices.len() < vertex_count / 2 {
                sparse = result.indices.clone();

                // mesh will be reindexed during processing
                for (i, index) in result.indices.iter_mut().enumerate() {
                    *index = i as u32;
                }
            }

            for attr in &primitive.attributes {
                if attr.type_ == AttributeType::Invalid || (attr.type_ == AttributeType::Custom && !is_id_attribute(attr_name(attr))) {
                    eprintln!(
                        "Warning: ignoring {} attribute {} in primitive {pi} of mesh {mi}",
                        if attr.type_ == AttributeType::Invalid {
                            "unknown"
                        } else {
                            "custom"
                        },
                        attr_name(attr)
                    );
                    continue;
                }

                if result.streams.len() == MAX_STREAMS {
                    eprintln!(
                        "Warning: ignoring attribute {} in primitive {pi} of mesh {mi} (limit {MAX_STREAMS} reached)",
                        attr_name(attr)
                    );
                    continue;
                }

                let accessor = &data.accessors[attr.data];

                let mut s = Stream {
                    type_: attr.type_,
                    index: attr.index,
                    ..Default::default()
                };

                if attr.type_ == AttributeType::Custom {
                    s.custom_name = attr.name.clone();
                }

                s.data = if sparse.is_empty() {
                    read_accessor(data, accessor)
                } else {
                    read_accessor_sparse(data, accessor, &sparse)
                };

                if attr.type_ == AttributeType::Color && accessor.type_ == Type::Vec3 {
                    for a in s.data.iter_mut() {
                        a[3] = 1.0;
                    }
                }

                result.streams.push(s);
            }

            for (ti, target) in primitive.targets.iter().enumerate() {
                for attr in &target.attributes {
                    if attr.type_ == AttributeType::Invalid || attr.type_ == AttributeType::Custom {
                        eprintln!(
                            "Warning: ignoring {} attribute {} in morph target {ti} of primitive {pi} of mesh {mi}",
                            if attr.type_ == AttributeType::Invalid {
                                "unknown"
                            } else {
                                "custom"
                            },
                            attr_name(attr)
                        );
                        continue;
                    }

                    let accessor = &data.accessors[attr.data];

                    let s = Stream {
                        type_: attr.type_,
                        index: attr.index,
                        target: (ti + 1) as i32,
                        data: if sparse.is_empty() {
                            read_accessor(data, accessor)
                        } else {
                            read_accessor_sparse(data, accessor, &sparse)
                        },
                        ..Default::default()
                    };

                    result.streams.push(s);
                }
            }

            result.targets = primitive.targets.len();
            result.target_weights = mesh.weights.clone().unwrap_or_default();
            result.target_names = mesh.target_names.clone().unwrap_or_default();

            result.variants.clone_from(&primitive.mappings);

            meshes.push(result);
        }

        mesh_remap[mi] = (remap_offset, meshes.len());
    }
}

fn parse_mesh_instances_gltf(instances: &mut Vec<Instance>, data: &Data, node_index: usize) {
    let node = &data.nodes[node_index];

    let mut translation = None;
    let mut rotation = None;
    let mut scale = None;
    let mut color = None;

    for attr in &node.mesh_gpu_instancing.attributes {
        let accessor = &data.accessors[attr.data];
        let name = attr_name(attr);

        if name == "TRANSLATION" && accessor.type_ == Type::Vec3 {
            translation = Some(accessor);
        } else if name == "ROTATION" && accessor.type_ == Type::Vec4 {
            rotation = Some(accessor);
        } else if name == "SCALE" && accessor.type_ == Type::Vec3 {
            scale = Some(accessor);
        } else if name == "_COLOR_0" && (accessor.type_ == Type::Vec3 || accessor.type_ == Type::Vec4) {
            color = Some(accessor);
        } else {
            eprintln!(
                "Warning: ignoring {} instance attribute {name} in node {node_index}",
                if name.starts_with('_') { "custom" } else { "unknown" }
            );
        }
    }

    let count = node
        .mesh_gpu_instancing
        .attributes
        .first()
        .map_or(0, |a| data.accessors[a.data].count);

    instances.reserve(count);

    let mut instance = cgltf::Node {
        parent: Some(node_index),
        has_translation: translation.is_some(),
        has_rotation: rotation.is_some(),
        has_scale: scale.is_some(),
        ..Default::default()
    };
    instance.rotation[3] = 1.0;
    instance.scale = [1.0; 3];

    // cgltf_accessor_read_float with element_size 4 into float[3] fields; the 4th float lands in the next field
    // (rotation[0] / matrix[0]) only for accessors without buffer views, which is unobservable here
    let read3 = |accessor: &cgltf::Accessor, i: usize, out: &mut [f32; 3]| {
        let mut v = [out[0], out[1], out[2], 0.0];
        if cgltf::accessor_read_float(data, accessor, i, &mut v) {
            out.copy_from_slice(&v[..3]);
        }
    };

    for i in 0..count {
        if let Some(t) = translation {
            read3(t, i, &mut instance.translation);
        }
        if let Some(r) = rotation {
            cgltf::accessor_read_float(data, r, i, &mut instance.rotation);
        }
        if let Some(s) = scale {
            read3(s, i, &mut instance.scale);
        }

        let mut obj = Instance {
            transform: cgltf::node_transform_world(&data.nodes, &instance),
            color: [1.0; 4],
        };

        if let Some(c) = color {
            cgltf::accessor_read_float(data, c, i, &mut obj.color);
        }

        instances.push(obj);
    }
}

fn parse_mesh_nodes_gltf(data: &Data, meshes: &mut Vec<Mesh>, mesh_remap: &[(usize, usize)]) {
    for (i, node) in data.nodes.iter().enumerate() {
        let Some(node_mesh) = node.mesh else {
            continue;
        };

        let range = mesh_remap[node_mesh];

        for mi in range.0..range.1 {
            let mut target = mi;

            {
                let mesh = &meshes[mi];
                if mesh.skin != node.skin && (!mesh.nodes.is_empty() || !mesh.instances.is_empty()) {
                    // this should be extremely rare - if the same mesh is used with different skins, we need to duplicate it
                    // in this case we don't spend any effort on keeping the number of duplicates to the minimum, because this
                    // should really never happen.
                    let copy = mesh.clone();
                    meshes.push(copy);
                    target = meshes.len() - 1;
                }
            }

            let mesh = &mut meshes[target];

            if node.has_mesh_gpu_instancing {
                mesh.scene = 0; // we need to assign scene index since instances are attached to a scene; for now we assume 0
                parse_mesh_instances_gltf(&mut mesh.instances, data, i);
            } else {
                mesh.skin = node.skin;
                mesh.nodes.push(i);
            }
        }
    }

    for mesh in meshes.iter_mut() {
        // because the rest of gltfpack assumes that empty nodes array = world-space mesh, we need to filter unused meshes
        if mesh.nodes.is_empty() && mesh.instances.is_empty() {
            mesh.streams.clear();
            mesh.indices.clear();
        }
    }
}

fn parse_animations_gltf(data: &Data, animations: &mut Vec<Animation>) {
    animations.reserve(data.animations.len());

    for (i, animation) in data.animations.iter().enumerate() {
        let name = animation.name.as_deref().unwrap_or("");

        let mut result = Animation {
            name: animation.name.clone(),
            ..Default::default()
        };

        result.tracks.reserve(animation.channels.len());

        for (j, channel) in animation.channels.iter().enumerate() {
            let Some(target_node) = channel.target_node else {
                eprintln!("Warning: ignoring channel {j} of animation {i} ({name}) because it has no target node");
                continue;
            };

            let sampler = &animation.samplers[channel.sampler];

            let components = if channel.target_path == AnimationPathType::Weights {
                data.nodes[target_node]
                    .mesh
                    .and_then(|m| data.meshes[m].primitives.first())
                    .map_or(0, |p| p.targets.len())
            } else {
                1
            };

            result.tracks.push(Track {
                node: target_node,
                path: channel.target_path,
                components,
                interpolation: sampler.interpolation,
                time: read_accessor_scalar(data, &data.accessors[sampler.input]),
                data: read_accessor(data, &data.accessors[sampler.output]),
                ..Default::default()
            });
        }

        if result.tracks.is_empty() {
            eprintln!("Warning: ignoring animation {i} ({name}) because it has no valid tracks");
            continue;
        }

        animations.push(result);
    }
}

fn requires_extension(data: &Data, name: &str) -> bool {
    data.extensions_required.iter().any(|e| e == name)
}

fn buffer_has_data(data: &Data, buffer: usize) -> bool {
    data.buffers.get(buffer).is_some_and(|b| b.data.is_some())
}

fn needs_dummy_buffers(data: &Data) -> bool {
    for accessor in &data.accessors {
        if let Some(view) = accessor.buffer_view.map(|v| &data.buffer_views[v])
            && view.data.is_none()
            && !buffer_has_data(data, view.buffer)
        {
            return true;
        }

        if accessor.is_sparse {
            let sparse = &accessor.sparse;

            if !buffer_has_data(data, data.buffer_views[sparse.indices_buffer_view].buffer) {
                return true;
            }
            if !buffer_has_data(data, data.buffer_views[sparse.values_buffer_view].buffer) {
                return true;
            }
        }
    }

    for image in &data.images {
        if let Some(view) = image.buffer_view
            && !buffer_has_data(data, data.buffer_views[view].buffer)
        {
            return true;
        }
    }

    false
}

/// `freeUnusedBuffers`: buffer data is only needed after parsing for skin matrices and embedded images.
fn free_unused_buffers(data: &mut Data) {
    let mut used = vec![false; data.buffers.len()];

    for skin in &data.skins {
        if let Some(view) = skin.inverse_bind_matrices.and_then(|a| data.accessors[a].buffer_view) {
            used[data.buffer_views[view].buffer] = true;
        }
    }

    for image in &data.images {
        if let Some(view) = image.buffer_view {
            used[data.buffer_views[view].buffer] = true;
        }
    }

    for (buffer, used) in data.buffers.iter_mut().zip(used) {
        if !used {
            buffer.data = None;
        }
    }
}

fn decompress_meshopt(data: &mut Data) -> Result<(), CgltfResult> {
    for i in 0..data.buffer_views.len() {
        if !data.buffer_views[i].has_meshopt_compression {
            continue;
        }

        let mc = data.buffer_views[i].meshopt_compression.clone();

        let source = data
            .buffers
            .get(mc.buffer)
            .and_then(|b| b.data.as_deref())
            .ok_or(CgltfResult::InvalidGltf)?
            .get(mc.offset..)
            .ok_or(CgltfResult::InvalidGltf)?;

        let mut result = vec![0u8; mc.count * mc.stride];

        let rc = match mc.mode {
            cgltf::MeshoptCompressionMode::Attributes => mo::decode_vertex_buffer(&mut result, mc.count, mc.stride, source, mc.size),
            cgltf::MeshoptCompressionMode::Triangles | cgltf::MeshoptCompressionMode::Indices => {
                let triangles = mc.mode == cgltf::MeshoptCompressionMode::Triangles;
                decode_indices(&mut result, mc.count, mc.stride, source, mc.size, triangles)
            }
            cgltf::MeshoptCompressionMode::Invalid => return Err(CgltfResult::InvalidGltf),
        };

        if rc.is_err() {
            return Err(CgltfResult::IoError);
        }

        // filters are validated by cgltf_validate; invalid strides are rejected by the decoders
        let _ = match mc.filter {
            cgltf::MeshoptCompressionFilter::Octahedral => mo::decode_filter_oct(&mut result, mc.count, mc.stride),
            cgltf::MeshoptCompressionFilter::Quaternion => mo::decode_filter_quat(&mut result, mc.count, mc.stride),
            cgltf::MeshoptCompressionFilter::Exponential => mo::decode_filter_exp(&mut result, mc.count, mc.stride),
            cgltf::MeshoptCompressionFilter::Color => mo::decode_filter_color(&mut result, mc.count, mc.stride),
            cgltf::MeshoptCompressionFilter::None => Ok(()),
        };

        data.buffer_views[i].data = Some(result);
    }

    Ok(())
}

/// `meshopt_decodeIndexBuffer`/`meshopt_decodeIndexSequence` with a runtime index size.
fn decode_indices(result: &mut [u8], count: usize, stride: usize, source: &[u8], size: usize, triangles: bool) -> mo::Result<()> {
    match stride {
        2 => {
            let mut out = vec![0u16; count];
            if triangles {
                mo::decode_index_buffer(&mut out, count, source, size)?;
            } else {
                mo::decode_index_sequence(&mut out, count, source, size)?;
            }
            for (d, v) in result.as_chunks_mut::<2>().0.iter_mut().zip(&out) {
                *d = v.to_le_bytes();
            }
            Ok(())
        }
        4 => {
            let mut out = vec![0u32; count];
            if triangles {
                mo::decode_index_buffer(&mut out, count, source, size)?;
            } else {
                mo::decode_index_sequence(&mut out, count, source, size)?;
            }
            for (d, v) in result.as_chunks_mut::<4>().0.iter_mut().zip(&out) {
                *d = v.to_le_bytes();
            }
            Ok(())
        }
        _ => Err(mo::Error::InvalidArgument("index size must be 2 or 4")),
    }
}

fn parse_gltf_data(
    data: Option<Data>,
    result: Result<(), CgltfResult>,
    meshes: &mut Vec<Mesh>,
    animations: &mut Vec<Animation>,
) -> Result<Data, &'static str> {
    let mut data = match (data, result) {
        (data, Err(e)) => return Err(get_error(e, data.is_some())),
        (None, Ok(())) => return Err(get_error(CgltfResult::InvalidGltf, false)),
        (Some(data), Ok(())) => data,
    };

    if requires_extension(&data, "KHR_draco_mesh_compression") {
        return Err("file requires Draco mesh compression support");
    }

    if needs_dummy_buffers(&data) {
        return Err("buffer has no data");
    }

    if requires_extension(&data, "KHR_mesh_quantization") {
        eprintln!("Warning: file uses quantized geometry; repacking may result in increased quantization error");
    }
    if requires_extension(&data, "EXT_mesh_gpu_instancing") && data.scenes.len() > 1 {
        eprintln!("Warning: file uses instancing and has more than one scene; results may be incorrect");
    }

    let mut mesh_remap = Vec::new();

    parse_meshes_gltf(&data, meshes, &mut mesh_remap);
    parse_mesh_nodes_gltf(&data, meshes, &mesh_remap);
    parse_animations_gltf(&data, animations);

    free_unused_buffers(&mut data);

    Ok(data)
}

fn load(data: &mut Option<Data>, options: &cgltf::Options, path: Option<&str>) -> Result<(), CgltfResult> {
    let d = data.as_mut().ok_or(CgltfResult::InvalidGltf)?;
    cgltf::load_buffers(options, d, path)?;
    cgltf::validate(d)?;
    decompress_meshopt(d)
}

/// `parseGltf(path, ...)`: returns the parsed data or an error message.
pub fn parse_gltf(path: &str, meshes: &mut Vec<Mesh>, animations: &mut Vec<Animation>) -> Result<Data, &'static str> {
    let options = cgltf::Options::default();

    let (mut data, result) = match cgltf::parse_file(&options, path) {
        Ok(d) => (Some(d), Ok(())),
        Err(e) => (None, Err(e)),
    };

    let result = result.and_then(|()| load(&mut data, &options, Some(path)));

    parse_gltf_data(data, result, meshes, animations)
}

/// `parseGlb(buffer, size, ...)`
pub fn parse_glb(buffer: &[u8], meshes: &mut Vec<Mesh>, animations: &mut Vec<Animation>) -> Result<Data, &'static str> {
    let options = cgltf::Options {
        type_: cgltf::FileType::Glb,
        ..Default::default()
    };

    let (mut data, result) = match cgltf::parse(&options, buffer) {
        Ok(d) => (Some(d), Ok(())),
        Err(e) => (None, Err(e)),
    };

    let result = result.and_then(|()| load(&mut data, &options, None));

    parse_gltf_data(data, result, meshes, animations)
}

pub fn are_extras_equal(lhs: &cgltf::Extras, rhs: &cgltf::Extras) -> bool {
    lhs.data == rhs.data
}
