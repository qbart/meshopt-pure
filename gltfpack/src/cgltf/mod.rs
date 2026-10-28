//! Port of cgltf 1.15 (https://github.com/jkuhlmann/cgltf), the glTF 2.0 parser used by gltfpack.
//! See `docs/spec/31-cgltf.md` for the mapping from the C API.

mod jsmn;
mod parse;
mod types;

pub use types::*;

const GLB_HEADER_SIZE: usize = 12;
const GLB_CHUNK_HEADER_SIZE: usize = 8;
const GLB_VERSION: u32 = 2;
const GLB_MAGIC: u32 = 0x4654_6C67;
const GLB_MAGIC_JSON_CHUNK: u32 = 0x4E4F_534A;
const GLB_MAGIC_BIN_CHUNK: u32 = 0x004E_4942;

fn read_u32(data: &[u8], offset: usize) -> u32 {
    let mut b = [0u8; 4];
    if let Some(s) = data.get(offset..offset + 4) {
        b.copy_from_slice(s);
    }
    u32::from_le_bytes(b)
}

/// `cgltf_parse`: parses glTF JSON or GLB data. Buffers are not loaded, see [`load_buffers`].
pub fn parse(options: &Options, data: &[u8]) -> Result<Data, CgltfResult> {
    let size = data.len();

    if size < GLB_HEADER_SIZE {
        return Err(CgltfResult::DataTooShort);
    }

    let mut fixed_options = *options;

    // Magic
    let tmp = read_u32(data, 0);
    if tmp != GLB_MAGIC {
        if fixed_options.type_ == FileType::Invalid {
            fixed_options.type_ = FileType::Gltf;
        } else if fixed_options.type_ == FileType::Glb {
            return Err(CgltfResult::UnknownFormat);
        }
    }

    if fixed_options.type_ == FileType::Gltf {
        let mut result = parse::parse_json(&mut fixed_options, data)?;
        result.file_type = FileType::Gltf;
        return Ok(result);
    }

    // Version
    let version = read_u32(data, 4);
    if version != GLB_VERSION {
        return Err(if version < GLB_VERSION {
            CgltfResult::LegacyGltf
        } else {
            CgltfResult::UnknownFormat
        });
    }

    // Total length
    if read_u32(data, 8) as usize > size {
        return Err(CgltfResult::DataTooShort);
    }

    let json_chunk = GLB_HEADER_SIZE;

    if GLB_HEADER_SIZE + GLB_CHUNK_HEADER_SIZE > size {
        return Err(CgltfResult::DataTooShort);
    }

    // JSON chunk: length
    let json_length = read_u32(data, json_chunk) as usize;
    if json_length > size - GLB_HEADER_SIZE - GLB_CHUNK_HEADER_SIZE {
        return Err(CgltfResult::DataTooShort);
    }

    // JSON chunk: magic
    if read_u32(data, json_chunk + 4) != GLB_MAGIC_JSON_CHUNK {
        return Err(CgltfResult::UnknownFormat);
    }

    let json_chunk = json_chunk + GLB_CHUNK_HEADER_SIZE;

    let mut bin = None;

    if GLB_CHUNK_HEADER_SIZE <= size - GLB_HEADER_SIZE - GLB_CHUNK_HEADER_SIZE - json_length {
        // We can read another chunk
        let bin_chunk = json_chunk + json_length;

        // Bin chunk: length
        let bin_length = read_u32(data, bin_chunk) as usize;
        if bin_length > size - GLB_HEADER_SIZE - GLB_CHUNK_HEADER_SIZE - json_length - GLB_CHUNK_HEADER_SIZE {
            return Err(CgltfResult::DataTooShort);
        }

        // Bin chunk: magic
        if read_u32(data, bin_chunk + 4) != GLB_MAGIC_BIN_CHUNK {
            return Err(CgltfResult::UnknownFormat);
        }

        let bin_chunk = bin_chunk + GLB_CHUNK_HEADER_SIZE;

        bin = data.get(bin_chunk..bin_chunk + bin_length).map(<[u8]>::to_vec);
    }

    let json = data.get(json_chunk..json_chunk + json_length).unwrap_or(&[]);
    let mut result = parse::parse_json(&mut fixed_options, json)?;

    result.file_type = FileType::Glb;
    result.bin_size = bin.as_ref().map_or(0, Vec::len);
    result.bin = bin;

    Ok(result)
}

/// `cgltf_default_file_read`: reads `size` bytes (or the whole file when `size == 0`).
fn file_read(path: &str, size: usize) -> Result<Vec<u8>, CgltfResult> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).map_err(|_| CgltfResult::FileNotFound)?;

    let file_size = if size == 0 {
        let length = file.metadata().map_err(|_| CgltfResult::IoError)?.len();
        usize::try_from(length).map_err(|_| CgltfResult::IoError)?
    } else {
        size
    };

    let mut data = Vec::new();
    data.try_reserve_exact(file_size).map_err(|_| CgltfResult::OutOfMemory)?;

    let read_size = file
        .by_ref()
        .take(file_size as u64)
        .read_to_end(&mut data)
        .map_err(|_| CgltfResult::IoError)?;
    if read_size != file_size {
        return Err(CgltfResult::IoError);
    }

    Ok(data)
}

/// `cgltf_parse_file`
pub fn parse_file(options: &Options, path: &str) -> Result<Data, CgltfResult> {
    let file_data = file_read(path, 0)?;
    parse(options, &file_data)
}

/// `cgltf_combine_paths`
fn combine_paths(base: &str, uri: &str) -> (String, usize) {
    let slash = base.rfind(['/', '\\']);

    match slash {
        Some(slash) => {
            let prefix = &base[..=slash];
            (format!("{prefix}{uri}"), prefix.len())
        }
        None => (uri.to_string(), 0),
    }
}

fn load_buffer_file(size: usize, uri: &str, gltf_path: &str) -> Result<Vec<u8>, CgltfResult> {
    let (path, prefix) = combine_paths(gltf_path, uri);

    // after combining, the tail of the resulting path is a uri; decode_uri converts it into path
    let path = format!("{}{}", &path[..prefix], decode_uri(&path[prefix..]));

    file_read(&path, size)
}

/// `cgltf_load_buffer_base64`: decodes `size` bytes of base64 data.
pub fn load_buffer_base64(size: usize, base64: &[u8]) -> Result<Vec<u8>, CgltfResult> {
    let mut data = Vec::new();
    data.try_reserve_exact(size).map_err(|_| CgltfResult::OutOfMemory)?;

    let mut buffer: u32 = 0;
    let mut buffer_bits: u32 = 0;
    let mut pos = 0;

    for _ in 0..size {
        while buffer_bits < 8 {
            let ch = base64.get(pos).copied().unwrap_or(0);
            pos += 1;

            let index = match ch {
                b'A'..=b'Z' => u32::from(ch - b'A'),
                b'a'..=b'z' => u32::from(ch - b'a') + 26,
                b'0'..=b'9' => u32::from(ch - b'0') + 52,
                b'+' => 62,
                b'/' => 63,
                _ => return Err(CgltfResult::IoError),
            };

            buffer = (buffer << 6) | index;
            buffer_bits += 6;
        }

        data.push((buffer >> (buffer_bits - 8)) as u8);
        buffer_bits -= 8;
    }

    Ok(data)
}

fn unhex(ch: u8) -> Option<u8> {
    (ch as char).to_digit(16).map(|d| d as u8)
}

/// `cgltf_decode_string`: resolves JSON escapes (`\n`, `\uXXXX`, ...) in a raw JSON string.
pub fn decode_string(string: &str) -> String {
    let bytes = string.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;

    while i < bytes.len() {
        let c = bytes[i];
        i += 1;

        if c != b'\\' {
            out.push(c);
            continue;
        }

        let Some(&e) = bytes.get(i) else {
            break;
        };
        i += 1;

        match e {
            b'"' => out.push(b'"'),
            b'/' => out.push(b'/'),
            b'\\' => out.push(b'\\'),
            b'b' => out.push(b'\x08'),
            b'f' => out.push(b'\x0c'),
            b'r' => out.push(b'\r'),
            b'n' => out.push(b'\n'),
            b't' => out.push(b'\t'),
            b'u' => {
                // UCS-2 codepoint \uXXXX to UTF-8
                let mut character: u32 = 0;
                for _ in 0..4 {
                    let d = bytes.get(i).copied().and_then(unhex).map_or(u32::MAX, u32::from);
                    character = (character << 4).wrapping_add(d);
                    i += 1;
                }

                if character <= 0x7F {
                    out.push(character as u8);
                } else if character <= 0x7FF {
                    out.push(0xC0 | (character >> 6) as u8);
                    out.push(0x80 | (character & 0x3F) as u8);
                } else {
                    out.push(0xE0 | (character >> 12) as u8);
                    out.push(0x80 | ((character >> 6) & 0x3F) as u8);
                    out.push(0x80 | (character & 0x3F) as u8);
                }
            }
            _ => {}
        }
    }

    String::from_utf8_lossy(&out).into_owned()
}

/// `cgltf_decode_uri`: resolves `%XX` escapes.
pub fn decode_uri(uri: &str) -> String {
    let bytes = uri.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'%'
            && let Some(ch1) = bytes.get(i + 1).copied().and_then(unhex)
            && let Some(ch2) = bytes.get(i + 2).copied().and_then(unhex)
        {
            out.push(ch1 * 16 + ch2);
            i += 3;
            continue;
        }

        out.push(bytes[i]);
        i += 1;
    }

    String::from_utf8_lossy(&out).into_owned()
}

/// `cgltf_load_buffers`: loads buffer data from the GLB BIN chunk, data URIs and external files (relative to `gltf_path`).
pub fn load_buffers(_options: &Options, data: &mut Data, gltf_path: Option<&str>) -> Result<(), CgltfResult> {
    if let Some(first) = data.buffers.first()
        && first.data.is_none()
        && first.uri.is_none()
        && data.bin.is_some()
    {
        if data.bin_size < first.size {
            return Err(CgltfResult::DataTooShort);
        }

        data.buffers[0].data = data.bin.take();
        data.buffers[0].data_free_method = DataFreeMethod::None;
    }

    for buffer in data.buffers.iter_mut() {
        if buffer.data.is_some() {
            continue;
        }

        let Some(uri) = buffer.uri.as_deref() else {
            continue;
        };

        if uri.starts_with("data:") {
            let comma = uri.find(',');

            match comma {
                Some(comma) if comma >= 7 && &uri[comma - 7..comma] == ";base64" => {
                    let res = load_buffer_base64(buffer.size, &uri.as_bytes()[comma + 1..]);
                    buffer.data_free_method = DataFreeMethod::MemoryFree;
                    buffer.data = Some(res?);
                }
                _ => return Err(CgltfResult::UnknownFormat),
            }
        } else if !uri.contains("://")
            && let Some(gltf_path) = gltf_path
        {
            let res = load_buffer_file(buffer.size, uri, gltf_path);
            buffer.data_free_method = DataFreeMethod::FileRelease;
            buffer.data = Some(res?);
        } else {
            return Err(CgltfResult::UnknownFormat);
        }
    }

    Ok(())
}

fn calc_index_bound(data: &Data, buffer_view: usize, offset: usize, component_type: ComponentType, count: usize) -> usize {
    let view = &data.buffer_views[buffer_view];
    let Some(bytes) = data.buffers.get(view.buffer).and_then(|b| b.data.as_deref()) else {
        return 0;
    };
    let base = offset.wrapping_add(view.offset);

    let size = component_size(component_type);
    if size == 0 || !matches!(component_type, ComponentType::R8u | ComponentType::R16u | ComponentType::R32u) {
        return 0;
    }

    (0..count)
        .map_while(|i| bytes.get(base.wrapping_add(i * size)..).and_then(|b| read_index(b, component_type)))
        .max()
        .unwrap_or(0)
}

macro_rules! assert_if {
    ($cond:expr, $result:expr) => {
        if $cond {
            return Err($result);
        }
    };
}

/// `cgltf_validate`: additional checks that the parsed glTF data is consistent.
pub fn validate(data: &Data) -> Result<(), CgltfResult> {
    for accessor in &data.accessors {
        assert_if!(accessor.component_type == ComponentType::Invalid, CgltfResult::InvalidGltf);
        assert_if!(accessor.type_ == Type::Invalid, CgltfResult::InvalidGltf);

        let element_size = calc_size(accessor.type_, accessor.component_type);

        if let Some(bv) = accessor.buffer_view {
            let req_size = accessor
                .offset
                .wrapping_add(accessor.stride.wrapping_mul(accessor.count.wrapping_sub(1)))
                .wrapping_add(element_size);

            assert_if!(data.buffer_views[bv].size < req_size, CgltfResult::DataTooShort);
        }

        if accessor.is_sparse {
            let sparse = &accessor.sparse;

            let indices_component_size = component_size(sparse.indices_component_type);
            let indices_req_size = sparse
                .indices_byte_offset
                .wrapping_add(indices_component_size.wrapping_mul(sparse.count));
            let values_req_size = sparse.values_byte_offset.wrapping_add(element_size.wrapping_mul(sparse.count));

            assert_if!(
                data.buffer_views[sparse.indices_buffer_view].size < indices_req_size
                    || data.buffer_views[sparse.values_buffer_view].size < values_req_size,
                CgltfResult::DataTooShort
            );

            assert_if!(
                !matches!(
                    sparse.indices_component_type,
                    ComponentType::R8u | ComponentType::R16u | ComponentType::R32u
                ),
                CgltfResult::InvalidGltf
            );

            let indices_view = &data.buffer_views[sparse.indices_buffer_view];
            if data.buffers.get(indices_view.buffer).is_some_and(|b| b.data.is_some()) {
                let index_bound = calc_index_bound(
                    data,
                    sparse.indices_buffer_view,
                    sparse.indices_byte_offset,
                    sparse.indices_component_type,
                    sparse.count,
                );

                assert_if!(index_bound >= accessor.count, CgltfResult::DataTooShort);
            }
        }
    }

    for view in &data.buffer_views {
        let req_size = view.offset.wrapping_add(view.size);

        assert_if!(data.buffers[view.buffer].size < req_size, CgltfResult::DataTooShort);

        if view.has_meshopt_compression {
            let mc = &view.meshopt_compression;

            assert_if!(
                data.buffers[mc.buffer].size < mc.offset.wrapping_add(mc.size),
                CgltfResult::DataTooShort
            );

            assert_if!(view.stride != 0 && mc.stride != view.stride, CgltfResult::InvalidGltf);

            assert_if!(view.size != mc.stride.wrapping_mul(mc.count), CgltfResult::InvalidGltf);

            assert_if!(mc.mode == MeshoptCompressionMode::Invalid, CgltfResult::InvalidGltf);

            assert_if!(
                mc.mode == MeshoptCompressionMode::Attributes && !(mc.stride % 4 == 0 && mc.stride <= 256),
                CgltfResult::InvalidGltf
            );

            assert_if!(
                mc.mode == MeshoptCompressionMode::Triangles && mc.count % 3 != 0,
                CgltfResult::InvalidGltf
            );

            let is_index = matches!(mc.mode, MeshoptCompressionMode::Triangles | MeshoptCompressionMode::Indices);

            assert_if!(is_index && mc.stride != 2 && mc.stride != 4, CgltfResult::InvalidGltf);

            assert_if!(is_index && mc.filter != MeshoptCompressionFilter::None, CgltfResult::InvalidGltf);

            assert_if!(
                mc.filter == MeshoptCompressionFilter::Octahedral && mc.stride != 4 && mc.stride != 8,
                CgltfResult::InvalidGltf
            );
            assert_if!(
                mc.filter == MeshoptCompressionFilter::Quaternion && mc.stride != 8,
                CgltfResult::InvalidGltf
            );
            assert_if!(
                mc.filter == MeshoptCompressionFilter::Color && mc.stride != 4 && mc.stride != 8,
                CgltfResult::InvalidGltf
            );
        }
    }

    for mesh in &data.meshes {
        let first_targets = mesh.primitives.first().map(|p| p.targets.len());

        if let Some(weights) = &mesh.weights {
            assert_if!(first_targets.is_some_and(|t| t != weights.len()), CgltfResult::InvalidGltf);
        }

        if let Some(target_names) = &mesh.target_names {
            assert_if!(first_targets.is_some_and(|t| t != target_names.len()), CgltfResult::InvalidGltf);
        }

        for prim in &mesh.primitives {
            assert_if!(prim.type_ == PrimitiveType::Invalid, CgltfResult::InvalidGltf);
            assert_if!(Some(prim.targets.len()) != first_targets, CgltfResult::InvalidGltf);

            let Some(first_attr) = prim.attributes.first() else {
                return Err(CgltfResult::InvalidGltf);
            };

            let first = &data.accessors[first_attr.data];

            assert_if!(first.count == 0, CgltfResult::InvalidGltf);

            for attr in &prim.attributes {
                assert_if!(data.accessors[attr.data].count != first.count, CgltfResult::InvalidGltf);
            }

            for target in &prim.targets {
                for attr in &target.attributes {
                    assert_if!(data.accessors[attr.data].count != first.count, CgltfResult::InvalidGltf);
                }
            }

            if let Some(indices) = prim.indices.map(|i| &data.accessors[i]) {
                assert_if!(
                    !matches!(
                        indices.component_type,
                        ComponentType::R8u | ComponentType::R16u | ComponentType::R32u
                    ),
                    CgltfResult::InvalidGltf
                );

                assert_if!(indices.type_ != Type::Scalar, CgltfResult::InvalidGltf);
                assert_if!(indices.stride != component_size(indices.component_type), CgltfResult::InvalidGltf);

                if let Some(bv) = indices.buffer_view
                    && data.buffers[data.buffer_views[bv].buffer].data.is_some()
                {
                    let index_bound = calc_index_bound(data, bv, indices.offset, indices.component_type, indices.count);

                    assert_if!(index_bound >= first.count, CgltfResult::DataTooShort);
                }
            }

            for mapping in &prim.mappings {
                assert_if!(mapping.variant >= data.variants.len(), CgltfResult::InvalidGltf);
            }
        }
    }

    for node in &data.nodes {
        if let Some(weights) = &node.weights
            && let Some(mesh) = node.mesh
        {
            let mesh = &data.meshes[mesh];
            assert_if!(
                mesh.primitives.first().is_some_and(|p| p.targets.len() != weights.len()),
                CgltfResult::InvalidGltf
            );
        }

        if node.has_mesh_gpu_instancing {
            assert_if!(node.mesh.is_none(), CgltfResult::InvalidGltf);

            let Some(first_attr) = node.mesh_gpu_instancing.attributes.first() else {
                return Err(CgltfResult::InvalidGltf);
            };

            let first = &data.accessors[first_attr.data];

            for attr in &node.mesh_gpu_instancing.attributes {
                assert_if!(data.accessors[attr.data].count != first.count, CgltfResult::InvalidGltf);
            }
        }
    }

    for node in &data.nodes {
        let parent = |n: Option<usize>| n.and_then(|n| data.nodes[n].parent);

        let mut p1 = node.parent;
        let mut p2 = parent(p1);

        while p1.is_some() && p2.is_some() {
            assert_if!(p1 == p2, CgltfResult::InvalidGltf);

            p1 = parent(p1);
            p2 = parent(parent(p2));
        }
    }

    for scene in &data.scenes {
        for &node in &scene.nodes {
            assert_if!(data.nodes[node].parent.is_some(), CgltfResult::InvalidGltf);
        }
    }

    for animation in &data.animations {
        for channel in &animation.channels {
            let Some(target_node) = channel.target_node else {
                continue;
            };

            let mut components = 1;

            if channel.target_path == AnimationPathType::Weights {
                let mesh = data.nodes[target_node].mesh.map(|m| &data.meshes[m]);
                let Some(prim) = mesh.and_then(|m| m.primitives.first()) else {
                    return Err(CgltfResult::InvalidGltf);
                };

                components = prim.targets.len();
            }

            let sampler = &animation.samplers[channel.sampler];
            let values = if sampler.interpolation == InterpolationType::CubicSpline {
                3
            } else {
                1
            };

            assert_if!(
                data.accessors[sampler.input].count.wrapping_mul(components).wrapping_mul(values) != data.accessors[sampler.output].count,
                CgltfResult::InvalidGltf
            );
        }
    }

    for variant in &data.variants {
        assert_if!(variant.name.is_none(), CgltfResult::InvalidGltf);
    }

    Ok(())
}

/// `cgltf_free`: releases all memory owned by `data`.
pub fn free(data: Data) {
    drop(data);
}

/// `cgltf_node_transform_local`
pub fn node_transform_local(node: &Node) -> [f32; 16] {
    if node.has_matrix {
        return node.matrix;
    }

    let [tx, ty, tz] = node.translation;
    let [qx, qy, qz, qw] = node.rotation;
    let [sx, sy, sz] = node.scale;

    [
        (1.0 - 2.0 * qy * qy - 2.0 * qz * qz) * sx,
        (2.0 * qx * qy + 2.0 * qz * qw) * sx,
        (2.0 * qx * qz - 2.0 * qy * qw) * sx,
        0.0,
        (2.0 * qx * qy - 2.0 * qz * qw) * sy,
        (1.0 - 2.0 * qx * qx - 2.0 * qz * qz) * sy,
        (2.0 * qy * qz + 2.0 * qx * qw) * sy,
        0.0,
        (2.0 * qx * qz + 2.0 * qy * qw) * sz,
        (2.0 * qy * qz - 2.0 * qx * qw) * sz,
        (1.0 - 2.0 * qx * qx - 2.0 * qy * qy) * sz,
        0.0,
        tx,
        ty,
        tz,
        1.0,
    ]
}

/// `cgltf_node_transform_world`: `node` may be a temporary whose `parent` refers into `nodes`.
pub fn node_transform_world(nodes: &[Node], node: &Node) -> [f32; 16] {
    let mut lm = node_transform_local(node);

    let mut parent = node.parent;
    let mut steps = 0;

    while let Some(p) = parent.and_then(|p| nodes.get(p)) {
        let pm = node_transform_local(p);

        for i in 0..4 {
            let l0 = lm[i * 4];
            let l1 = lm[i * 4 + 1];
            let l2 = lm[i * 4 + 2];

            let r0 = l0 * pm[0] + l1 * pm[4] + l2 * pm[8];
            let r1 = l0 * pm[1] + l1 * pm[5] + l2 * pm[9];
            let r2 = l0 * pm[2] + l1 * pm[6] + l2 * pm[10];

            lm[i * 4] = r0;
            lm[i * 4 + 1] = r1;
            lm[i * 4 + 2] = r2;
        }

        lm[12] += pm[12];
        lm[13] += pm[13];
        lm[14] += pm[14];

        parent = p.parent;

        // cycles are rejected by validate(); guard against unvalidated data
        steps += 1;
        if steps > nodes.len() {
            break;
        }
    }

    lm
}

/// `cgltf_buffer_view_data`: the view's bytes (decoded extension data or the buffer range starting at `offset`).
pub fn buffer_view_data<'a>(data: &'a Data, view: &'a BufferView) -> Option<&'a [u8]> {
    if let Some(d) = &view.data {
        return Some(d);
    }

    let buffer = data.buffers.get(view.buffer)?.data.as_deref()?;
    buffer.get(view.offset..)
}

/// `cgltf_find_accessor`
pub fn find_accessor(prim: &Primitive, type_: AttributeType, index: i32) -> Option<usize> {
    prim.attributes
        .iter()
        .find(|a| a.type_ == type_ && a.index == index)
        .map(|a| a.data)
}

/// `cgltf_num_components`
pub fn num_components(type_: Type) -> CgltfSize {
    match type_ {
        Type::Vec2 => 2,
        Type::Vec3 => 3,
        Type::Vec4 => 4,
        Type::Mat2 => 4,
        Type::Mat3 => 9,
        Type::Mat4 => 16,
        Type::Invalid | Type::Scalar => 1,
    }
}

/// `cgltf_component_size`
pub fn component_size(component_type: ComponentType) -> CgltfSize {
    match component_type {
        ComponentType::R8 | ComponentType::R8u => 1,
        ComponentType::R16 | ComponentType::R16u => 2,
        ComponentType::R32u | ComponentType::R32f => 4,
        ComponentType::Invalid => 0,
    }
}

/// `cgltf_calc_size`
pub fn calc_size(type_: Type, component_type: ComponentType) -> CgltfSize {
    let component_size = component_size(component_type);
    if type_ == Type::Mat2 && component_size == 1 {
        8 * component_size
    } else if type_ == Type::Mat3 && (component_size == 1 || component_size == 2) {
        12 * component_size
    } else {
        component_size * num_components(type_)
    }
}

fn bytes_at<const N: usize>(b: &[u8], offset: usize) -> Option<[u8; N]> {
    let s = b.get(offset..offset.checked_add(N)?)?;
    let mut out = [0u8; N];
    out.copy_from_slice(s);
    Some(out)
}

fn read_integer(b: &[u8], component_type: ComponentType) -> Option<i64> {
    Some(match component_type {
        ComponentType::R16 => i64::from(i16::from_le_bytes(bytes_at(b, 0)?)),
        ComponentType::R16u => i64::from(u16::from_le_bytes(bytes_at(b, 0)?)),
        ComponentType::R32u => i64::from(u32::from_le_bytes(bytes_at(b, 0)?)),
        ComponentType::R8 => i64::from(i8::from_le_bytes(bytes_at(b, 0)?)),
        ComponentType::R8u => i64::from(u8::from_le_bytes(bytes_at(b, 0)?)),
        _ => 0,
    })
}

/// `cgltf_component_read_index`
fn read_index(b: &[u8], component_type: ComponentType) -> Option<usize> {
    Some(match component_type {
        ComponentType::R16u => usize::from(u16::from_le_bytes(bytes_at(b, 0)?)),
        ComponentType::R32u => u32::from_le_bytes(bytes_at(b, 0)?) as usize,
        ComponentType::R8u => usize::from(u8::from_le_bytes(bytes_at(b, 0)?)),
        _ => 0,
    })
}

/// `cgltf_component_read_float`
fn read_float(b: &[u8], component_type: ComponentType, normalized: bool) -> Option<f32> {
    if component_type == ComponentType::R32f {
        return Some(f32::from_le_bytes(bytes_at(b, 0)?));
    }

    if normalized {
        return Some(match component_type {
            // note: glTF spec doesn't currently define normalized conversions for 32-bit integers
            ComponentType::R16 => f32::from(i16::from_le_bytes(bytes_at(b, 0)?)) / 32767.0,
            ComponentType::R16u => f32::from(u16::from_le_bytes(bytes_at(b, 0)?)) / 65535.0,
            ComponentType::R8 => f32::from(i8::from_le_bytes(bytes_at(b, 0)?)) / 127.0,
            ComponentType::R8u => f32::from(u8::from_le_bytes(bytes_at(b, 0)?)) / 255.0,
            _ => 0.0,
        });
    }

    Some(read_integer(b, component_type)? as f32)
}

/// `cgltf_element_read_float`; `out.len()` is the C `element_size`.
fn element_read_float(element: &[u8], type_: Type, component_type: ComponentType, normalized: bool, out: &mut [f32]) -> bool {
    let num_components = num_components(type_);

    if out.len() < num_components {
        return false;
    }

    // There are three special cases for component extraction, see #data-alignment in the 2.0 spec.

    let component_size = component_size(component_type);

    let offsets: &[usize] = match (type_, component_size) {
        (Type::Mat2, 1) => &[0, 1, 4, 5],
        (Type::Mat3, 1) => &[0, 1, 2, 4, 5, 6, 8, 9, 10],
        (Type::Mat3, 2) => &[0, 2, 4, 8, 10, 12, 16, 18, 20],
        _ => &[],
    };

    if !offsets.is_empty() {
        for (o, &offset) in out.iter_mut().zip(offsets) {
            let Some(v) = element.get(offset..).and_then(|b| read_float(b, component_type, normalized)) else {
                return false;
            };
            *o = v;
        }
        return true;
    }

    for (i, o) in out.iter_mut().take(num_components).enumerate() {
        let Some(v) = element
            .get(component_size * i..)
            .and_then(|b| read_float(b, component_type, normalized))
        else {
            return false;
        };
        *o = v;
    }
    true
}

/// `cgltf_find_sparse_index`
fn find_sparse_index<'a>(data: &'a Data, accessor: &Accessor, needle: usize) -> Option<&'a [u8]> {
    let sparse = &accessor.sparse;
    let index_data = buffer_view_data(data, data.buffer_views.get(sparse.indices_buffer_view)?)?;
    let value_data = buffer_view_data(data, data.buffer_views.get(sparse.values_buffer_view)?)?;

    let index_data = index_data.get(sparse.indices_byte_offset..)?;
    let value_data = value_data.get(sparse.values_byte_offset..)?;

    let index_stride = component_size(sparse.indices_component_type);

    let mut offset = 0;
    let mut length = sparse.count;

    while length != 0 {
        let rem = length % 2;
        length /= 2;

        let index = read_index(index_data.get((offset + length) * index_stride..)?, sparse.indices_component_type)?;
        offset += if index < needle { length + rem } else { 0 };
    }

    if offset == sparse.count {
        return None;
    }

    let index = read_index(index_data.get(offset * index_stride..)?, sparse.indices_component_type)?;
    if index == needle {
        value_data.get(offset * accessor.stride..)
    } else {
        None
    }
}

fn accessor_element<'a>(data: &'a Data, accessor: &Accessor, index: usize) -> Option<&'a [u8]> {
    let view = data.buffer_views.get(accessor.buffer_view?)?;
    let element = buffer_view_data(data, view)?;
    element.get(accessor.offset.checked_add(accessor.stride.checked_mul(index)?)?..)
}

/// `cgltf_accessor_read_float`: reads element `index` into `out` (`out.len()` is the C `element_size`).
pub fn accessor_read_float(data: &Data, accessor: &Accessor, index: usize, out: &mut [f32]) -> bool {
    if accessor.is_sparse
        && let Some(element) = find_sparse_index(data, accessor, index)
    {
        return element_read_float(element, accessor.type_, accessor.component_type, accessor.normalized, out);
    }
    if accessor.buffer_view.is_none() {
        out.fill(0.0);
        return true;
    }
    let Some(element) = accessor_element(data, accessor, index) else {
        return false;
    };
    element_read_float(element, accessor.type_, accessor.component_type, accessor.normalized, out)
}

/// `cgltf_accessor_unpack_floats`: `out == None` returns the number of floats required.
pub fn accessor_unpack_floats(data: &Data, accessor: &Accessor, out: Option<&mut [f32]>) -> CgltfSize {
    let floats_per_element = num_components(accessor.type_);
    let available_floats = accessor.count.saturating_mul(floats_per_element);
    let Some(out) = out else {
        return available_floats;
    };

    let float_count = available_floats.min(out.len());
    let element_count = float_count / floats_per_element;

    // First pass: convert each element in the base accessor.
    if accessor.buffer_view.is_none() {
        out[..element_count * floats_per_element].fill(0.0);
    } else {
        let Some(view) = accessor.buffer_view.and_then(|bv| data.buffer_views.get(bv)) else {
            return 0;
        };
        let Some(element) = buffer_view_data(data, view).and_then(|e| e.get(accessor.offset..)) else {
            return 0;
        };

        if accessor.component_type == ComponentType::R32f && accessor.stride == floats_per_element * 4 {
            let Some(bytes) = element.get(..element_count * floats_per_element * 4) else {
                return 0;
            };
            for (o, b) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*b);
            }
        } else {
            for index in 0..element_count {
                let dest = &mut out[index * floats_per_element..(index + 1) * floats_per_element];
                let Some(e) = element.get(index * accessor.stride..) else {
                    return 0;
                };
                if !element_read_float(e, accessor.type_, accessor.component_type, accessor.normalized, dest) {
                    return 0;
                }
            }
        }
    }

    // Second pass: write out each element in the sparse accessor.
    if accessor.is_sparse {
        let sparse = &accessor.sparse;

        let index_data = data
            .buffer_views
            .get(sparse.indices_buffer_view)
            .and_then(|v| buffer_view_data(data, v));
        let reader_head = data
            .buffer_views
            .get(sparse.values_buffer_view)
            .and_then(|v| buffer_view_data(data, v));

        let (Some(index_data), Some(reader_head)) = (index_data, reader_head) else {
            return 0;
        };

        let (Some(index_data), Some(reader_head)) = (
            index_data.get(sparse.indices_byte_offset..),
            reader_head.get(sparse.values_byte_offset..),
        ) else {
            return 0;
        };

        let index_stride = component_size(sparse.indices_component_type);
        for reader_index in 0..sparse.count {
            let writer_index = index_data
                .get(reader_index * index_stride..)
                .and_then(|b| read_index(b, sparse.indices_component_type));
            let element = reader_head.get(reader_index * accessor.stride..);

            let (Some(writer_index), Some(element)) = (writer_index, element) else {
                return 0;
            };

            let start = writer_index * floats_per_element;
            let Some(writer_head) = out.get_mut(start..start + floats_per_element) else {
                return 0;
            };

            if !element_read_float(element, accessor.type_, accessor.component_type, accessor.normalized, writer_head) {
                return 0;
            }
        }
    }

    element_count * floats_per_element
}

/// `cgltf_component_read_uint`
fn read_uint(b: &[u8], component_type: ComponentType) -> Option<u32> {
    Some(match component_type {
        ComponentType::R8 => i32::from(i8::from_le_bytes(bytes_at(b, 0)?)) as u32,
        ComponentType::R8u => u32::from(u8::from_le_bytes(bytes_at(b, 0)?)),
        ComponentType::R16 => i32::from(i16::from_le_bytes(bytes_at(b, 0)?)) as u32,
        ComponentType::R16u => u32::from(u16::from_le_bytes(bytes_at(b, 0)?)),
        ComponentType::R32u => u32::from_le_bytes(bytes_at(b, 0)?),
        _ => 0,
    })
}

/// `cgltf_element_read_uint`
fn element_read_uint(element: &[u8], type_: Type, component_type: ComponentType, out: &mut [u32]) -> bool {
    let num_components = num_components(type_);

    if out.len() < num_components {
        return false;
    }

    // Reading integer matrices is not a valid use case
    if matches!(type_, Type::Mat2 | Type::Mat3 | Type::Mat4) {
        return false;
    }

    let component_size = component_size(component_type);

    for (i, o) in out.iter_mut().take(num_components).enumerate() {
        let Some(v) = element.get(component_size * i..).and_then(|b| read_uint(b, component_type)) else {
            return false;
        };
        *o = v;
    }
    true
}

/// `cgltf_accessor_read_uint`
pub fn accessor_read_uint(data: &Data, accessor: &Accessor, index: usize, out: &mut [u32]) -> bool {
    if accessor.is_sparse
        && let Some(element) = find_sparse_index(data, accessor, index)
    {
        return element_read_uint(element, accessor.type_, accessor.component_type, out);
    }
    if accessor.buffer_view.is_none() {
        out.fill(0);
        return true;
    }
    let Some(element) = accessor_element(data, accessor, index) else {
        return false;
    };
    element_read_uint(element, accessor.type_, accessor.component_type, out)
}

/// `cgltf_accessor_read_index`
pub fn accessor_read_index(data: &Data, accessor: &Accessor, index: usize) -> CgltfSize {
    if accessor.is_sparse
        && let Some(element) = find_sparse_index(data, accessor, index)
    {
        return read_index(element, accessor.component_type).unwrap_or(0);
    }
    if accessor.buffer_view.is_none() {
        return 0;
    }
    // This is an error case, but we can't communicate the error with existing interface.
    accessor_element(data, accessor, index)
        .and_then(|e| read_index(e, accessor.component_type))
        .unwrap_or(0)
}

/// `cgltf_accessor_unpack_indices` with 32-bit output (`out_component_size == 4`, the only form gltfpack uses).
/// `out == None` returns the number of indices required.
pub fn accessor_unpack_indices(data: &Data, accessor: &Accessor, out: Option<&mut [u32]>) -> CgltfSize {
    let Some(out) = out else {
        return accessor.count;
    };

    let numbers_per_element = num_components(accessor.type_);
    let available_numbers = accessor.count.saturating_mul(numbers_per_element);

    let index_count = available_numbers.min(out.len());
    let index_component_size = component_size(accessor.component_type);

    if accessor.is_sparse {
        return 0;
    }
    if index_component_size > 4 {
        return 0;
    }
    let Some(element) = accessor_element(data, accessor, 0) else {
        return 0;
    };

    for (index, o) in out.iter_mut().take(index_count).enumerate() {
        let Some(v) = element
            .get(index * accessor.stride..)
            .and_then(|e| read_index(e, accessor.component_type))
        else {
            return 0;
        };
        *o = v as u32;
    }

    index_count
}
