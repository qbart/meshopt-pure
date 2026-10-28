//! glTF JSON/buffer output (`write.cpp`).

use crate::cgltf::{
    self, AlphaMode, AnimationPathType, AttributeType, ComponentType, Data, InterpolationType, LightType, PrimitiveType, TextureView, Type,
};
use crate::fileio::{get_file_name, get_full_path, write_file};
use crate::image::{mime_extension, read_image};
use crate::json::*;
use crate::material::{bits_eq, is_valid_transform};
use crate::node::decompose_transform;
use crate::splats::{splat_attribute_type, write_splat_stream};
use crate::stream::*;
use crate::types::*;

fn component_type(type_: ComponentType) -> &'static str {
    match type_ {
        ComponentType::R8 => "5120",
        ComponentType::R8u => "5121",
        ComponentType::R16 => "5122",
        ComponentType::R16u => "5123",
        ComponentType::R32u => "5125",
        ComponentType::R32f => "5126",
        ComponentType::Invalid => "0",
    }
}

fn shape_type(type_: Type) -> &'static str {
    match type_ {
        Type::Scalar => "SCALAR",
        Type::Vec2 => "VEC2",
        Type::Vec3 => "VEC3",
        Type::Vec4 => "VEC4",
        Type::Mat2 => "MAT2",
        Type::Mat3 => "MAT3",
        Type::Mat4 => "MAT4",
        Type::Invalid => "",
    }
}

pub fn attribute_type(type_: AttributeType) -> &'static str {
    match type_ {
        AttributeType::Position => "POSITION",
        AttributeType::Normal => "NORMAL",
        AttributeType::Tangent => "TANGENT",
        AttributeType::Texcoord => "TEXCOORD",
        AttributeType::Color => "COLOR",
        AttributeType::Joints => "JOINTS",
        AttributeType::Weights => "WEIGHTS",
        AttributeType::Custom => "CUSTOM",
        AttributeType::Invalid => "ATTRIBUTE",
    }
}

pub fn animation_path(type_: AnimationPathType) -> &'static str {
    match type_ {
        AnimationPathType::Translation => "translation",
        AnimationPathType::Rotation => "rotation",
        AnimationPathType::Scale => "scale",
        AnimationPathType::Weights => "weights",
        AnimationPathType::Invalid => "",
    }
}

fn light_type(type_: LightType) -> &'static str {
    match type_ {
        LightType::Directional => "directional",
        LightType::Point => "point",
        LightType::Spot => "spot",
        LightType::Invalid => "",
    }
}

fn alpha_mode(mode: AlphaMode) -> &'static str {
    match mode {
        AlphaMode::Opaque => "OPAQUE",
        AlphaMode::Mask => "MASK",
        AlphaMode::Blend => "BLEND",
    }
}

fn interpolation_type(type_: InterpolationType) -> &'static str {
    match type_ {
        InterpolationType::Linear => "LINEAR",
        InterpolationType::Step => "STEP",
        InterpolationType::CubicSpline => "CUBICSPLINE",
    }
}

fn compression_mode(mode: Compression) -> &'static str {
    match mode {
        Compression::Attribute => "ATTRIBUTES",
        Compression::Index => "TRIANGLES",
        Compression::IndexSequence => "INDICES",
        Compression::None => "",
    }
}

fn compression_filter(filter: Filter) -> &'static str {
    match filter {
        Filter::None => "NONE",
        Filter::Oct => "OCTAHEDRAL",
        Filter::Quat => "QUATERNION",
        Filter::Exp => "EXPONENTIAL",
        Filter::Color => "COLOR",
    }
}

/// `size_t(v)` of a signed value, as printed by `%zu`.
fn size(v: i32) -> usize {
    v as isize as usize
}

fn write_texture_info(
    json: &mut String,
    view: &TextureView,
    qt: Option<&QuantizationTexture>,
    textures: &[TextureInfo],
    scale: Option<&str>,
) {
    let Some(texture) = view.texture else {
        return;
    };

    let mut needs_transform = false;
    let mut transform = cgltf::TextureTransform {
        scale: [1.0; 2],
        ..Default::default()
    };

    if view.has_transform && is_valid_transform(&view.transform) {
        transform = view.transform;
        needs_transform = true;
    }

    if let Some(qt) = qt {
        let (rcos, rsin) = (transform.rotation.cos(), transform.rotation.sin());
        let (ox, oy) = (transform.scale[0] * qt.offset[0], transform.scale[1] * qt.offset[1]);

        let denom = ((1i32.wrapping_shl(qt.bits as u32)).wrapping_sub(1)) as f32;
        let norm = if qt.normalized { 65535.0 } else { 1.0 };

        transform.offset[0] += rcos * ox + rsin * oy;
        transform.offset[1] += -rsin * ox + rcos * oy;
        transform.scale[0] *= qt.scale[0] / denom * norm;
        transform.scale[1] *= qt.scale[1] / denom * norm;
        needs_transform = true;
    }

    append(json, "{\"index\":");
    append_size(json, size(textures[texture].remap));
    if view.texcoord != 0 {
        append(json, ",\"texCoord\":");
        append_size(json, size(view.texcoord));
    }
    if let Some(scale) = scale
        && view.scale != 1.0
    {
        append(json, ",\"");
        append(json, scale);
        append(json, "\":");
        append_float(json, view.scale);
    }
    if needs_transform {
        append(json, ",\"extensions\":{\"KHR_texture_transform\":{");
        append(json, "\"offset\":");
        append_floats(json, &transform.offset);
        append(json, ",\"scale\":");
        append_floats(json, &transform.scale);
        if transform.rotation != 0.0 {
            append(json, ",\"rotation\":");
            append_float(json, transform.rotation);
        }
        append(json, "}}");
    }
    append(json, "}");
}

const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

/// Writes `"name":<texture info>` if the view has a texture.
fn texture_field(
    json: &mut String,
    name: &str,
    view: &TextureView,
    qt: Option<&QuantizationTexture>,
    textures: &[TextureInfo],
    scale: Option<&str>,
) {
    if view.texture.is_some() {
        comma(json);
        append(json, "\"");
        append(json, name);
        append(json, "\":");
        write_texture_info(json, view, qt, textures, scale);
    }
}

fn float_field(json: &mut String, name: &str, value: f32, default: f32) {
    if value != default {
        comma(json);
        append(json, "\"");
        append(json, name);
        append(json, "\":");
        append_float(json, value);
    }
}

fn floats_field(json: &mut String, name: &str, value: &[f32], default: &[f32]) {
    if !bits_eq(value, &default[..value.len()]) {
        comma(json);
        append(json, "\"");
        append(json, name);
        append(json, "\":");
        append_floats(json, value);
    }
}

pub fn write_material(
    json: &mut String,
    material: &cgltf::Material,
    qp: Option<&QuantizationPosition>,
    qt: Option<&QuantizationTexture>,
    textures: &[TextureInfo],
) {
    if let Some(name) = material.name.as_deref().filter(|n| !n.is_empty()) {
        comma(json);
        append(json, "\"name\":\"");
        append(json, name);
        append(json, "\"");
    }

    if material.has_pbr_metallic_roughness {
        let pbr = &material.pbr_metallic_roughness;

        comma(json);
        append(json, "\"pbrMetallicRoughness\":{");
        floats_field(json, "baseColorFactor", &pbr.base_color_factor, &WHITE);
        texture_field(json, "baseColorTexture", &pbr.base_color_texture, qt, textures, None);
        float_field(json, "metallicFactor", pbr.metallic_factor, 1.0);
        float_field(json, "roughnessFactor", pbr.roughness_factor, 1.0);
        texture_field(
            json,
            "metallicRoughnessTexture",
            &pbr.metallic_roughness_texture,
            qt,
            textures,
            None,
        );
        append(json, "}");
    }

    texture_field(json, "normalTexture", &material.normal_texture, qt, textures, Some("scale"));
    texture_field(
        json,
        "occlusionTexture",
        &material.occlusion_texture,
        qt,
        textures,
        Some("strength"),
    );
    texture_field(json, "emissiveTexture", &material.emissive_texture, qt, textures, None);
    floats_field(json, "emissiveFactor", &material.emissive_factor, &BLACK);

    if material.alpha_mode != AlphaMode::Opaque {
        comma(json);
        append(json, "\"alphaMode\":\"");
        append(json, alpha_mode(material.alpha_mode));
        append(json, "\"");
    }

    float_field(json, "alphaCutoff", material.alpha_cutoff, 0.5);

    if material.double_sided {
        comma(json);
        append(json, "\"doubleSided\":true");
    }

    if material.has_pbr_specular_glossiness
        || material.has_clearcoat
        || material.has_transmission
        || material.has_ior
        || material.has_specular
        || material.has_sheen
        || material.has_volume
        || material.has_emissive_strength
        || material.has_iridescence
        || material.has_anisotropy
        || material.has_dispersion
        || material.has_diffuse_transmission
        || material.unlit
    {
        comma(json);
        append(json, "\"extensions\":{");

        if material.has_pbr_specular_glossiness {
            let pbr = &material.pbr_specular_glossiness;

            comma(json);
            append(json, "\"KHR_materials_pbrSpecularGlossiness\":{");
            texture_field(json, "diffuseTexture", &pbr.diffuse_texture, qt, textures, None);
            texture_field(
                json,
                "specularGlossinessTexture",
                &pbr.specular_glossiness_texture,
                qt,
                textures,
                None,
            );
            floats_field(json, "diffuseFactor", &pbr.diffuse_factor, &WHITE);
            floats_field(json, "specularFactor", &pbr.specular_factor, &WHITE);
            float_field(json, "glossinessFactor", pbr.glossiness_factor, 1.0);
            append(json, "}");
        }

        if material.has_clearcoat {
            let cc = &material.clearcoat;

            comma(json);
            append(json, "\"KHR_materials_clearcoat\":{");
            texture_field(json, "clearcoatTexture", &cc.clearcoat_texture, qt, textures, None);
            texture_field(
                json,
                "clearcoatRoughnessTexture",
                &cc.clearcoat_roughness_texture,
                qt,
                textures,
                None,
            );
            texture_field(
                json,
                "clearcoatNormalTexture",
                &cc.clearcoat_normal_texture,
                qt,
                textures,
                Some("scale"),
            );
            float_field(json, "clearcoatFactor", cc.clearcoat_factor, 0.0);
            float_field(json, "clearcoatRoughnessFactor", cc.clearcoat_roughness_factor, 0.0);
            append(json, "}");
        }

        if material.has_transmission {
            let tm = &material.transmission;

            comma(json);
            append(json, "\"KHR_materials_transmission\":{");
            texture_field(json, "transmissionTexture", &tm.transmission_texture, qt, textures, None);
            float_field(json, "transmissionFactor", tm.transmission_factor, 0.0);
            append(json, "}");
        }

        if material.has_ior {
            comma(json);
            append(json, "\"KHR_materials_ior\":{");
            append(json, "\"ior\":");
            append_float(json, material.ior.ior);
            append(json, "}");
        }

        if material.has_specular {
            let tm = &material.specular;

            comma(json);
            append(json, "\"KHR_materials_specular\":{");
            texture_field(json, "specularTexture", &tm.specular_texture, qt, textures, None);
            texture_field(json, "specularColorTexture", &tm.specular_color_texture, qt, textures, None);
            float_field(json, "specularFactor", tm.specular_factor, 1.0);
            floats_field(json, "specularColorFactor", &tm.specular_color_factor, &WHITE);
            append(json, "}");
        }

        if material.has_sheen {
            let tm = &material.sheen;

            comma(json);
            append(json, "\"KHR_materials_sheen\":{");
            texture_field(json, "sheenColorTexture", &tm.sheen_color_texture, qt, textures, None);
            texture_field(json, "sheenRoughnessTexture", &tm.sheen_roughness_texture, qt, textures, None);
            floats_field(json, "sheenColorFactor", &tm.sheen_color_factor, &BLACK);
            float_field(json, "sheenRoughnessFactor", tm.sheen_roughness_factor, 0.0);
            append(json, "}");
        }

        if material.has_volume {
            let tm = &material.volume;

            comma(json);
            append(json, "\"KHR_materials_volume\":{");
            texture_field(json, "thicknessTexture", &tm.thickness_texture, qt, textures, None);
            if tm.thickness_factor != 0.0 {
                // thickness is in mesh coordinate space which is rescaled by quantization
                comma(json);
                append(json, "\"thicknessFactor\":");
                append_float(json, tm.thickness_factor / qp.map_or(1.0, |q| q.node_scale));
            }
            floats_field(json, "attenuationColor", &tm.attenuation_color, &WHITE);
            float_field(json, "attenuationDistance", tm.attenuation_distance, f32::MAX);
            append(json, "}");
        }

        if material.has_emissive_strength {
            comma(json);
            append(json, "\"KHR_materials_emissive_strength\":{");
            float_field(json, "emissiveStrength", material.emissive_strength.emissive_strength, 1.0);
            append(json, "}");
        }

        if material.has_iridescence {
            let tm = &material.iridescence;

            comma(json);
            append(json, "\"KHR_materials_iridescence\":{");
            float_field(json, "iridescenceFactor", tm.iridescence_factor, 0.0);
            texture_field(json, "iridescenceTexture", &tm.iridescence_texture, qt, textures, None);
            float_field(json, "iridescenceIor", tm.iridescence_ior, 1.3);
            float_field(json, "iridescenceThicknessMinimum", tm.iridescence_thickness_min, 100.0);
            float_field(json, "iridescenceThicknessMaximum", tm.iridescence_thickness_max, 400.0);
            texture_field(
                json,
                "iridescenceThicknessTexture",
                &tm.iridescence_thickness_texture,
                qt,
                textures,
                None,
            );
            append(json, "}");
        }

        if material.has_anisotropy {
            let tm = &material.anisotropy;

            comma(json);
            append(json, "\"KHR_materials_anisotropy\":{");
            float_field(json, "anisotropyStrength", tm.anisotropy_strength, 0.0);
            float_field(json, "anisotropyRotation", tm.anisotropy_rotation, 0.0);
            texture_field(json, "anisotropyTexture", &tm.anisotropy_texture, qt, textures, None);
            append(json, "}");
        }

        if material.has_dispersion {
            comma(json);
            append(json, "\"KHR_materials_dispersion\":{");
            append(json, "\"dispersion\":");
            append_float(json, material.dispersion.dispersion);
            append(json, "}");
        }

        if material.has_diffuse_transmission {
            let tm = &material.diffuse_transmission;

            comma(json);
            append(json, "\"KHR_materials_diffuse_transmission\":{");
            float_field(json, "diffuseTransmissionFactor", tm.diffuse_transmission_factor, 0.0);
            texture_field(
                json,
                "diffuseTransmissionTexture",
                &tm.diffuse_transmission_texture,
                qt,
                textures,
                None,
            );
            floats_field(
                json,
                "diffuseTransmissionColorFactor",
                &tm.diffuse_transmission_color_factor,
                &WHITE,
            );
            texture_field(
                json,
                "diffuseTransmissionColorTexture",
                &tm.diffuse_transmission_color_texture,
                qt,
                textures,
                None,
            );
            append(json, "}");
        }

        if material.unlit {
            comma(json);
            append(json, "\"KHR_materials_unlit\":{}");
        }

        append(json, "}");
    }
}

pub fn get_buffer_view(
    views: &mut Vec<BufferView>,
    kind: BufferViewKind,
    filter: Filter,
    compression: Compression,
    stride: usize,
    variant: i32,
) -> usize {
    if variant >= 0
        && let Some(i) = views.iter().position(|v| {
            v.kind == kind && v.filter == filter && v.compression == compression && v.stride == stride && v.variant == variant
        })
    {
        return i;
    }

    views.push(BufferView {
        kind,
        filter,
        compression,
        stride,
        variant,
        ..Default::default()
    });

    views.len() - 1
}

#[allow(clippy::too_many_arguments)]
pub fn write_buffer_view(
    json: &mut String,
    kind: BufferViewKind,
    filter: Filter,
    count: usize,
    stride: usize,
    bin_offset: usize,
    bin_size: usize,
    compression: Compression,
    compressed_offset: usize,
    compressed_size: usize,
    meshopt_ext: &str,
) {
    // when compression is enabled, we store uncompressed data in buffer 1 and compressed data in buffer 0
    // when compression is disabled, we store uncompressed data in buffer 0
    let buffer = usize::from(compression != Compression::None);

    append(json, "{\"buffer\":");
    append_size(json, buffer);
    append(json, ",\"byteOffset\":");
    append_size(json, bin_offset);
    append(json, ",\"byteLength\":");
    append_size(json, bin_size);
    if kind == BufferViewKind::Vertex || kind == BufferViewKind::Splat {
        append(json, ",\"byteStride\":");
        append_size(json, stride);
    }
    if kind == BufferViewKind::Vertex || kind == BufferViewKind::Index || kind == BufferViewKind::Splat {
        append(json, ",\"target\":");
        append(json, if kind == BufferViewKind::Index { "34963" } else { "34962" });
    }
    if compression != Compression::None {
        append(json, ",\"extensions\":{");
        append(json, "\"");
        append(json, meshopt_ext);
        append(json, "\":{");
        append(json, "\"buffer\":0");
        append(json, ",\"byteOffset\":");
        append_size(json, compressed_offset);
        append(json, ",\"byteLength\":");
        append_size(json, compressed_size);
        append(json, ",\"byteStride\":");
        append_size(json, stride);
        append(json, ",\"mode\":\"");
        append(json, compression_mode(compression));
        append(json, "\"");
        if filter != Filter::None {
            append(json, ",\"filter\":\"");
            append(json, compression_filter(filter));
            append(json, "\"");
        }
        append(json, ",\"count\":");
        append_size(json, count);
        append(json, "}}");
    }
    append(json, "}");
}

#[allow(clippy::too_many_arguments)]
fn write_accessor(
    json: &mut String,
    view: usize,
    offset: usize,
    type_: Type,
    component_type_: ComponentType,
    normalized: bool,
    count: usize,
    minmax: Option<(&[f32], &[f32])>,
) {
    append(json, "{\"bufferView\":");
    append_size(json, view);
    append(json, ",\"byteOffset\":");
    append_size(json, offset);
    append(json, ",\"componentType\":");
    append(json, component_type(component_type_));
    append(json, ",\"count\":");
    append_size(json, count);
    append(json, ",\"type\":\"");
    append(json, shape_type(type_));
    append(json, "\"");

    if normalized {
        append(json, ",\"normalized\":true");
    }

    if let Some((min, max)) = minmax {
        append(json, ",\"min\":");
        append_floats(json, min);
        append(json, ",\"max\":");
        append_floats(json, max);
    }

    append(json, "}");
}

fn write_embedded_image(json: &mut String, views: &mut Vec<BufferView>, data: &[u8], mime_type: &str, kind: TextureKind) {
    let view = get_buffer_view(views, BufferViewKind::Image, Filter::None, Compression::None, 1, -1 - kind as i32);

    views[view].data = data.to_vec();

    append(json, "\"bufferView\":");
    append_size(json, view);
    append(json, ",\"mimeType\":\"");
    append(json, mime_type);
    append(json, "\"");
}

fn decode_uri(uri: &str) -> String {
    let result = cgltf::decode_uri(uri);
    result.split('\0').next().unwrap_or("").to_string()
}

pub fn write_sampler(json: &mut String, sampler: &cgltf::Sampler) {
    if sampler.mag_filter != cgltf::FILTER_TYPE_UNDEFINED {
        comma(json);
        append(json, "\"magFilter\":");
        append_size(json, size(sampler.mag_filter));
    }
    if sampler.min_filter != cgltf::FILTER_TYPE_UNDEFINED {
        comma(json);
        append(json, "\"minFilter\":");
        append_size(json, size(sampler.min_filter));
    }
    if sampler.wrap_s != cgltf::WRAP_MODE_REPEAT {
        comma(json);
        append(json, "\"wrapS\":");
        append_size(json, size(sampler.wrap_s));
    }
    if sampler.wrap_t != cgltf::WRAP_MODE_REPEAT {
        comma(json);
        append(json, "\"wrapT\":");
        append_size(json, size(sampler.wrap_t));
    }
}

fn write_image_error(json: &mut String, action: &str, index: usize, uri: Option<&str>, reason: Option<&str>) {
    append(json, "\"uri\":\"");
    append(json, "data:image/png;base64,ERR/");
    append(json, "\"");

    eprintln!(
        "Warning: unable to {action} image {} ({}), skipping{}{}{}",
        index as i32,
        uri.unwrap_or("embedded"),
        if reason.is_some() { " (" } else { "" },
        reason.unwrap_or(""),
        if reason.is_some() { ")" } else { "" }
    );
}

#[allow(clippy::too_many_arguments)]
fn write_image_data(
    json: &mut String,
    views: &mut Vec<BufferView>,
    index: usize,
    uri: Option<&str>,
    mime_type: &str,
    contents: &[u8],
    output_path: Option<&str>,
    kind: TextureKind,
    embed: bool,
) {
    let data_uri = uri.is_some_and(|u| u.starts_with("data:"));

    if let (false, Some(uri), false, Some(output_path)) = (embed, uri, data_uri, output_path) {
        let file_name = get_file_name(uri) + mime_extension(mime_type);
        let file_path = get_full_path(&decode_uri(&file_name), output_path);

        if write_file(&file_path, contents) {
            append(json, "\"uri\":\"");
            append(json, &file_name);
            append(json, "\"");
        } else {
            write_image_error(json, "save", index, Some(uri), Some(&file_path));
        }
    } else {
        write_embedded_image(json, views, contents, mime_type, kind);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn write_image(
    json: &mut String,
    views: &mut Vec<BufferView>,
    data: &Data,
    image: &cgltf::Image,
    info: &ImageInfo,
    encoded: Option<&[u8]>,
    index: usize,
    input_path: Option<&str>,
    output_path: Option<&str>,
    settings: &Settings,
) {
    if let Some(name) = image.name.as_deref().filter(|n| !n.is_empty()) {
        append(json, "\"name\":\"");
        append(json, name);
        append(json, "\",");
    }

    let uri = image.uri.as_deref();

    if let Some(encoded) = encoded {
        let mime_type = if settings.texture_mode[info.kind as usize] == TextureMode::WebP {
            "image/webp"
        } else {
            "image/ktx2"
        };

        // image was pre-encoded via encodeImages (which might have failed!)
        if encoded.starts_with(b"error") {
            write_image_error(json, "encode", index, uri, Some(&String::from_utf8_lossy(encoded)));
        } else {
            write_image_data(
                json,
                views,
                index,
                uri,
                mime_type,
                encoded,
                output_path,
                info.kind,
                settings.texture_embed,
            );
        }
        return;
    }

    let data_uri = uri.is_some_and(|u| u.starts_with("data:"));

    if let Some(uri) = uri
        && !data_uri
        && settings.texture_ref
    {
        // fast-path: we don't need to read the image to memory
        append(json, "\"uri\":\"");
        append(json, uri);
        append(json, "\"");
        return;
    }

    let Some((img_data, mime_type)) = read_image(data, image, input_path) else {
        write_image_error(json, "read", index, uri, None);
        return;
    };

    write_image_data(
        json,
        views,
        index,
        uri,
        &mime_type,
        &img_data,
        output_path,
        info.kind,
        settings.texture_embed,
    );
}

pub fn write_texture(json: &mut String, texture: &cgltf::Texture, info: Option<&ImageInfo>, settings: &Settings) {
    if let Some(sampler) = texture.sampler {
        append(json, "\"sampler\":");
        append_size(json, sampler);
    }

    if let Some(image) = texture.image {
        if let Some(info) = info
            && settings.texture_mode[info.kind as usize] != TextureMode::Raw
        {
            let texture_ext = if settings.texture_mode[info.kind as usize] == TextureMode::WebP {
                "EXT_texture_webp"
            } else {
                "KHR_texture_basisu"
            };

            comma(json);
            append(json, "\"extensions\":{\"");
            append(json, texture_ext);
            append(json, "\":{\"source\":");
            append_size(json, image);
            append(json, "}}");

            return; // skip input basisu image if present, as we replace it with the one we encoded
        }

        comma(json);
        append(json, "\"source\":");
        append_size(json, image);
    }

    if let Some(image) = texture.basisu_image {
        comma(json);
        append(json, "\"extensions\":{\"KHR_texture_basisu\":{\"source\":");
        append_size(json, image);
        append(json, "}}");
    } else if let Some(image) = texture.webp_image {
        comma(json);
        append(json, "\"extensions\":{\"EXT_texture_webp\":{\"source\":");
        append_size(json, image);
        append(json, "}}");
    }
}

fn write_primitive_accessor(
    json_accessors: &mut String,
    stream: &Stream,
    view: usize,
    offset: usize,
    format: &StreamFormat,
    qp: &QuantizationPosition,
    settings: &Settings,
) {
    comma(json_accessors);
    if stream.type_ == AttributeType::Position {
        let (min, max) = get_position_bounds(stream, qp, settings);

        write_accessor(
            json_accessors,
            view,
            offset,
            format.type_,
            format.component_type,
            format.normalized,
            stream.data.len(),
            Some((&min, &max)),
        );
    } else {
        write_accessor(
            json_accessors,
            view,
            offset,
            format.type_,
            format.component_type,
            format.normalized,
            stream.data.len(),
            None,
        );
    }
}

fn write_primitive_attribute(json: &mut String, stream: &Stream, accessor: usize) {
    comma(json);
    append(json, "\"");
    if stream.type_ == AttributeType::Custom
        && let Some(name) = &stream.custom_name
    {
        append(json, name);
    } else if stream.type_ == AttributeType::Custom && stream.splats != 0 {
        let attr = splat_attribute_type(stream.splats);

        append(json, "KHR_gaussian_splatting:");
        append(json, attr);
        if attr.contains("DEGREE") {
            append(json, "_");
            append_size(json, size(stream.index));
        }
    } else {
        append(json, attribute_type(stream.type_));
        if stream.type_ != AttributeType::Position && stream.type_ != AttributeType::Normal && stream.type_ != AttributeType::Tangent {
            append(json, "_");
            append_size(json, size(stream.index));
        }
    }
    append(json, "\":");
    append_size(json, accessor);
}

#[allow(clippy::too_many_arguments)]
fn write_mesh_attributes_interleaved(
    json: &mut String,
    views: &mut Vec<BufferView>,
    json_accessors: &mut String,
    accr_offset: &mut usize,
    mesh: &Mesh,
    target: i32,
    qp: &QuantizationPosition,
    qt: &QuantizationTexture,
    settings: &Settings,
) {
    struct Attribute<'a> {
        stream: &'a Stream,
        format: StreamFormat,
        data: Vec<u8>,
    }

    let mut attributes = Vec::new();
    let mut stride = 0;

    for stream in &mesh.streams {
        if stream.target != target {
            continue;
        }

        let mut data = Vec::new();
        let format = write_vertex_stream(&mut data, stream, qp, qt, settings, /* filters= */ false);

        stride += format.stride;
        attributes.push(Attribute { stream, format, data });
    }

    let compression = if settings.compress {
        Compression::Attribute
    } else {
        Compression::None
    };
    let view = get_buffer_view(views, BufferViewKind::Vertex, Filter::None, compression, stride, 0);
    let mut offset = views[view].data.len();

    let vertex_count = mesh.streams[0].data.len();

    let vd = &mut views[view].data;
    vd.reserve(stride * vertex_count);

    for i in 0..vertex_count {
        for attr in &attributes {
            let s = attr.format.stride;
            vd.extend_from_slice(attr.data.get(i * s..(i + 1) * s).unwrap_or(&[]));
        }
    }

    for attr in &attributes {
        write_primitive_accessor(json_accessors, attr.stream, view, offset, &attr.format, qp, settings);

        let vertex_accr = *accr_offset;
        *accr_offset += 1;
        write_primitive_attribute(json, attr.stream, vertex_accr);

        offset += attr.format.stride;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn write_mesh_attributes(
    json: &mut String,
    views: &mut Vec<BufferView>,
    json_accessors: &mut String,
    accr_offset: &mut usize,
    mesh: &Mesh,
    target: i32,
    qp: &QuantizationPosition,
    qt: &QuantizationTexture,
    settings: &Settings,
) {
    if settings.mesh_interleaved {
        return write_mesh_attributes_interleaved(json, views, json_accessors, accr_offset, mesh, target, qp, qt, settings);
    }

    let mut scratch = Vec::new();

    for stream in &mesh.streams {
        if stream.target != target {
            continue;
        }

        scratch.clear();
        let format = write_vertex_stream(&mut scratch, stream, qp, qt, settings, true);
        let compression = if settings.compress {
            Compression::Attribute
        } else {
            Compression::None
        };

        let view = get_buffer_view(
            views,
            BufferViewKind::Vertex,
            format.filter,
            compression,
            format.stride,
            stream.type_ as i32,
        );
        let offset = views[view].data.len();
        views[view].data.extend_from_slice(&scratch);

        write_primitive_accessor(json_accessors, stream, view, offset, &format, qp, settings);

        let vertex_accr = *accr_offset;
        *accr_offset += 1;
        write_primitive_attribute(json, stream, vertex_accr);
    }
}

pub fn write_mesh_indices(
    views: &mut Vec<BufferView>,
    json_accessors: &mut String,
    accr_offset: &mut usize,
    indices: &[u32],
    type_: PrimitiveType,
    settings: &Settings,
) -> usize {
    let mut scratch = Vec::new();
    let format = write_index_stream(&mut scratch, indices);
    let compression = if !settings.compress {
        Compression::None
    } else if type_ == PrimitiveType::Triangles {
        Compression::Index
    } else {
        Compression::IndexSequence
    };

    let view = get_buffer_view(views, BufferViewKind::Index, Filter::None, compression, format.stride, 0);
    let offset = views[view].data.len();
    views[view].data.extend_from_slice(&scratch);

    comma(json_accessors);
    write_accessor(
        json_accessors,
        view,
        offset,
        format.type_,
        format.component_type,
        format.normalized,
        indices.len(),
        None,
    );

    let index_accr = *accr_offset;
    *accr_offset += 1;
    index_accr
}

#[allow(clippy::too_many_arguments)]
pub fn write_mesh_geometry(
    json: &mut String,
    views: &mut Vec<BufferView>,
    json_accessors: &mut String,
    accr_offset: &mut usize,
    mesh: &Mesh,
    qp: &QuantizationPosition,
    qt: &QuantizationTexture,
    settings: &Settings,
) {
    append(json, "{\"attributes\":{");
    write_mesh_attributes(json, views, json_accessors, accr_offset, mesh, 0, qp, qt, settings);
    append(json, "}");
    if mesh.type_ != PrimitiveType::Triangles {
        append(json, ",\"mode\":");
        append_size(json, (mesh.type_ as usize).wrapping_sub(PrimitiveType::Points as usize));
    }
    if mesh.targets != 0 {
        append(json, ",\"targets\":[");
        for j in 0..mesh.targets {
            comma(json);
            append(json, "{");
            write_mesh_attributes(json, views, json_accessors, accr_offset, mesh, (1 + j) as i32, qp, qt, settings);
            append(json, "}");
        }
        append(json, "]");
    }

    if !mesh.indices.is_empty() {
        let index_accr = write_mesh_indices(views, json_accessors, accr_offset, &mesh.indices, mesh.type_, settings);

        append(json, ",\"indices\":");
        append_size(json, index_accr);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn write_mesh_splats(
    json: &mut String,
    views: &mut Vec<BufferView>,
    json_accessors: &mut String,
    accr_offset: &mut usize,
    mesh: &Mesh,
    qp: &QuantizationPosition,
    settings: &Settings,
) {
    let mut scratch = Vec::new();

    append(json, "{\"attributes\":{");

    for stream in &mesh.streams {
        scratch.clear();
        let format = write_splat_stream(&mut scratch, stream, settings);
        let compression = if settings.compress {
            Compression::Attribute
        } else {
            Compression::None
        };

        let view = get_buffer_view(
            views,
            BufferViewKind::Splat,
            format.filter,
            compression,
            format.stride,
            stream.splats,
        );
        let offset = views[view].data.len();
        views[view].data.extend_from_slice(&scratch);

        write_primitive_accessor(json_accessors, stream, view, offset, &format, qp, settings);

        let vertex_accr = *accr_offset;
        *accr_offset += 1;
        write_primitive_attribute(json, stream, vertex_accr);
    }

    append(json, "},\"mode\":0,\"extensions\":{\"KHR_gaussian_splatting\":{");
    append(json, "\"kernel\":\"ellipse\",\"colorSpace\":\"srgb_rec709_display\"");
    append(json, "}}");
}

fn write_animation_time(
    views: &mut Vec<BufferView>,
    json_accessors: &mut String,
    accr_offset: &mut usize,
    time: &[f32],
    settings: &Settings,
) -> usize {
    let mut scratch = Vec::new();
    let format = write_time_stream(&mut scratch, time);
    let compression = if settings.compress {
        Compression::Attribute
    } else {
        Compression::None
    };

    let view = get_buffer_view(views, BufferViewKind::Time, Filter::None, compression, format.stride, 0);
    let offset = views[view].data.len();
    views[view].data.extend_from_slice(&scratch);

    let front = time.first().copied().unwrap_or(0.0);
    let back = time.last().copied().unwrap_or(0.0);

    comma(json_accessors);
    write_accessor(
        json_accessors,
        view,
        offset,
        Type::Scalar,
        format.component_type,
        format.normalized,
        time.len(),
        Some((&[front], &[back])),
    );

    let time_accr = *accr_offset;
    *accr_offset += 1;
    time_accr
}

fn write_animation_time_range(
    views: &mut Vec<BufferView>,
    json_accessors: &mut String,
    accr_offset: &mut usize,
    mint: f32,
    frames: i32,
    period: f32,
    settings: &Settings,
) -> usize {
    let time: Vec<f32> = (0..frames.max(0)).map(|j| mint + j as f32 * period).collect();

    write_animation_time(views, json_accessors, accr_offset, &time, settings)
}

pub fn write_joint_bind_matrices(
    views: &mut Vec<BufferView>,
    json_accessors: &mut String,
    accr_offset: &mut usize,
    data: &Data,
    skin: &cgltf::Skin,
    qp: &QuantizationPosition,
    settings: &Settings,
) -> usize {
    let mut scratch = Vec::new();

    for j in 0..skin.joints.len() {
        let mut transform: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

        if let Some(ibm) = skin.inverse_bind_matrices {
            cgltf::accessor_read_float(data, &data.accessors[ibm], j, &mut transform);
        }

        if settings.quantize && !settings.pos_float {
            // pos_offset has to be applied first, thus it results in an offset rotated by the bind matrix
            transform[12] += qp.offset[0] * transform[0] + qp.offset[1] * transform[4] + qp.offset[2] * transform[8];
            transform[13] += qp.offset[0] * transform[1] + qp.offset[1] * transform[5] + qp.offset[2] * transform[9];
            transform[14] += qp.offset[0] * transform[2] + qp.offset[1] * transform[6] + qp.offset[2] * transform[10];

            // node_scale will be applied before the rotation/scale from transform
            for t in transform.iter_mut().take(12) {
                *t *= qp.node_scale;
            }
        }

        push_f32s(&mut scratch, &transform);
    }

    let compression = if settings.compress {
        Compression::Attribute
    } else {
        Compression::None
    };

    let view = get_buffer_view(views, BufferViewKind::Skin, Filter::None, compression, 64, 0);
    let offset = views[view].data.len();
    views[view].data.extend_from_slice(&scratch);

    comma(json_accessors);
    write_accessor(
        json_accessors,
        view,
        offset,
        Type::Mat4,
        ComponentType::R32f,
        false,
        skin.joints.len(),
        None,
    );

    let matrix_accr = *accr_offset;
    *accr_offset += 1;
    matrix_accr
}

fn write_instance_data(
    views: &mut Vec<BufferView>,
    json_accessors: &mut String,
    type_: AnimationPathType,
    data: &[Attr],
    settings: &Settings,
) {
    let compression = if settings.compress {
        Compression::Attribute
    } else {
        Compression::None
    };

    let mut scratch = Vec::new();
    let format = write_keyframe_stream(&mut scratch, type_, data, settings, false);

    let view = get_buffer_view(
        views,
        BufferViewKind::Instance,
        format.filter,
        compression,
        format.stride,
        type_ as i32,
    );
    let offset = views[view].data.len();
    views[view].data.extend_from_slice(&scratch);

    comma(json_accessors);
    write_accessor(
        json_accessors,
        view,
        offset,
        format.type_,
        format.component_type,
        format.normalized,
        data.len(),
        None,
    );
}

pub fn write_instances(
    views: &mut Vec<BufferView>,
    json_accessors: &mut String,
    accr_offset: &mut usize,
    instances: &[Instance],
    qp: &QuantizationPosition,
    has_color: bool,
    settings: &Settings,
) -> usize {
    let mut position = vec![[0.0f32; 4]; instances.len()];
    let mut rotation = vec![[0.0f32; 4]; instances.len()];
    let mut scale = vec![[0.0f32; 4]; instances.len()];

    let mut color = Stream {
        type_: AttributeType::Color,
        ..Default::default()
    };
    if has_color {
        color.data = vec![[0.0; 4]; instances.len()];
    }

    for (i, instance) in instances.iter().enumerate() {
        let (mut t, mut r, mut s) = ([0.0; 3], [0.0; 4], [0.0; 3]);
        decompose_transform(&mut t, &mut r, &mut s, &instance.transform);
        position[i][..3].copy_from_slice(&t);
        rotation[i] = r;
        scale[i][..3].copy_from_slice(&s);

        if settings.quantize && !settings.pos_float {
            let transform = &instance.transform;

            // pos_offset has to be applied first, thus it results in an offset rotated by the instance matrix
            position[i][0] += qp.offset[0] * transform[0] + qp.offset[1] * transform[4] + qp.offset[2] * transform[8];
            position[i][1] += qp.offset[0] * transform[1] + qp.offset[1] * transform[5] + qp.offset[2] * transform[9];
            position[i][2] += qp.offset[0] * transform[2] + qp.offset[1] * transform[6] + qp.offset[2] * transform[10];

            // node_scale will be applied before the rotation/scale from transform
            for k in 0..3 {
                scale[i][k] *= qp.node_scale;
            }
        }

        if has_color {
            color.data[i] = instance.color;
        }
    }

    write_instance_data(views, json_accessors, AnimationPathType::Translation, &position, settings);
    write_instance_data(views, json_accessors, AnimationPathType::Rotation, &rotation, settings);
    write_instance_data(views, json_accessors, AnimationPathType::Scale, &scale, settings);

    let result = *accr_offset;
    *accr_offset += 3;

    if has_color {
        let compression = if settings.compress {
            Compression::Attribute
        } else {
            Compression::None
        };

        let mut scratch = Vec::new();
        let format = write_vertex_stream(
            &mut scratch,
            &color,
            &QuantizationPosition::default(),
            &QuantizationTexture::default(),
            settings,
            true,
        );

        let view = get_buffer_view(views, BufferViewKind::Instance, format.filter, compression, format.stride, 0);
        let offset = views[view].data.len();
        views[view].data.extend_from_slice(&scratch);

        comma(json_accessors);
        write_accessor(
            json_accessors,
            view,
            offset,
            format.type_,
            format.component_type,
            format.normalized,
            instances.len(),
            None,
        );
        *accr_offset += 1;
    }

    result
}

pub fn write_mesh_node(
    json: &mut String,
    mesh_offset: usize,
    node: Option<&cgltf::Node>,
    skin: Option<usize>,
    qp: Option<&QuantizationPosition>,
) {
    comma(json);
    append(json, "{\"mesh\":");
    append_size(json, mesh_offset);
    if let Some(skin) = skin {
        append(json, ",\"skin\":");
        append_size(json, skin);
    }
    if let Some(qp) = qp {
        append(json, ",\"translation\":");
        append_floats(json, &qp.offset);
        append(json, ",\"scale\":[");
        append_float(json, qp.node_scale);
        append(json, ",");
        append_float(json, qp.node_scale);
        append(json, ",");
        append_float(json, qp.node_scale);
        append(json, "]");
    }
    if let Some(weights) = node.and_then(|n| n.weights.as_deref()).filter(|w| !w.is_empty()) {
        append(json, ",\"weights\":");
        append_floats(json, weights);
    }
    append(json, "}");
}

pub fn write_mesh_node_instanced(json: &mut String, mesh_offset: usize, accr_offset: usize, has_color: bool) {
    comma(json);
    append(json, "{\"mesh\":");
    append_size(json, mesh_offset);
    append(json, ",\"extensions\":{\"EXT_mesh_gpu_instancing\":{\"attributes\":{");

    comma(json);
    append(json, "\"TRANSLATION\":");
    append_size(json, accr_offset);

    comma(json);
    append(json, "\"ROTATION\":");
    append_size(json, accr_offset + 1);

    comma(json);
    append(json, "\"SCALE\":");
    append_size(json, accr_offset + 2);

    if has_color {
        comma(json);
        append(json, "\"_COLOR_0\":");
        append_size(json, accr_offset + 3);
    }

    append(json, "}}}");
    append(json, "}");
}

pub fn write_skin(json: &mut String, skin: &cgltf::Skin, matrix_accr: usize, nodes: &[NodeInfo]) {
    comma(json);
    append(json, "{");
    if let Some(name) = skin.name.as_deref().filter(|n| !n.is_empty()) {
        append(json, "\"name\":\"");
        append(json, name);
        append(json, "\",");
    }
    append(json, "\"joints\":[");
    for &joint in &skin.joints {
        comma(json);
        append_size(json, size(nodes[joint].remap));
    }
    append(json, "]");
    append(json, ",\"inverseBindMatrices\":");
    append_size(json, matrix_accr);
    if let Some(skeleton) = skin.skeleton {
        comma(json);
        append(json, "\"skeleton\":");
        append_size(json, size(nodes[skeleton].remap));
    }
    append(json, "}");
}

pub fn write_node(json: &mut String, node: &cgltf::Node, index: usize, nodes: &[NodeInfo]) {
    let ni = &nodes[index];

    if let Some(name) = node.name.as_deref().filter(|n| !n.is_empty()) {
        comma(json);
        append(json, "\"name\":\"");
        append(json, name);
        append(json, "\"");
    }
    if node.has_translation {
        comma(json);
        append(json, "\"translation\":");
        append_floats(json, &node.translation);
    }
    if node.has_rotation {
        comma(json);
        append(json, "\"rotation\":");
        append_floats(json, &node.rotation);
    }
    if node.has_scale {
        comma(json);
        append(json, "\"scale\":");
        append_floats(json, &node.scale);
    }
    if node.has_matrix {
        comma(json);
        append(json, "\"matrix\":");
        append_floats(json, &node.matrix);
    }

    let has_children = !ni.mesh_nodes.is_empty() || node.children.iter().any(|&c| nodes[c].keep);

    if has_children {
        comma(json);
        append(json, "\"children\":[");
        for &child in &node.children {
            let ci = &nodes[child];

            if ci.keep {
                comma(json);
                append_size(json, size(ci.remap));
            }
        }
        for &mesh_node in &ni.mesh_nodes {
            comma(json);
            append_size(json, mesh_node);
        }
        append(json, "]");
    }
    if ni.has_mesh {
        comma(json);
        append(json, "\"mesh\":");
        append_size(json, ni.mesh_index);
        if let Some(skin) = ni.mesh_skin {
            append(json, ",\"skin\":");
            append_size(json, skin);
        }
        if let Some(weights) = node.weights.as_deref().filter(|w| !w.is_empty()) {
            append(json, ",\"weights\":");
            append_floats(json, weights);
        }
    }
    if let Some(camera) = node.camera {
        comma(json);
        append(json, "\"camera\":");
        append_size(json, camera);
    }
    if let Some(light) = node.light {
        comma(json);
        append(json, "\"extensions\":{\"KHR_lights_punctual\":{\"light\":");
        append_size(json, light);
        append(json, "}}");
    }
}

#[allow(clippy::too_many_arguments)]
pub fn write_animation(
    json: &mut String,
    views: &mut Vec<BufferView>,
    json_accessors: &mut String,
    accr_offset: &mut usize,
    animation: &Animation,
    i: usize,
    nodes: &[NodeInfo],
    settings: &Settings,
) {
    let tracks: Vec<&Track> = animation
        .tracks
        .iter()
        .filter(|track| {
            let ni = &nodes[track.node];

            ni.keep && (settings.anim_const || (ni.animated_path_mask & (1 << track.path as u32)) != 0)
        })
        .collect();

    if tracks.is_empty() {
        eprintln!(
            "Warning: ignoring animation {} ({}) because it has no tracks with motion; use -ac to override",
            i as i32,
            animation.name.as_deref().unwrap_or("")
        );
        return;
    }

    let mut needs_time = false;
    let mut needs_pose = false;

    for track in &tracks {
        needs_time = needs_time || (track.time.is_empty() && !track.constant);
        needs_pose = needs_pose || track.constant;
    }

    let needs_range = needs_pose && !needs_time && animation.frames > 1;

    needs_pose = needs_pose && !(needs_range && tracks.len() == 1);

    let animation_period = 1.0 / settings.anim_freq as f32;
    let animation_length = (animation.frames - 1) as f32 * animation_period;

    let time_accr = if needs_time {
        write_animation_time_range(
            views,
            json_accessors,
            accr_offset,
            animation.start,
            animation.frames,
            animation_period,
            settings,
        )
    } else {
        0
    };
    let pose_accr = if needs_pose {
        write_animation_time_range(views, json_accessors, accr_offset, animation.start, 1, 0.0, settings)
    } else {
        0
    };
    let range_accr = if needs_range {
        write_animation_time_range(views, json_accessors, accr_offset, animation.start, 2, animation_length, settings)
    } else {
        0
    };

    let mut json_samplers = String::new();
    let mut json_channels = String::new();

    let mut track_offset = 0usize;

    let mut last_track_time_accr = 0usize;
    let mut last_track_time: Option<&Track> = None;

    for (j, track) in tracks.iter().enumerate() {
        let range = needs_range && j == 0;
        let range_size = if range { 2 } else { 1 };

        let mut track_time_accr = time_accr;
        if !track.time.is_empty() {
            // reuse time accessors between consecutive tracks if possible
            if let Some(last) = last_track_time
                && track.time == last.time
            {
                track_time_accr = last_track_time_accr;
            } else {
                track_time_accr = write_animation_time(views, json_accessors, accr_offset, &track.time, settings);
                last_track_time_accr = track_time_accr;
                last_track_time = Some(track);
            }
        }

        let mut scratch = Vec::new();
        let format = write_keyframe_stream(
            &mut scratch,
            track.path,
            &track.data,
            settings,
            track.interpolation == InterpolationType::CubicSpline,
        );

        if range {
            let copy = scratch.clone();
            scratch.extend_from_slice(&copy);
        }

        let compression = if settings.compress && track.path != AnimationPathType::Weights {
            Compression::Attribute
        } else {
            Compression::None
        };

        let view = get_buffer_view(
            views,
            BufferViewKind::Keyframe,
            format.filter,
            compression,
            format.stride,
            track.path as i32,
        );
        let offset = views[view].data.len();
        views[view].data.extend_from_slice(&scratch);

        comma(json_accessors);
        write_accessor(
            json_accessors,
            view,
            offset,
            format.type_,
            format.component_type,
            format.normalized,
            track.data.len() * range_size,
            None,
        );

        let data_accr = *accr_offset;
        *accr_offset += 1;

        comma(&mut json_samplers);
        append(&mut json_samplers, "{\"input\":");
        append_size(
            &mut json_samplers,
            if range {
                range_accr
            } else if track.constant {
                pose_accr
            } else {
                track_time_accr
            },
        );
        append(&mut json_samplers, ",\"output\":");
        append_size(&mut json_samplers, data_accr);
        if track.interpolation != InterpolationType::Linear {
            append(&mut json_samplers, ",\"interpolation\":\"");
            append(&mut json_samplers, interpolation_type(track.interpolation));
            append(&mut json_samplers, "\"");
        }
        append(&mut json_samplers, "}");

        let tni = &nodes[track.node];
        let mut target_node = size(tni.remap);

        // when animating morph weights, quantization may move mesh assignments to a mesh node in which case we need to move the animation output
        if track.path == AnimationPathType::Weights && tni.mesh_nodes.len() == 1 {
            target_node = tni.mesh_nodes[0];
        }

        comma(&mut json_channels);
        append(&mut json_channels, "{\"sampler\":");
        append_size(&mut json_channels, track_offset);
        append(&mut json_channels, ",\"target\":{\"node\":");
        append_size(&mut json_channels, target_node);
        append(&mut json_channels, ",\"path\":\"");
        append(&mut json_channels, animation_path(track.path));
        append(&mut json_channels, "\"}}");

        track_offset += 1;
    }

    comma(json);
    append(json, "{");
    if let Some(name) = animation.name.as_deref().filter(|n| !n.is_empty()) {
        append(json, "\"name\":\"");
        append(json, name);
        append(json, "\",");
    }
    append(json, "\"samplers\":[");
    append(json, &json_samplers);
    append(json, "],\"channels\":[");
    append(json, &json_channels);
    append(json, "]}");
}

pub fn write_camera(json: &mut String, camera: &cgltf::Camera) {
    comma(json);
    append(json, "{");

    match camera.type_ {
        cgltf::CameraType::Perspective => {
            let p = &camera.perspective;
            append(json, "\"type\":\"perspective\",\"perspective\":{");
            append(json, "\"yfov\":");
            append_float(json, p.yfov);
            append(json, ",\"znear\":");
            append_float(json, p.znear);
            if p.aspect_ratio != 0.0 {
                append(json, ",\"aspectRatio\":");
                append_float(json, p.aspect_ratio);
            }
            if p.zfar != 0.0 {
                append(json, ",\"zfar\":");
                append_float(json, p.zfar);
            }
            append(json, "}");
        }
        cgltf::CameraType::Orthographic => {
            let o = &camera.orthographic;
            append(json, "\"type\":\"orthographic\",\"orthographic\":{");
            append(json, "\"xmag\":");
            append_float(json, o.xmag);
            append(json, ",\"ymag\":");
            append_float(json, o.ymag);
            append(json, ",\"znear\":");
            append_float(json, o.znear);
            append(json, ",\"zfar\":");
            append_float(json, o.zfar);
            append(json, "}");
        }
        cgltf::CameraType::Invalid => {
            eprintln!("Warning: skipping camera of unknown type");
        }
    }

    append(json, "}");
}

pub fn write_light(json: &mut String, light: &cgltf::Light) {
    comma(json);
    append(json, "{\"type\":\"");
    append(json, light_type(light.type_));
    append(json, "\"");
    floats_field(json, "color", &light.color, &WHITE);
    float_field(json, "intensity", light.intensity, 1.0);
    float_field(json, "range", light.range, 0.0);
    if light.type_ == LightType::Spot {
        comma(json);
        append(json, "\"spot\":{");
        append(json, "\"innerConeAngle\":");
        append_float(json, light.spot_inner_cone_angle);
        append(json, ",\"outerConeAngle\":");
        append_float(
            json,
            if light.spot_outer_cone_angle == 0.0 {
                // C++ literal 0.78539816339f rounds to the same float
                std::f32::consts::FRAC_PI_4
            } else {
                light.spot_outer_cone_angle
            },
        );
        append(json, "}");
    }
    append(json, "}");
}

pub fn write_array(json: &mut String, name: &str, contents: &str) {
    if contents.is_empty() {
        return;
    }

    comma(json);
    append(json, "\"");
    append(json, name);
    append(json, "\":[");
    append(json, contents);
    append(json, "]");
}

pub fn write_extensions(json: &mut String, extensions: &[ExtensionInfo]) {
    let used_extensions = extensions.iter().any(|e| e.used);
    let required_extensions = extensions.iter().any(|e| e.used && e.required);

    if used_extensions {
        comma(json);
        append(json, "\"extensionsUsed\":[");
        for e in extensions.iter().filter(|e| e.used) {
            comma(json);
            append(json, "\"");
            append(json, e.name);
            append(json, "\"");
        }
        append(json, "]");
    }

    if required_extensions {
        comma(json);
        append(json, "\"extensionsRequired\":[");
        for e in extensions.iter().filter(|e| e.used && e.required) {
            comma(json);
            append(json, "\"");
            append(json, e.name);
            append(json, "\"");
        }
        append(json, "]");
    }
}

pub fn write_extras(json: &mut String, extras: &cgltf::Extras) {
    let Some(data) = &extras.data else {
        return;
    };

    comma(json);
    append(json, "\"extras\":");
    append_json(json, data);
}

pub fn write_scene(json: &mut String, scene: &cgltf::Scene, roots: &str, settings: &Settings) {
    comma(json);
    append(json, "{");
    if let Some(name) = scene.name.as_deref().filter(|n| !n.is_empty()) {
        append(json, "\"name\":\"");
        append(json, name);
        append(json, "\"");
    }
    if !roots.is_empty() {
        comma(json);
        append(json, "\"nodes\":[");
        append(json, roots);
        append(json, "]");
    }
    if settings.keep_extras {
        write_extras(json, &scene.extras);
    }
    append(json, "}");
}
