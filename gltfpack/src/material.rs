//! Material analysis and merging (`material.cpp`).

use crate::cgltf::{self, AlphaMode, Data, Material, TextureView};
use crate::image::{has_alpha, read_image};
use crate::parsegltf::are_extras_equal;
use crate::types::*;

/// `memcmp(lhs, rhs, n * sizeof(float)) == 0`
pub(crate) fn bits_eq(lhs: &[f32], rhs: &[f32]) -> bool {
    lhs.len() == rhs.len() && lhs.iter().zip(rhs).all(|(a, b)| a.to_bits() == b.to_bits())
}

fn are_textures_equal(lhs: &cgltf::Texture, rhs: &cgltf::Texture) -> bool {
    lhs.image == rhs.image && lhs.sampler == rhs.sampler && lhs.basisu_image == rhs.basisu_image && lhs.webp_image == rhs.webp_image
}

fn are_texture_views_equal(data: &Data, lhs: &TextureView, rhs: &TextureView) -> bool {
    if lhs.has_transform != rhs.has_transform {
        return false;
    }

    if lhs.has_transform {
        let lt = &lhs.transform;
        let rt = &rhs.transform;

        if !bits_eq(&lt.offset, &rt.offset) {
            return false;
        }

        if lt.rotation != rt.rotation {
            return false;
        }

        if !bits_eq(&lt.scale, &rt.scale) {
            return false;
        }

        if lt.texcoord != rt.texcoord {
            return false;
        }
    }

    if lhs.texture != rhs.texture {
        match (lhs.texture, rhs.texture) {
            (Some(l), Some(r)) if are_textures_equal(&data.textures[l], &data.textures[r]) => {}
            _ => return false,
        }
    }

    lhs.texcoord == rhs.texcoord && lhs.scale == rhs.scale
}

fn are_materials_equal(data: &Data, lhs: &Material, rhs: &Material, settings: &Settings) -> bool {
    let tv = |l: &TextureView, r: &TextureView| are_texture_views_equal(data, l, r);

    if lhs.has_pbr_metallic_roughness != rhs.has_pbr_metallic_roughness {
        return false;
    }

    if lhs.has_pbr_metallic_roughness {
        let (l, r) = (&lhs.pbr_metallic_roughness, &rhs.pbr_metallic_roughness);
        if !(tv(&l.base_color_texture, &r.base_color_texture)
            && tv(&l.metallic_roughness_texture, &r.metallic_roughness_texture)
            && bits_eq(&l.base_color_factor, &r.base_color_factor)
            && l.metallic_factor == r.metallic_factor
            && l.roughness_factor == r.roughness_factor)
        {
            return false;
        }
    }

    if lhs.has_pbr_specular_glossiness != rhs.has_pbr_specular_glossiness {
        return false;
    }

    if lhs.has_pbr_specular_glossiness {
        let (l, r) = (&lhs.pbr_specular_glossiness, &rhs.pbr_specular_glossiness);
        if !(tv(&l.diffuse_texture, &r.diffuse_texture)
            && tv(&l.specular_glossiness_texture, &r.specular_glossiness_texture)
            && bits_eq(&l.diffuse_factor, &r.diffuse_factor)
            && bits_eq(&l.specular_factor, &r.specular_factor)
            && l.glossiness_factor == r.glossiness_factor)
        {
            return false;
        }
    }

    if lhs.has_clearcoat != rhs.has_clearcoat {
        return false;
    }

    if lhs.has_clearcoat {
        let (l, r) = (&lhs.clearcoat, &rhs.clearcoat);
        if !(tv(&l.clearcoat_texture, &r.clearcoat_texture)
            && tv(&l.clearcoat_roughness_texture, &r.clearcoat_roughness_texture)
            && tv(&l.clearcoat_normal_texture, &r.clearcoat_normal_texture)
            && l.clearcoat_factor == r.clearcoat_factor
            && l.clearcoat_roughness_factor == r.clearcoat_roughness_factor)
        {
            return false;
        }
    }

    if lhs.has_transmission != rhs.has_transmission {
        return false;
    }

    if lhs.has_transmission {
        let (l, r) = (&lhs.transmission, &rhs.transmission);
        if !(tv(&l.transmission_texture, &r.transmission_texture) && l.transmission_factor == r.transmission_factor) {
            return false;
        }
    }

    if lhs.has_ior != rhs.has_ior {
        return false;
    }

    if lhs.has_ior && lhs.ior.ior != rhs.ior.ior {
        return false;
    }

    if lhs.has_specular != rhs.has_specular {
        return false;
    }

    if lhs.has_specular {
        let (l, r) = (&lhs.specular, &rhs.specular);
        if !(tv(&l.specular_texture, &r.specular_texture)
            && tv(&l.specular_color_texture, &r.specular_color_texture)
            && l.specular_factor == r.specular_factor
            && bits_eq(&l.specular_color_factor, &r.specular_color_factor))
        {
            return false;
        }
    }

    if lhs.has_sheen != rhs.has_sheen {
        return false;
    }

    if lhs.has_sheen {
        let (l, r) = (&lhs.sheen, &rhs.sheen);
        if !(tv(&l.sheen_color_texture, &r.sheen_color_texture)
            && bits_eq(&l.sheen_color_factor, &r.sheen_color_factor)
            && tv(&l.sheen_roughness_texture, &r.sheen_roughness_texture)
            && l.sheen_roughness_factor == r.sheen_roughness_factor)
        {
            return false;
        }
    }

    if lhs.has_volume != rhs.has_volume {
        return false;
    }

    if lhs.has_volume {
        let (l, r) = (&lhs.volume, &rhs.volume);
        if !(tv(&l.thickness_texture, &r.thickness_texture)
            && l.thickness_factor == r.thickness_factor
            && bits_eq(&l.attenuation_color, &r.attenuation_color)
            && l.attenuation_distance == r.attenuation_distance)
        {
            return false;
        }
    }

    if lhs.has_emissive_strength != rhs.has_emissive_strength {
        return false;
    }

    if lhs.has_emissive_strength && lhs.emissive_strength.emissive_strength != rhs.emissive_strength.emissive_strength {
        return false;
    }

    if lhs.has_iridescence != rhs.has_iridescence {
        return false;
    }

    if lhs.has_iridescence {
        let (l, r) = (&lhs.iridescence, &rhs.iridescence);
        if !(l.iridescence_factor == r.iridescence_factor
            && tv(&l.iridescence_texture, &r.iridescence_texture)
            && l.iridescence_ior == r.iridescence_ior
            && l.iridescence_thickness_min == r.iridescence_thickness_min
            && l.iridescence_thickness_max == r.iridescence_thickness_max
            && tv(&l.iridescence_thickness_texture, &r.iridescence_thickness_texture))
        {
            return false;
        }
    }

    if lhs.has_anisotropy != rhs.has_anisotropy {
        return false;
    }

    if lhs.has_anisotropy {
        let (l, r) = (&lhs.anisotropy, &rhs.anisotropy);
        if !(l.anisotropy_strength == r.anisotropy_strength
            && l.anisotropy_rotation == r.anisotropy_rotation
            && tv(&l.anisotropy_texture, &r.anisotropy_texture))
        {
            return false;
        }
    }

    if lhs.has_dispersion != rhs.has_dispersion {
        return false;
    }

    if lhs.has_dispersion && lhs.dispersion.dispersion != rhs.dispersion.dispersion {
        return false;
    }

    if lhs.has_diffuse_transmission != rhs.has_diffuse_transmission {
        return false;
    }

    if lhs.has_diffuse_transmission {
        let (l, r) = (&lhs.diffuse_transmission, &rhs.diffuse_transmission);
        if !(l.diffuse_transmission_factor == r.diffuse_transmission_factor
            && bits_eq(&l.diffuse_transmission_color_factor, &r.diffuse_transmission_color_factor)
            && tv(&l.diffuse_transmission_texture, &r.diffuse_transmission_texture)
            && tv(&l.diffuse_transmission_color_texture, &r.diffuse_transmission_color_texture))
        {
            return false;
        }
    }

    if !tv(&lhs.normal_texture, &rhs.normal_texture) {
        return false;
    }

    if !tv(&lhs.occlusion_texture, &rhs.occlusion_texture) {
        return false;
    }

    if !tv(&lhs.emissive_texture, &rhs.emissive_texture) {
        return false;
    }

    if !bits_eq(&lhs.emissive_factor, &rhs.emissive_factor) {
        return false;
    }

    if lhs.alpha_mode != rhs.alpha_mode || lhs.alpha_cutoff != rhs.alpha_cutoff {
        return false;
    }

    if lhs.double_sided != rhs.double_sided || lhs.unlit != rhs.unlit {
        return false;
    }

    if settings.keep_extras && !are_extras_equal(&lhs.extras, &rhs.extras) {
        return false;
    }

    true
}

fn has_name(name: &Option<String>) -> bool {
    name.as_deref().is_some_and(|n| !n.is_empty())
}

pub fn merge_mesh_materials(data: &Data, meshes: &mut [Mesh], settings: &Settings) {
    let mut material_remap: Vec<usize> = (0..data.materials.len()).collect();

    for i in 0..data.materials.len() {
        if settings.keep_materials && has_name(&data.materials[i].name) {
            continue;
        }

        for j in 0..i {
            if settings.keep_materials && has_name(&data.materials[j].name) {
                continue;
            }

            if are_materials_equal(data, &data.materials[i], &data.materials[j], settings) {
                material_remap[i] = j;
                break;
            }
        }
    }

    for mesh in meshes.iter_mut() {
        if let Some(m) = mesh.material {
            mesh.material = Some(material_remap[m]);
        }

        for variant in mesh.variants.iter_mut() {
            variant.material = material_remap[variant.material];
        }
    }
}

pub fn mark_needed_materials(data: &Data, materials: &mut [MaterialInfo], meshes: &[Mesh], settings: &Settings) {
    // mark all used materials as kept
    for mesh in meshes {
        if let Some(m) = mesh.material {
            materials[m].keep = true;
        }

        for variant in &mesh.variants {
            materials[variant.material].keep = true;
        }
    }

    // mark all named materials as kept if requested
    if settings.keep_materials {
        for (i, material) in data.materials.iter().enumerate() {
            if has_name(&material.name) {
                materials[i].keep = true;
            }
        }
    }
}

pub fn is_valid_transform(transform: &cgltf::TextureTransform) -> bool {
    transform.offset[0] != 0.0
        || transform.offset[1] != 0.0
        || transform.scale[0] != 1.0
        || transform.scale[1] != 1.0
        || transform.rotation != 0.0
}

fn get_texture_image(data: &Data, texture: Option<usize>) -> Option<usize> {
    let texture = &data.textures[texture?];
    texture.image.or(texture.basisu_image).or(texture.webp_image)
}

fn analyze_material_texture(
    view: &TextureView,
    kind: TextureKind,
    mi: &mut MaterialInfo,
    data: &Data,
    textures: &mut [TextureInfo],
    images: &mut [ImageInfo],
) {
    mi.uses_texture_transform |= view.has_transform && is_valid_transform(&view.transform);

    if let Some(t) = view.texture {
        textures[t].keep = true;

        mi.texture_set_mask |= 1u32.wrapping_shl(view.texcoord as u32);
        mi.needs_tangents |= kind == TextureKind::Normal;
    }

    if let Some(image) = get_texture_image(data, view.texture) {
        let info = &mut images[image];

        if info.kind == TextureKind::Generic || info.kind > kind {
            // this is useful to keep color textures that have attrib data in alpha tagged as color
            info.kind = kind;
        }

        info.normal_map |= kind == TextureKind::Normal;
        info.srgb |= kind == TextureKind::Color;
    }
}

fn analyze_material(material: &Material, mi: &mut MaterialInfo, data: &Data, textures: &mut [TextureInfo], images: &mut [ImageInfo]) {
    let mut a = |view: &TextureView, kind: TextureKind| analyze_material_texture(view, kind, mi, data, textures, images);

    if material.has_pbr_metallic_roughness {
        a(&material.pbr_metallic_roughness.base_color_texture, TextureKind::Color);
        a(&material.pbr_metallic_roughness.metallic_roughness_texture, TextureKind::Attrib);
    }

    if material.has_pbr_specular_glossiness {
        a(&material.pbr_specular_glossiness.diffuse_texture, TextureKind::Color);
        a(&material.pbr_specular_glossiness.specular_glossiness_texture, TextureKind::Attrib);
    }

    if material.has_clearcoat {
        a(&material.clearcoat.clearcoat_texture, TextureKind::Attrib);
        a(&material.clearcoat.clearcoat_roughness_texture, TextureKind::Attrib);
        a(&material.clearcoat.clearcoat_normal_texture, TextureKind::Normal);
    }

    if material.has_transmission {
        a(&material.transmission.transmission_texture, TextureKind::Attrib);
    }

    if material.has_specular {
        a(&material.specular.specular_texture, TextureKind::Attrib);
        a(&material.specular.specular_color_texture, TextureKind::Color);
    }

    if material.has_sheen {
        a(&material.sheen.sheen_color_texture, TextureKind::Color);
        a(&material.sheen.sheen_roughness_texture, TextureKind::Attrib);
    }

    if material.has_volume {
        a(&material.volume.thickness_texture, TextureKind::Attrib);
    }

    if material.has_iridescence {
        a(&material.iridescence.iridescence_texture, TextureKind::Attrib);
        a(&material.iridescence.iridescence_thickness_texture, TextureKind::Attrib);
    }

    if material.has_anisotropy {
        a(&material.anisotropy.anisotropy_texture, TextureKind::Normal);
    }

    if material.has_diffuse_transmission {
        a(&material.diffuse_transmission.diffuse_transmission_texture, TextureKind::Attrib);
        a(
            &material.diffuse_transmission.diffuse_transmission_color_texture,
            TextureKind::Color,
        );
    }

    a(&material.normal_texture, TextureKind::Normal);
    a(&material.occlusion_texture, TextureKind::Attrib);
    a(&material.emissive_texture, TextureKind::Color);

    if material.unlit {
        mi.unlit = true;
    }
}

pub fn analyze_materials(data: &Data, materials: &mut [MaterialInfo], textures: &mut [TextureInfo], images: &mut [ImageInfo]) {
    for (i, material) in data.materials.iter().enumerate() {
        analyze_material(material, &mut materials[i], data, textures, images);
    }
}

fn get_channels(data: &Data, image: usize, info: &mut ImageInfo, input_path: Option<&str>) -> i32 {
    if info.channels != 0 {
        return info.channels;
    }

    info.channels = match read_image(data, &data.images[image], input_path) {
        Some((img_data, mime_type)) => {
            if has_alpha(&img_data, &mime_type) {
                4
            } else {
                3
            }
        }
        None => -1,
    };

    info.channels
}

fn should_keep_alpha(color: &TextureView, alpha: f32, data: &Data, input_path: Option<&str>, images: &mut [ImageInfo]) -> bool {
    if alpha != 1.0 {
        return true;
    }

    match get_texture_image(data, color.texture) {
        Some(image) => get_channels(data, image, &mut images[image], input_path) == 4,
        None => false,
    }
}

pub fn optimize_materials(data: &mut Data, materials: &[MaterialInfo], images: &mut [ImageInfo], input_path: Option<&str>) {
    for i in 0..data.materials.len() {
        // remove BLEND/MASK from materials that don't have alpha information
        let material = &data.materials[i];

        if material.alpha_mode != AlphaMode::Opaque && !materials[i].mesh_alpha {
            if material.has_pbr_metallic_roughness
                && should_keep_alpha(
                    &material.pbr_metallic_roughness.base_color_texture,
                    material.pbr_metallic_roughness.base_color_factor[3],
                    data,
                    input_path,
                    images,
                )
            {
                continue;
            }

            if material.has_pbr_specular_glossiness
                && should_keep_alpha(
                    &material.pbr_specular_glossiness.diffuse_texture,
                    material.pbr_specular_glossiness.diffuse_factor[3],
                    data,
                    input_path,
                    images,
                )
            {
                continue;
            }

            let material = &mut data.materials[i];
            material.alpha_mode = AlphaMode::Opaque;
            material.alpha_cutoff = 0.5; // reset to default to avoid writing it to output
        }
    }
}

pub fn merge_textures(data: &Data, textures: &mut [TextureInfo]) {
    let mut offset = 0;

    for i in 0..textures.len() {
        if !textures[i].keep {
            continue;
        }

        for j in 0..i {
            if textures[j].keep && are_textures_equal(&data.textures[i], &data.textures[j]) {
                textures[i].keep = false;
                textures[i].remap = textures[j].remap;
                break;
            }
        }

        if textures[i].keep {
            textures[i].remap = offset;
            offset += 1;
        }
    }
}
