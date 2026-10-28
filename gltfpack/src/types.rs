//! gltfpack data structures (`gltfpack.h`). References into `cgltf::Data` are indices.

use crate::cgltf;

/// `struct Attr { float f[4]; }`
pub type Attr = [f32; 4];

#[derive(Debug, Clone, Default)]
pub struct Stream {
    pub type_: cgltf::AttributeType,
    pub index: i32,
    /// 0 = base mesh, 1+ = morph target
    pub target: i32,
    /// 1+ = splat attribute type
    pub splats: i32,

    /// only valid for `AttributeType::Custom`
    pub custom_name: Option<String>,

    pub data: Vec<Attr>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Instance {
    pub transform: [f32; 16],
    pub color: [f32; 4],
}

#[derive(Debug, Clone, Default)]
pub struct Mesh {
    pub scene: i32,
    /// indices into `data.nodes`
    pub nodes: Vec<usize>,
    pub instances: Vec<Instance>,

    pub material: Option<usize>,
    pub skin: Option<usize>,

    pub extras: cgltf::Extras,

    pub type_: cgltf::PrimitiveType,
    pub splats: bool,

    pub streams: Vec<Stream>,
    pub indices: Vec<u32>,

    pub geometry_duplicate: bool,
    pub geometry_hash: [u64; 2],

    pub targets: usize,
    pub target_weights: Vec<f32>,
    pub target_names: Vec<String>,

    pub variants: Vec<cgltf::MaterialMapping>,

    pub quality: f32,
}

#[derive(Debug, Clone, Default)]
pub struct Track {
    /// index into `data.nodes`
    pub node: usize,
    pub path: cgltf::AnimationPathType,

    pub constant: bool,
    pub dummy: bool,

    /// 1 unless path is `AnimationPathType::Weights`
    pub components: usize,

    pub interpolation: cgltf::InterpolationType,

    /// empty for resampled or constant animations
    pub time: Vec<f32>,
    pub data: Vec<Attr>,
}

#[derive(Debug, Clone, Default)]
pub struct Animation {
    pub name: Option<String>,

    pub start: f32,
    pub frames: i32,

    pub tracks: Vec<Track>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum TextureKind {
    #[default]
    Generic,
    Color,
    Normal,
    Attrib,
}

pub const TEXTURE_KIND_COUNT: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextureMode {
    #[default]
    Raw,
    Etc1s,
    Uastc,
    WebP,
}

#[derive(Debug, Clone, Default)]
pub struct Settings {
    pub pos_bits: i32,
    pub tex_bits: i32,
    pub nrm_bits: i32,
    pub col_bits: i32,

    pub pos_normalized: bool,
    pub pos_float: bool,
    pub tex_float: bool,
    pub nrm_float: bool,

    pub trn_bits: i32,
    pub rot_bits: i32,
    pub scl_bits: i32,

    pub anim_freq: i32,
    pub anim_const: bool,

    pub keep_nodes: bool,
    pub keep_materials: bool,
    pub keep_extras: bool,
    pub keep_attributes: bool,

    pub mesh_dedup: bool,
    pub mesh_merge: bool,
    pub mesh_instancing: bool,
    pub mesh_interleaved: bool,
    pub mesh_tangents: bool,
    pub mesh_normals: bool,

    pub normals_crease: f32,

    pub simplify_ratio: f32,
    pub simplify_error: f32,
    pub simplify_aggressive: bool,
    pub simplify_lock_borders: bool,
    pub simplify_permissive: bool,
    pub simplify_update: bool,

    pub texture_ktx2: bool,
    pub texture_webp: bool,
    pub texture_embed: bool,
    pub texture_ref: bool,

    pub texture_pow2: bool,
    pub texture_flipy: bool,
    pub texture_scale: [f32; TEXTURE_KIND_COUNT],
    pub texture_limit: [i32; TEXTURE_KIND_COUNT],

    pub texture_mode: [TextureMode; TEXTURE_KIND_COUNT],
    pub texture_quality: [i32; TEXTURE_KIND_COUNT],

    pub texture_jobs: i32,

    pub splats_dim: i32,

    pub quantize: bool,

    pub compress: bool,
    pub compressmore: bool,
    pub compresskhr: bool,
    pub fallback: bool,

    pub verbose: i32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct QuantizationPosition {
    pub offset: [f32; 3],
    pub scale: f32,
    pub bits: i32,
    pub normalized: bool,

    /// computed from scale/bits/normalized
    pub node_scale: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct QuantizationTexture {
    pub offset: [f32; 2],
    pub scale: [f32; 2],
    pub bits: i32,
    pub normalized: bool,
}

/// `StreamFormat::Filter`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Filter {
    #[default]
    None,
    Oct,
    Quat,
    Exp,
    Color,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StreamFormat {
    pub type_: cgltf::Type,
    pub component_type: cgltf::ComponentType,
    pub normalized: bool,
    pub stride: usize,
    pub filter: Filter,
}

impl StreamFormat {
    pub fn new(type_: cgltf::Type, component_type: cgltf::ComponentType, normalized: bool, stride: usize, filter: Filter) -> Self {
        Self {
            type_,
            component_type,
            normalized,
            stride,
            filter,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct NodeInfo {
    pub scene: i32,

    pub keep: bool,
    pub animated: bool,

    pub animated_path_mask: u32,

    pub remap: i32,

    pub mesh_nodes: Vec<usize>,

    pub has_mesh: bool,
    pub mesh_index: usize,
    pub mesh_skin: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MaterialInfo {
    pub keep: bool,

    pub uses_texture_transform: bool,
    pub needs_tangents: bool,
    pub unlit: bool,
    pub mesh_alpha: bool,

    pub texture_set_mask: u32,

    pub remap: i32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TextureInfo {
    pub keep: bool,

    pub remap: i32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ImageInfo {
    pub kind: TextureKind,
    pub normal_map: bool,
    pub srgb: bool,

    pub channels: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct ExtensionInfo<'a> {
    pub name: &'a str,

    pub used: bool,
    pub required: bool,
}

/// `BufferView::Kind`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BufferViewKind {
    #[default]
    Vertex,
    Index,
    Skin,
    Time,
    Keyframe,
    Instance,
    Image,
    Splat,
}

pub const BUFFER_VIEW_KIND_COUNT: usize = 8;

/// `BufferView::Compression`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compression {
    #[default]
    None,
    Attribute,
    Index,
    IndexSequence,
}

#[derive(Debug, Clone, Default)]
pub struct BufferView {
    pub kind: BufferViewKind,
    pub filter: Filter,
    pub compression: Compression,
    pub stride: usize,
    pub variant: i32,

    pub data: Vec<u8>,

    pub bytes: usize,
}
