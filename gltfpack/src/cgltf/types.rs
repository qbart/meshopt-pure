//! cgltf data model (`cgltf_*` structs and enums). Object references are indices into the owning [`Data`] vectors.
//!
//! While parsing, reference fields hold the raw `CGLTF_PTRINDEX` value (JSON index + 1, 0 = none); `fixup_pointers`
//! range-checks them and converts them to indices.

pub type CgltfSize = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FileType {
    #[default]
    Invalid,
    Gltf,
    Glb,
}

/// `cgltf_result`; `Success` is represented by `Ok(..)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CgltfResult {
    Success,
    DataTooShort,
    UnknownFormat,
    InvalidJson,
    InvalidGltf,
    InvalidOptions,
    FileNotFound,
    IoError,
    OutOfMemory,
    LegacyGltf,
}

/// `cgltf_options` (memory/file callbacks are not ported).
#[derive(Debug, Clone, Copy, Default)]
pub struct Options {
    /// invalid == auto detect
    pub type_: FileType,
    /// 0 == auto
    pub json_token_count: CgltfSize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BufferViewType {
    #[default]
    Invalid,
    Indices,
    Vertices,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, PartialOrd, Ord, Hash)]
pub enum AttributeType {
    #[default]
    Invalid,
    Position,
    Normal,
    Tangent,
    Texcoord,
    Color,
    Joints,
    Weights,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComponentType {
    #[default]
    Invalid,
    /// BYTE
    R8,
    /// UNSIGNED_BYTE
    R8u,
    /// SHORT
    R16,
    /// UNSIGNED_SHORT
    R16u,
    /// UNSIGNED_INT
    R32u,
    /// FLOAT
    R32f,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Type {
    #[default]
    Invalid,
    Scalar,
    Vec2,
    Vec3,
    Vec4,
    Mat2,
    Mat3,
    Mat4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrimitiveType {
    #[default]
    Invalid,
    Points,
    Lines,
    LineLoop,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlphaMode {
    #[default]
    Opaque,
    Mask,
    Blend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnimationPathType {
    #[default]
    Invalid,
    Translation,
    Rotation,
    Scale,
    Weights,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpolationType {
    #[default]
    Linear,
    Step,
    CubicSpline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CameraType {
    #[default]
    Invalid,
    Perspective,
    Orthographic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LightType {
    #[default]
    Invalid,
    Directional,
    Point,
    Spot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataFreeMethod {
    #[default]
    None,
    FileRelease,
    MemoryFree,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Extras {
    /// deprecated in cgltf; offsets of the extras value in the JSON chunk
    pub start_offset: CgltfSize,
    pub end_offset: CgltfSize,

    pub data: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Extension {
    pub name: Option<String>,
    pub data: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Buffer {
    pub name: Option<String>,
    pub size: CgltfSize,
    pub uri: Option<String>,
    /// loaded by `load_buffers`
    pub data: Option<Vec<u8>>,
    pub data_free_method: DataFreeMethod,
    pub extras: Extras,
    pub extensions: Vec<Extension>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeshoptCompressionMode {
    #[default]
    Invalid,
    Attributes,
    Triangles,
    Indices,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MeshoptCompressionFilter {
    #[default]
    None,
    Octahedral,
    Quaternion,
    Exponential,
    Color,
}

#[derive(Debug, Clone, Default)]
pub struct MeshoptCompression {
    /// required reference (C: `CGLTF_PTRFIXUP_REQ`)
    pub buffer: usize,
    pub offset: CgltfSize,
    pub size: CgltfSize,
    pub stride: CgltfSize,
    pub count: CgltfSize,
    pub mode: MeshoptCompressionMode,
    pub filter: MeshoptCompressionFilter,
    pub is_khr: bool,
}

#[derive(Debug, Clone, Default)]
pub struct BufferView {
    pub name: Option<String>,
    /// required reference (C: `CGLTF_PTRFIXUP_REQ`)
    pub buffer: usize,
    pub offset: CgltfSize,
    pub size: CgltfSize,
    /// 0 == automatically determined by accessor
    pub stride: CgltfSize,
    pub type_: BufferViewType,
    /// overrides buffer->data if present, filled by extensions
    pub data: Option<Vec<u8>>,
    pub has_meshopt_compression: bool,
    pub meshopt_compression: MeshoptCompression,
    pub extras: Extras,
    pub extensions: Vec<Extension>,
}

#[derive(Debug, Clone, Default)]
pub struct AccessorSparse {
    pub count: CgltfSize,
    /// required reference when the accessor is sparse
    pub indices_buffer_view: usize,
    pub indices_byte_offset: CgltfSize,
    pub indices_component_type: ComponentType,
    /// required reference when the accessor is sparse
    pub values_buffer_view: usize,
    pub values_byte_offset: CgltfSize,
}

#[derive(Debug, Clone, Default)]
pub struct Accessor {
    pub name: Option<String>,
    pub component_type: ComponentType,
    pub normalized: bool,
    pub type_: Type,
    pub offset: CgltfSize,
    pub count: CgltfSize,
    pub stride: CgltfSize,
    pub buffer_view: Option<usize>,
    pub has_min: bool,
    pub min: [f32; 16],
    pub has_max: bool,
    pub max: [f32; 16],
    pub is_sparse: bool,
    pub sparse: AccessorSparse,
    pub extras: Extras,
    pub extensions: Vec<Extension>,
}

#[derive(Debug, Clone, Default)]
pub struct Attribute {
    pub name: Option<String>,
    pub type_: AttributeType,
    pub index: i32,
    /// required reference (C: `CGLTF_PTRFIXUP_REQ`)
    pub data: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Image {
    pub name: Option<String>,
    pub uri: Option<String>,
    pub buffer_view: Option<usize>,
    pub mime_type: Option<String>,
    pub extras: Extras,
    pub extensions: Vec<Extension>,
}

pub type FilterType = i32;
pub const FILTER_TYPE_UNDEFINED: FilterType = 0;
pub const FILTER_TYPE_NEAREST: FilterType = 9728;
pub const FILTER_TYPE_LINEAR: FilterType = 9729;
pub const FILTER_TYPE_NEAREST_MIPMAP_NEAREST: FilterType = 9984;
pub const FILTER_TYPE_LINEAR_MIPMAP_NEAREST: FilterType = 9985;
pub const FILTER_TYPE_NEAREST_MIPMAP_LINEAR: FilterType = 9986;
pub const FILTER_TYPE_LINEAR_MIPMAP_LINEAR: FilterType = 9987;

pub type WrapMode = i32;
pub const WRAP_MODE_CLAMP_TO_EDGE: WrapMode = 33071;
pub const WRAP_MODE_MIRRORED_REPEAT: WrapMode = 33648;
pub const WRAP_MODE_REPEAT: WrapMode = 10497;

#[derive(Debug, Clone, Default)]
pub struct Sampler {
    pub name: Option<String>,
    pub mag_filter: FilterType,
    pub min_filter: FilterType,
    pub wrap_s: WrapMode,
    pub wrap_t: WrapMode,
    pub extras: Extras,
    pub extensions: Vec<Extension>,
}

#[derive(Debug, Clone, Default)]
pub struct Texture {
    pub name: Option<String>,
    pub image: Option<usize>,
    pub sampler: Option<usize>,
    pub has_basisu: bool,
    pub basisu_image: Option<usize>,
    pub has_webp: bool,
    pub webp_image: Option<usize>,
    pub extras: Extras,
    pub extensions: Vec<Extension>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TextureTransform {
    pub offset: [f32; 2],
    pub rotation: f32,
    pub scale: [f32; 2],
    pub has_texcoord: bool,
    pub texcoord: i32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TextureView {
    pub texture: Option<usize>,
    pub texcoord: i32,
    /// equivalent to strength for occlusion_texture
    pub scale: f32,
    pub has_transform: bool,
    pub transform: TextureTransform,
}

#[derive(Debug, Clone, Default)]
pub struct PbrMetallicRoughness {
    pub base_color_texture: TextureView,
    pub metallic_roughness_texture: TextureView,

    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
}

#[derive(Debug, Clone, Default)]
pub struct PbrSpecularGlossiness {
    pub diffuse_texture: TextureView,
    pub specular_glossiness_texture: TextureView,

    pub diffuse_factor: [f32; 4],
    pub specular_factor: [f32; 3],
    pub glossiness_factor: f32,
}

#[derive(Debug, Clone, Default)]
pub struct Clearcoat {
    pub clearcoat_texture: TextureView,
    pub clearcoat_roughness_texture: TextureView,
    pub clearcoat_normal_texture: TextureView,

    pub clearcoat_factor: f32,
    pub clearcoat_roughness_factor: f32,
}

#[derive(Debug, Clone, Default)]
pub struct Transmission {
    pub transmission_texture: TextureView,
    pub transmission_factor: f32,
}

#[derive(Debug, Clone, Default)]
pub struct Ior {
    pub ior: f32,
}

#[derive(Debug, Clone, Default)]
pub struct Specular {
    pub specular_texture: TextureView,
    pub specular_color_texture: TextureView,
    pub specular_color_factor: [f32; 3],
    pub specular_factor: f32,
}

#[derive(Debug, Clone, Default)]
pub struct Volume {
    pub thickness_texture: TextureView,
    pub thickness_factor: f32,
    pub attenuation_color: [f32; 3],
    pub attenuation_distance: f32,
}

#[derive(Debug, Clone, Default)]
pub struct Sheen {
    pub sheen_color_texture: TextureView,
    pub sheen_color_factor: [f32; 3],
    pub sheen_roughness_texture: TextureView,
    pub sheen_roughness_factor: f32,
}

#[derive(Debug, Clone, Default)]
pub struct EmissiveStrength {
    pub emissive_strength: f32,
}

#[derive(Debug, Clone, Default)]
pub struct Iridescence {
    pub iridescence_factor: f32,
    pub iridescence_texture: TextureView,
    pub iridescence_ior: f32,
    pub iridescence_thickness_min: f32,
    pub iridescence_thickness_max: f32,
    pub iridescence_thickness_texture: TextureView,
}

#[derive(Debug, Clone, Default)]
pub struct DiffuseTransmission {
    pub diffuse_transmission_texture: TextureView,
    pub diffuse_transmission_factor: f32,
    pub diffuse_transmission_color_factor: [f32; 3],
    pub diffuse_transmission_color_texture: TextureView,
}

#[derive(Debug, Clone, Default)]
pub struct Anisotropy {
    pub anisotropy_strength: f32,
    pub anisotropy_rotation: f32,
    pub anisotropy_texture: TextureView,
}

#[derive(Debug, Clone, Default)]
pub struct Dispersion {
    pub dispersion: f32,
}

#[derive(Debug, Clone, Default)]
pub struct Material {
    pub name: Option<String>,
    pub has_pbr_metallic_roughness: bool,
    pub has_pbr_specular_glossiness: bool,
    pub has_clearcoat: bool,
    pub has_transmission: bool,
    pub has_volume: bool,
    pub has_ior: bool,
    pub has_specular: bool,
    pub has_sheen: bool,
    pub has_emissive_strength: bool,
    pub has_iridescence: bool,
    pub has_diffuse_transmission: bool,
    pub has_anisotropy: bool,
    pub has_dispersion: bool,
    pub pbr_metallic_roughness: PbrMetallicRoughness,
    pub pbr_specular_glossiness: PbrSpecularGlossiness,
    pub clearcoat: Clearcoat,
    pub ior: Ior,
    pub specular: Specular,
    pub sheen: Sheen,
    pub transmission: Transmission,
    pub volume: Volume,
    pub emissive_strength: EmissiveStrength,
    pub iridescence: Iridescence,
    pub diffuse_transmission: DiffuseTransmission,
    pub anisotropy: Anisotropy,
    pub dispersion: Dispersion,
    pub normal_texture: TextureView,
    pub occlusion_texture: TextureView,
    pub emissive_texture: TextureView,
    pub emissive_factor: [f32; 3],
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    pub unlit: bool,
    pub extras: Extras,
    pub extensions: Vec<Extension>,
}

#[derive(Debug, Clone, Default)]
pub struct MaterialMapping {
    pub variant: CgltfSize,
    /// required reference (C: `CGLTF_PTRFIXUP_REQ`)
    pub material: usize,
    pub extras: Extras,
}

#[derive(Debug, Clone, Default)]
pub struct MorphTarget {
    pub attributes: Vec<Attribute>,
}

#[derive(Debug, Clone, Default)]
pub struct DracoMeshCompression {
    /// required reference (C: `CGLTF_PTRFIXUP_REQ`)
    pub buffer_view: usize,
    pub attributes: Vec<Attribute>,
}

#[derive(Debug, Clone, Default)]
pub struct MeshGpuInstancing {
    pub attributes: Vec<Attribute>,
}

#[derive(Debug, Clone, Default)]
pub struct Primitive {
    pub type_: PrimitiveType,
    pub indices: Option<usize>,
    pub material: Option<usize>,
    pub attributes: Vec<Attribute>,
    pub targets: Vec<MorphTarget>,
    pub extras: Extras,
    pub has_draco_mesh_compression: bool,
    pub draco_mesh_compression: DracoMeshCompression,
    pub mappings: Vec<MaterialMapping>,
    pub extensions: Vec<Extension>,
}

#[derive(Debug, Clone, Default)]
pub struct Mesh {
    pub name: Option<String>,
    pub primitives: Vec<Primitive>,
    /// `None` when the JSON has no `weights` (C: NULL)
    pub weights: Option<Vec<f32>>,
    /// `None` when the JSON has no `extras.targetNames` (C: NULL)
    pub target_names: Option<Vec<String>>,
    pub extras: Extras,
    pub extensions: Vec<Extension>,
}

#[derive(Debug, Clone, Default)]
pub struct Skin {
    pub name: Option<String>,
    pub joints: Vec<usize>,
    pub skeleton: Option<usize>,
    pub inverse_bind_matrices: Option<usize>,
    pub extras: Extras,
    pub extensions: Vec<Extension>,
}

#[derive(Debug, Clone, Default)]
pub struct CameraPerspective {
    pub has_aspect_ratio: bool,
    pub aspect_ratio: f32,
    pub yfov: f32,
    pub has_zfar: bool,
    pub zfar: f32,
    pub znear: f32,
    pub extras: Extras,
}

#[derive(Debug, Clone, Default)]
pub struct CameraOrthographic {
    pub xmag: f32,
    pub ymag: f32,
    pub zfar: f32,
    pub znear: f32,
    pub extras: Extras,
}

#[derive(Debug, Clone, Default)]
pub struct Camera {
    pub name: Option<String>,
    pub type_: CameraType,
    /// C union member, valid when `type_ == Perspective`
    pub perspective: CameraPerspective,
    /// C union member, valid when `type_ == Orthographic`
    pub orthographic: CameraOrthographic,
    pub extras: Extras,
    pub extensions: Vec<Extension>,
}

#[derive(Debug, Clone, Default)]
pub struct Light {
    pub name: Option<String>,
    pub color: [f32; 3],
    pub intensity: f32,
    pub type_: LightType,
    pub range: f32,
    pub spot_inner_cone_angle: f32,
    pub spot_outer_cone_angle: f32,
    pub extras: Extras,
}

#[derive(Debug, Clone, Default)]
pub struct Node {
    pub name: Option<String>,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub skin: Option<usize>,
    pub mesh: Option<usize>,
    pub camera: Option<usize>,
    pub light: Option<usize>,
    /// `None` when the JSON has no `weights` (C: NULL)
    pub weights: Option<Vec<f32>>,
    pub has_translation: bool,
    pub has_rotation: bool,
    pub has_scale: bool,
    pub has_matrix: bool,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub matrix: [f32; 16],
    pub extras: Extras,
    pub has_mesh_gpu_instancing: bool,
    pub mesh_gpu_instancing: MeshGpuInstancing,
    pub extensions: Vec<Extension>,
}

#[derive(Debug, Clone, Default)]
pub struct Scene {
    pub name: Option<String>,
    pub nodes: Vec<usize>,
    pub extras: Extras,
    pub extensions: Vec<Extension>,
}

#[derive(Debug, Clone, Default)]
pub struct AnimationSampler {
    /// required reference (C: `CGLTF_PTRFIXUP_REQ`)
    pub input: usize,
    /// required reference (C: `CGLTF_PTRFIXUP_REQ`)
    pub output: usize,
    pub interpolation: InterpolationType,
    pub extras: Extras,
    pub extensions: Vec<Extension>,
}

#[derive(Debug, Clone, Default)]
pub struct AnimationChannel {
    /// index into the owning animation's `samplers`; required reference (C: `CGLTF_PTRFIXUP_REQ`)
    pub sampler: usize,
    pub target_node: Option<usize>,
    pub target_path: AnimationPathType,
    pub extras: Extras,
    pub extensions: Vec<Extension>,
}

#[derive(Debug, Clone, Default)]
pub struct Animation {
    pub name: Option<String>,
    pub samplers: Vec<AnimationSampler>,
    pub channels: Vec<AnimationChannel>,
    pub extras: Extras,
    pub extensions: Vec<Extension>,
}

#[derive(Debug, Clone, Default)]
pub struct MaterialVariant {
    pub name: Option<String>,
    pub extras: Extras,
}

#[derive(Debug, Clone, Default)]
pub struct Asset {
    pub copyright: Option<String>,
    pub generator: Option<String>,
    pub version: Option<String>,
    pub min_version: Option<String>,
    pub extras: Extras,
    pub extensions: Vec<Extension>,
}

#[derive(Debug, Clone, Default)]
pub struct Data {
    pub file_type: FileType,
    /// whole input file for `parse_file` (C: `file_data`/`file_size`)
    pub file_data: Option<Vec<u8>>,

    pub asset: Asset,

    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub accessors: Vec<Accessor>,
    pub buffer_views: Vec<BufferView>,
    pub buffers: Vec<Buffer>,
    pub images: Vec<Image>,
    pub textures: Vec<Texture>,
    pub samplers: Vec<Sampler>,
    pub skins: Vec<Skin>,
    pub cameras: Vec<Camera>,
    pub lights: Vec<Light>,
    pub nodes: Vec<Node>,
    pub scenes: Vec<Scene>,

    pub scene: Option<usize>,

    pub animations: Vec<Animation>,
    pub variants: Vec<MaterialVariant>,

    pub extras: Extras,

    pub data_extensions: Vec<Extension>,

    pub extensions_used: Vec<String>,
    pub extensions_required: Vec<String>,

    /// JSON chunk (C: `json`/`json_size`)
    pub json: Option<Vec<u8>>,

    /// GLB BIN chunk (C: `bin`/`bin_size`); moved into `buffers[0].data` by `load_buffers`
    pub bin: Option<Vec<u8>>,
    pub bin_size: CgltfSize,
}
