//! Gaussian splat input and encoding (`splats.cpp`).

use crate::cgltf::{self, AttributeType, ComponentType, Data, PrimitiveType, Type};
use crate::fileio::read_file;
use crate::micro_ply::{self, PLY_PROP_DECIMAL, PlyElement, PlyProp};
use crate::stream::{append_filter_exp, attr_floats, push_f32s};
use crate::types::*;
use meshoptimizer as mo;

pub const SPLAT_POSITION: i32 = 0;
pub const SPLAT_ROTATION: i32 = 1;
pub const SPLAT_SCALE: i32 = 2;
pub const SPLAT_OPACITY: i32 = 3;
pub const SPLAT_SH0: i32 = 4;
pub const SPLAT_SH1: i32 = 5;
pub const SPLAT_SH2: i32 = 6;
pub const SPLAT_SH3: i32 = 7;

fn find_prop_float<'a>(element: &'a PlyElement, name: Option<&str>) -> Option<&'a PlyProp> {
    let name = name?;

    element
        .properties
        .iter()
        .find(|p| p.name == name && p.type_ == PLY_PROP_DECIMAL && p.bytes == 4)
}

fn read_props(result: &mut Vec<Attr>, element: &PlyElement, x: Option<&str>, y: Option<&str>, z: Option<&str>, w: Option<&str>) -> bool {
    let props = [x, y, z, w].map(|n| (n, find_prop_float(element, n)));

    if props.iter().any(|(n, p)| n.is_some() && p.is_none()) {
        return false;
    }

    let count = element.count.max(0) as usize;
    let stride = element.data_stride.max(0) as usize;

    result.resize(count, [0.0; 4]);

    for (i, target) in result.iter_mut().enumerate() {
        for (k, (_, p)) in props.iter().enumerate() {
            if let Some(p) = p {
                let offset = i * stride + usize::from(p.offset);
                if let Some(b) = element.data.get(offset..offset + 4) {
                    target[k] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
            }
        }
    }

    true
}

fn count_sh_props(element: &PlyElement) -> i32 {
    for i in 0..=45 {
        if find_prop_float(element, Some(&format!("f_rest_{i}"))).is_none() {
            return i;
        }
    }

    // too many props; f_rest_45 implies >3 bands
    -1
}

fn read_sh_props(result: &mut [Vec<Attr>], vtxe: &PlyElement, offset: i32, stride: i32) -> bool {
    for (i, r) in result.iter_mut().enumerate() {
        let i = i as i32;
        let names = [0, 1, 2].map(|k| format!("f_rest_{}", offset + i + stride * k));

        if !read_props(r, vtxe, Some(&names[0]), Some(&names[1]), Some(&names[2]), None) {
            return false;
        }
    }

    true
}

#[allow(clippy::too_many_arguments)]
fn convert_splats(
    position: &mut [Attr],
    rotation: &mut [Attr],
    scale: &mut [Attr],
    opacity: &mut [Attr],
    sh1: &mut [Vec<Attr>; 3],
    sh2: &mut [Vec<Attr>; 5],
    sh3: &mut [Vec<Attr>; 7],
    shdim: i32,
) {
    for i in 0..position.len() {
        // PLY splats store scale in log units, and opacity before the sigmoid activation
        for k in 0..3 {
            scale[i][k] = scale[i][k].exp();
        }

        opacity[i][0] = 1.0 / (1.0 + (-opacity[i][0]).exp());

        // PLY splats are conventionally encoded in COLMAP coordinate space, with -X right -Y up -Z forward; glTF uses -X right +Y up +Z forward
        // the conversion requires a fixup of positions, rotations, and SH bands, flipping Y & Z signs
        position[i][1] *= -1.0;
        position[i][2] *= -1.0;

        rotation[i][1] *= -1.0;
        rotation[i][2] *= -1.0;

        // PLY rotation quaternions may or may not be normalized; normalize them just in case
        let r = rotation[i];
        let ql = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]).sqrt();

        for k in 0..4 {
            rotation[i][k] /= ql;
        }

        // for SH conversion, since coordinate space xform is flipping Y & Z, we need to flip coefficients for bands that encode odd-length products of Y & Z
        for k in 0..3 {
            if shdim >= 1 {
                sh1[0][i][k] *= -1.0;
                sh1[1][i][k] *= -1.0;
            }

            if shdim >= 2 {
                sh2[0][i][k] *= -1.0;
                sh2[3][i][k] *= -1.0;
            }

            if shdim >= 3 {
                sh3[0][i][k] *= -1.0;
                sh3[2][i][k] *= -1.0;
                sh3[3][i][k] *= -1.0;
                sh3[5][i][k] *= -1.0;
            }
        }
    }
}

fn append_stream(mesh: &mut Mesh, data: Vec<Attr>, type_: AttributeType, attr: i32, index: i32) {
    mesh.streams.push(Stream {
        type_,
        index,
        splats: attr,
        data,
        ..Default::default()
    });
}

fn create_scene() -> Data {
    Data {
        scenes: vec![cgltf::Scene::default()],
        scene: Some(0),
        ..Default::default()
    }
}

/// `parseSplats`: returns the scene or an error message.
pub fn parse_splats(path: &str, meshes: &mut Vec<Mesh>) -> Result<Data, &'static str> {
    let plydata = read_file(path).ok_or("file not found")?;

    let file = micro_ply::ply_read(&plydata).ok_or("invalid PLY")?;

    if file.elements.len() != 1 || file.elements[0].name != "vertex" || file.elements[0].list_data.is_some() {
        return Err("invalid 3DGS PLY: expected vertex array");
    }

    let vtxe = &file.elements[0];

    let mut position = Vec::new();
    let mut rotation = Vec::new();
    let mut scale = Vec::new();
    let mut opacity = Vec::new();
    let mut sh0 = Vec::new();

    if !read_props(&mut position, vtxe, Some("x"), Some("y"), Some("z"), None)
        || !read_props(&mut rotation, vtxe, Some("rot_1"), Some("rot_2"), Some("rot_3"), Some("rot_0"))
        || !read_props(&mut scale, vtxe, Some("scale_0"), Some("scale_1"), Some("scale_2"), None)
        || !read_props(&mut opacity, vtxe, Some("opacity"), None, None, None)
        || !read_props(&mut sh0, vtxe, Some("f_dc_0"), Some("f_dc_1"), Some("f_dc_2"), None)
    {
        return Err("invalid 3DGS PLY: missing required attributes");
    }

    let shprops = count_sh_props(vtxe);
    let shdim = match shprops {
        45 => 3,
        24 => 2,
        9 => 1,
        0 => 0,
        _ => -1,
    };

    let mut sh1: [Vec<Attr>; 3] = Default::default();
    let mut sh2: [Vec<Attr>; 5] = Default::default();
    let mut sh3: [Vec<Attr>; 7] = Default::default();

    if shdim < 0
        || (shdim >= 1 && !read_sh_props(&mut sh1, vtxe, 0, shprops / 3))
        || (shdim >= 2 && !read_sh_props(&mut sh2, vtxe, 3, shprops / 3))
        || (shdim >= 3 && !read_sh_props(&mut sh3, vtxe, 8, shprops / 3))
    {
        return Err("invalid 3DGS PLY: inconsistent SH attributes");
    }

    convert_splats(
        &mut position,
        &mut rotation,
        &mut scale,
        &mut opacity,
        &mut sh1,
        &mut sh2,
        &mut sh3,
        shdim,
    );

    let mut mesh = Mesh {
        type_: PrimitiveType::Points,
        splats: true,
        ..Default::default()
    };

    append_stream(&mut mesh, position, AttributeType::Position, SPLAT_POSITION, 0);
    append_stream(&mut mesh, rotation, AttributeType::Custom, SPLAT_ROTATION, 0);
    append_stream(&mut mesh, scale, AttributeType::Custom, SPLAT_SCALE, 0);
    append_stream(&mut mesh, opacity, AttributeType::Custom, SPLAT_OPACITY, 0);
    append_stream(&mut mesh, sh0, AttributeType::Custom, SPLAT_SH0, 0);

    if shdim >= 1 {
        for (k, s) in sh1.into_iter().enumerate() {
            append_stream(&mut mesh, s, AttributeType::Custom, SPLAT_SH1, k as i32);
        }
    }

    if shdim >= 2 {
        for (k, s) in sh2.into_iter().enumerate() {
            append_stream(&mut mesh, s, AttributeType::Custom, SPLAT_SH2, k as i32);
        }
    }

    if shdim >= 3 {
        for (k, s) in sh3.into_iter().enumerate() {
            append_stream(&mut mesh, s, AttributeType::Custom, SPLAT_SH3, k as i32);
        }
    }

    meshes.push(mesh);

    Ok(create_scene())
}

fn get_splat_stream(mesh: &Mesh, type_: AttributeType, attr: i32, index: i32) -> Option<usize> {
    mesh.streams
        .iter()
        .position(|s| s.type_ == type_ && s.index == index && s.splats == attr)
}

pub fn process_splat_mesh(mesh: &mut Mesh, settings: &Settings) {
    let positions = get_splat_stream(mesh, AttributeType::Position, SPLAT_POSITION, 0);
    let opacity = get_splat_stream(mesh, AttributeType::Custom, SPLAT_OPACITY, 0);
    let (Some(positions), Some(opacity)) = (positions, opacity) else {
        return;
    };

    // sort splats by position using Morton order
    let vertex_count = mesh.streams[0].data.len();

    let mut remap = vec![0u32; vertex_count];
    let _ = mo::spatial_sort_remap(&mut remap, attr_floats(&mesh.streams[positions].data), vertex_count, 16);

    let mut result_count = vertex_count;

    // when quantization is enabled, we can remove splats with low opacity
    // for now this is gated behind -cc so that -c produces original splat set
    if settings.quantize && settings.compressmore {
        let opacitymin = 2.0 / (1i32.wrapping_shl(settings.col_bits as u32)) as f32;

        // mark slots of surviving splats and number them in the remapped order (removed splats use ~0u)
        let mut slots = vec![!0u32; vertex_count];

        for i in 0..vertex_count {
            if mesh.streams[opacity].data[i][0] >= opacitymin {
                slots[remap[i] as usize] = 0;
            }
        }

        result_count = 0;
        for slot in slots.iter_mut() {
            if *slot != !0u32 {
                *slot = result_count as u32;
                result_count += 1;
            }
        }

        // update old->new remap; surviving splats get the new linearized index
        for r in remap.iter_mut() {
            *r = slots[*r as usize];
        }
    }

    for stream in mesh.streams.iter_mut() {
        let source = stream.data.clone();
        for (i, &r) in remap.iter().enumerate().take(source.len()) {
            if r != !0u32
                && let Some(d) = stream.data.get_mut(r as usize)
            {
                *d = source[i];
            }
        }
        stream.data.truncate(result_count);
    }

    // filter streams with excessive SH data
    if settings.splats_dim < 3 {
        mesh.streams.retain(|stream| {
            !(stream.type_ == AttributeType::Custom
                && stream.splats >= SPLAT_SH1
                && stream.splats <= SPLAT_SH3
                && (stream.splats - SPLAT_SH0) > settings.splats_dim)
        });
    }
}

pub fn splat_attribute_type(type_: i32) -> &'static str {
    match type_ {
        SPLAT_POSITION => "POSITION",
        SPLAT_ROTATION => "ROTATION",
        SPLAT_SCALE => "SCALE",
        SPLAT_OPACITY => "OPACITY",
        SPLAT_SH0 => "SH_DEGREE_0_COEF",
        SPLAT_SH1 => "SH_DEGREE_1_COEF",
        SPLAT_SH2 => "SH_DEGREE_2_COEF",
        SPLAT_SH3 => "SH_DEGREE_3_COEF",
        _ => "",
    }
}

fn write_splat_stream_raw(bin: &mut Vec<u8>, stream: &Stream, type_: Type, components: usize) -> StreamFormat {
    for a in &stream.data {
        push_f32s(bin, &a[..components]);
    }

    StreamFormat::new(type_, ComponentType::R32f, false, 4 * components, Filter::None)
}

#[allow(clippy::too_many_arguments)]
fn write_splat_stream_float(
    bin: &mut Vec<u8>,
    stream: &Stream,
    type_: Type,
    components: usize,
    expf: bool,
    bits: i32,
    mode: mo::EncodeExpMode,
) -> StreamFormat {
    let filter = if expf { Filter::Exp } else { Filter::None };

    if filter == Filter::Exp {
        append_filter_exp(bin, &stream.data, components, bits + 1, mode);
    } else {
        for a in &stream.data {
            let v: Vec<f32> = a[..components].iter().map(|&x| mo::quantize_float(x, bits).unwrap_or(x)).collect();
            push_f32s(bin, &v);
        }
    }

    StreamFormat::new(type_, ComponentType::R32f, false, 4 * components, filter)
}

fn write_splat_stream_exp_sh(bin: &mut Vec<u8>, stream: &Stream, bits: i32, range: i32) -> StreamFormat {
    // values in (-2^range, 2^range) are encoded with fixed 2^-bits precision (like meshopt_EncodeExpClamped, but we need custom range support)
    // to reduce the cost of sign flips on upper mantissa bytes, we try to shift mantissa left and leave low bytes as zero
    let mbits = bits + range;
    let shift = if mbits + 1 <= 8 {
        16
    } else if mbits + 1 <= 16 {
        8
    } else {
        0
    };
    let scale = 2f32.powi(-range);

    bin.reserve(stream.data.len() * 12);

    for a in &stream.data {
        let v = [a[0] * scale, a[1] * scale, a[2] * scale];
        let mut rb = [0u8; 12];
        let _ = mo::encode_filter_exp(&mut rb, 1, 12, mbits + 1, &v, mo::EncodeExpMode::Clamped);

        for k in 0..3 {
            let rk = u32::from_le_bytes([rb[k * 4], rb[k * 4 + 1], rb[k * 4 + 2], rb[k * 4 + 3]]);
            let mut m = ((rk << 8) as i32) >> 8;
            let e = (rk as i32) >> 24;

            // rounding may produce mantissa 2^mbits; clamp it so that m with sign bit fits into mbits+1 bits and shifting does not overflow
            let mmax = 1i32.wrapping_shl(mbits as u32);
            m = (-(mmax - 1)).max(m.min(mmax - 1));

            let r = (((m as u32).wrapping_shl(shift)) & 0xff_ffff) | ((e + range - shift as i32) as u32).wrapping_shl(24);
            bin.extend_from_slice(&r.to_le_bytes());
        }
    }

    StreamFormat::new(Type::Vec3, ComponentType::R32f, false, 12, Filter::Exp)
}

/// `encodeSnorm` (splats.cpp): adjusts the scale so that bit counts below the storage size dequantize exactly.
fn encode_snorm(destination: &mut [u8], count: usize, stride: usize, bits: i32, data: &[f32]) {
    // to support bits != 8/16, we need to adjust the quantization scale so that division by 127/32767 during dequantization does not lose precision
    // this is a no-op if bits matches stride, as step=1 and scale=1.0f
    let range: i32 = if stride == 4 { 128 } else { 32768 };
    let step: i32 = 1i32.wrapping_shl(((if stride == 4 { 8 } else { 16 }) - bits) as u32);
    let scale = (range - 1) as f32 / (range - step) as f32;

    for i in 0..count {
        let v = &data[i * 4..i * 4 + 4];

        let f: [i32; 4] = [0, 1, 2, 3].map(|k| mo::quantize_snorm(v[k] * scale, bits).wrapping_mul(step));

        if stride == 4 {
            for k in 0..4 {
                destination[i * 4 + k] = f[k] as i8 as u8;
            }
        } else {
            for k in 0..4 {
                destination[i * 8 + k * 2..i * 8 + k * 2 + 2].copy_from_slice(&(f[k] as i16).to_le_bytes());
            }
        }
    }
}

pub fn write_splat_stream(bin: &mut Vec<u8>, stream: &Stream, settings: &Settings) -> StreamFormat {
    use mo::EncodeExpMode as M;

    let attr = if stream.type_ == AttributeType::Position {
        SPLAT_POSITION
    } else {
        stream.splats
    };

    match attr {
        SPLAT_POSITION => {
            if !settings.quantize {
                return write_splat_stream_raw(bin, stream, Type::Vec3, 3);
            }

            write_splat_stream_float(bin, stream, Type::Vec3, 3, settings.compress, settings.pos_bits, M::SharedComponent)
        }
        SPLAT_ROTATION => {
            if !settings.quantize {
                return write_splat_stream_raw(bin, stream, Type::Vec4, 4);
            }

            let filter = if settings.compress { Filter::Quat } else { Filter::None };
            let bits = (settings.nrm_bits + if settings.compressmore { 0 } else { 1 }).clamp(4, 16); // quaternion filter supports 4-16 bits

            let offset = bin.len();
            let stride = 8;
            bin.resize(offset + stream.data.len() * stride, 0);

            let floats = attr_floats(&stream.data);
            if filter == Filter::Quat {
                let _ = mo::encode_filter_quat(&mut bin[offset..], stream.data.len(), stride, bits, floats);
            } else {
                encode_snorm(&mut bin[offset..], stream.data.len(), stride, 16, floats);
            }

            if filter == Filter::Quat && bits == 8 && settings.compressmore {
                // with 8-bit components, we left-align the 16-bit components to keep the least significant byte as 0; this compresses a little better
                // technically, .w now encodes a scale that's not quite 1.0; this works but is out of spec. in the future we could adjust the quantizer so that .w can encode 0x7ffc
                for chunk in bin[offset..].as_chunks_mut::<8>().0 {
                    let mut v = [0i16; 4];
                    for k in 0..4 {
                        v[k] = i16::from_le_bytes([chunk[k * 2], chunk[k * 2 + 1]]);
                    }

                    for k in 0..3 {
                        v[k] = ((v[k] as i32 as u32) << 8) as i16;
                    }
                    let w = i32::from(v[3]);
                    v[3] = (((w | 3).wrapping_shl(8)) | (w & 3)) as i16;

                    for k in 0..4 {
                        chunk[k * 2..k * 2 + 2].copy_from_slice(&v[k].to_le_bytes());
                    }
                }
            }

            StreamFormat::new(Type::Vec4, ComponentType::R16, true, 8, filter)
        }
        SPLAT_SCALE => {
            if !settings.quantize {
                return write_splat_stream_raw(bin, stream, Type::Vec3, 3);
            }

            write_splat_stream_float(bin, stream, Type::Vec3, 3, settings.compress, settings.tex_bits, M::SharedComponent)
        }
        SPLAT_OPACITY => {
            if !settings.quantize {
                return write_splat_stream_raw(bin, stream, Type::Scalar, 1);
            }

            write_splat_stream_float(bin, stream, Type::Scalar, 1, settings.compress, settings.col_bits, M::Clamped)
        }
        SPLAT_SH0 => {
            if !settings.quantize {
                return write_splat_stream_raw(bin, stream, Type::Vec3, 3);
            }

            if !settings.compress {
                return write_splat_stream_float(bin, stream, Type::Vec3, 3, false, settings.col_bits, M::SharedComponent);
            }

            // -2 accounts for a larger range [-2..2] with a sign bit
            let bits = (settings.col_bits - 2 - if settings.compressmore { 1 } else { 0 }).max(1);
            write_splat_stream_exp_sh(bin, stream, bits, 1)
        }
        SPLAT_SH1 | SPLAT_SH2 | SPLAT_SH3 => {
            if !settings.quantize {
                return write_splat_stream_raw(bin, stream, Type::Vec3, 3);
            }

            if settings.compressmore {
                // when using compressmore (-cc) we encode SH1-3 as normalized int8 to reduce decoding memory pressure and file sizes
                // this is outside of the spec, but de-facto safe for many viewers and represent a significant spec gap that should be closed
                let offset = bin.len();
                let stride = 4;
                bin.resize(offset + stream.data.len() * stride, 0);

                // 5/4/4 at default settings, including sign bit
                let bits = (settings.col_bits - if attr == SPLAT_SH1 { 3 } else { 4 }).clamp(1, 8);
                encode_snorm(&mut bin[offset..], stream.data.len(), stride, bits, attr_floats(&stream.data));

                StreamFormat::new(Type::Vec3, ComponentType::R8, true, 4, Filter::None)
            } else {
                // 6/5/5 at default settings, excluding sign bit
                let bits = (settings.col_bits - if attr == SPLAT_SH1 { 2 } else { 3 }).max(1);

                if !settings.compress {
                    return write_splat_stream_float(bin, stream, Type::Vec3, 3, false, bits, M::Clamped);
                }

                write_splat_stream_exp_sh(bin, stream, bits, 0)
            }
        }
        _ => StreamFormat::default(),
    }
}
