//! Vertex/index/animation stream quantization and encoding (`stream.cpp`).
//!
//! meshoptimizer calls that C++ guards with (disabled) asserts return `Err` in Rust; such streams are left zeroed.

use crate::cgltf::{AnimationPathType, AttributeType, ComponentType, Data, PrimitiveType, Type};
use crate::cstd::format_f;
use crate::cxx::{f2i32, f2u8, f2u16, f2u32, fmax, fmin};
use crate::types::*;
use meshoptimizer as mo;

#[derive(Clone, Copy)]
struct Bounds {
    min: Attr,
    max: Attr,
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            min: [f32::MAX; 4],
            max: [-f32::MAX; 4],
        }
    }
}

impl Bounds {
    fn is_valid(&self) -> bool {
        (0..4).all(|k| self.min[k] <= self.max[k])
    }

    fn get_extent(&self) -> f32 {
        fmax(
            self.max[0] - self.min[0],
            fmax(self.max[1] - self.min[1], self.max[2] - self.min[2]),
        )
    }

    fn merge(&mut self, other: &Bounds) {
        for k in 0..4 {
            self.min[k] = fmin(self.min[k], other.min[k]);
            self.max[k] = fmax(self.max[k], other.max[k]);
        }
    }
}

pub(crate) fn push_f32s(bin: &mut Vec<u8>, v: &[f32]) {
    for x in v {
        bin.extend_from_slice(&x.to_le_bytes());
    }
}

fn push_u16s(bin: &mut Vec<u8>, v: &[u16]) {
    for x in v {
        bin.extend_from_slice(&x.to_le_bytes());
    }
}

fn push_i16s(bin: &mut Vec<u8>, v: &[i16]) {
    for x in v {
        bin.extend_from_slice(&x.to_le_bytes());
    }
}

/// Flattens a slice of `Attr` into floats (`stream.data[0].f`).
pub(crate) fn attr_floats(data: &[Attr]) -> &[f32] {
    data.as_flattened()
}

fn compute_bounds(mesh: &Mesh, type_: AttributeType) -> Bounds {
    let mut b = Bounds::default();
    let mut pad: Attr = [0.0; 4];

    for s in &mesh.streams {
        if s.type_ != type_ {
            continue;
        }

        if s.target == 0 {
            for a in &s.data {
                for k in 0..4 {
                    b.min[k] = fmin(b.min[k], a[k]);
                    b.max[k] = fmax(b.max[k], a[k]);
                }
            }
        } else {
            for a in &s.data {
                for k in 0..4 {
                    pad[k] = fmax(pad[k], a[k].abs());
                }
            }
        }
    }

    for k in 0..4 {
        b.min[k] -= pad[k];
        b.max[k] += pad[k];
    }

    b
}

fn compute_uv_area(mesh: &Mesh) -> f32 {
    if mesh.indices.is_empty() || mesh.type_ != PrimitiveType::Triangles {
        return 0.0;
    }

    let mut result = 0.0f32;

    for s in &mesh.streams {
        if s.type_ != AttributeType::Texcoord {
            continue;
        }

        let at = |i: u32| s.data.get(i as usize).copied().unwrap_or_default();
        let mut uvarea = 0.0f32;

        for tri in mesh.indices.as_chunks::<3>().0 {
            let (va, vb, vc) = (at(tri[0]), at(tri[1]), at(tri[2]));

            uvarea += ((vb[0] - va[0]) * (vc[1] - va[1]) - (vc[0] - va[0]) * (vb[1] - va[1])).abs();
        }

        result = fmax(result, uvarea / (mesh.indices.len() / 3) as f32);
    }

    result
}

pub fn prepare_quantization_position(meshes: &[Mesh], settings: &Settings) -> QuantizationPosition {
    let mut result = QuantizationPosition {
        bits: settings.pos_bits,
        normalized: settings.pos_normalized,
        ..Default::default()
    };

    let bounds: Vec<Bounds> = meshes.iter().map(|m| compute_bounds(m, AttributeType::Position)).collect();

    let mut b = Bounds::default();
    for mb in &bounds {
        b.merge(mb);
    }

    if b.is_valid() {
        result.offset = [b.min[0], b.min[1], b.min[2]];
        result.scale = b.get_extent();
    }

    if b.is_valid() && settings.quantize && !settings.pos_float {
        let error = result.scale * 0.5 / (1i32.wrapping_shl((result.bits - 1) as u32)) as f32;
        let mut max_rel_error = 0.0f32;

        for mb in &bounds {
            if mb.is_valid() && mb.get_extent() > 1e-2 {
                max_rel_error = fmax(max_rel_error, error / mb.get_extent());
            }
        }

        if max_rel_error > 5e-2 {
            eprintln!(
                "Warning: position data has significant error ({}%); consider using floating-point quantization (-vpf) or more bits (-vp N)",
                format_f(f64::from(max_rel_error * 100.0), 0)
            );
        }
    }

    result.node_scale =
        result.scale / ((1i32.wrapping_shl(result.bits as u32)).wrapping_sub(1)) as f32 * if result.normalized { 65535.0 } else { 1.0 };

    result
}

fn follow(parents: &mut [usize], mut index: usize) -> usize {
    while index != parents[index] {
        let parent = parents[index];

        parents[index] = parents[parent];
        index = parent;
    }

    index
}

pub fn prepare_quantization_texture(
    _data: &Data,
    result: &mut [QuantizationTexture],
    indices: &mut [usize],
    meshes: &[Mesh],
    settings: &Settings,
) {
    // use union-find to associate each material with a canonical material
    // this is necessary because any set of materials that are used on the same mesh must use the same quantization
    let mut parents: Vec<usize> = (0..result.len()).collect();

    let material_of = |mesh: &Mesh| mesh.material.or_else(|| mesh.variants.first().map(|v| v.material));

    for (i, mesh) in meshes.iter().enumerate() {
        let Some(material) = material_of(mesh) else {
            continue;
        };

        let root = follow(&mut parents, material);

        for variant in &mesh.variants {
            let var = follow(&mut parents, variant.material);

            parents[var] = root;
        }

        indices[i] = root;
    }

    // compute canonical material bounds based on meshes that use them
    let mut bounds = vec![Bounds::default(); result.len()];

    for (i, mesh) in meshes.iter().enumerate() {
        if material_of(mesh).is_none() {
            continue;
        }

        indices[i] = follow(&mut parents, indices[i]);

        let mb = compute_bounds(mesh, AttributeType::Texcoord);
        bounds[indices[i]].merge(&mb);
    }

    // detect potential precision issues and warn about them
    if settings.quantize && !settings.tex_float {
        let mut max_rel_error = 0.0f32;

        for (i, mesh) in meshes.iter().enumerate() {
            if material_of(mesh).is_none() {
                continue;
            }

            let b = &bounds[indices[i]];
            if !b.is_valid() {
                continue;
            }

            let scale = fmax(b.max[0] - b.min[0], b.max[1] - b.min[1]);
            let error = scale * 0.5 / (1i32.wrapping_shl((settings.tex_bits - 1) as u32)) as f32;

            if error < 1e-3 {
                continue;
            }

            let uvarea = compute_uv_area(mesh);
            let rel_error = if uvarea > 0.0 { error / uvarea.sqrt() } else { 0.0 };

            max_rel_error = fmax(max_rel_error, rel_error);
        }

        if max_rel_error > 1e-1 {
            eprintln!(
                "Warning: texture coordinate data has significant error ({}%); consider using floating-point quantization (-vtf) or more bits (-vt N)",
                format_f(f64::from(max_rel_error * 100.0), 0)
            );
        }
    }

    // update all material data using canonical bounds
    for i in 0..result.len() {
        let qt = &mut result[i];

        qt.bits = settings.tex_bits;
        qt.normalized = true;

        let b = bounds[follow(&mut parents, i)];

        if b.is_valid() {
            qt.offset = [b.min[0], b.min[1]];
            qt.scale = [b.max[0] - b.min[0], b.max[1] - b.min[1]];
        }
    }
}

fn quantize_float(v: f32, bits: i32) -> f32 {
    mo::quantize_float(v, bits).unwrap_or(v)
}

pub fn get_position_bounds(stream: &Stream, qp: &QuantizationPosition, settings: &Settings) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX; 3];
    let mut max = [-f32::MAX; 3];

    for a in &stream.data {
        for k in 0..3 {
            min[k] = fmin(min[k], a[k]);
            max[k] = fmax(max[k], a[k]);
        }
    }

    if settings.quantize {
        if settings.pos_float {
            for k in 0..3 {
                min[k] = quantize_float(min[k], qp.bits);
                max[k] = quantize_float(max[k], qp.bits);
            }
        } else {
            let pos_rscale = if qp.scale == 0.0 {
                0.0
            } else {
                1.0 / qp.scale
                    * if stream.target > 0 && qp.normalized {
                        32767.0 / 65535.0
                    } else {
                        1.0
                    }
            };

            for k in 0..3 {
                if stream.target == 0 {
                    min[k] = mo::quantize_unorm((min[k] - qp.offset[k]) * pos_rscale, qp.bits) as f32;
                    max[k] = mo::quantize_unorm((max[k] - qp.offset[k]) * pos_rscale, qp.bits) as f32;
                } else {
                    min[k] = (if min[k] >= 0.0 { 1.0 } else { -1.0 }) * mo::quantize_unorm(min[k].abs() * pos_rscale, qp.bits) as f32;
                    max[k] = (if max[k] >= 0.0 { 1.0 } else { -1.0 }) * mo::quantize_unorm(max[k].abs() * pos_rscale, qp.bits) as f32;
                }
            }
        }
    }

    (min, max)
}

fn renormalize_weights(w: &mut [u8; 4]) {
    let sum: i32 = w.iter().map(|&x| i32::from(x)).sum();

    if sum == 255 {
        return;
    }

    // we assume that the total error is limited to 0.5/component = 2
    // this means that it's acceptable to adjust the max. component to compensate for the error
    let mut max = 0;

    for k in 1..4 {
        if w[k] > w[max] {
            max = k;
        }
    }

    w[max] = w[max].wrapping_add((255 - sum) as u8);
}

/// `encodeSnorm` (stream.cpp): `data` holds 4 floats per element.
fn encode_snorm(destination: &mut [u8], count: usize, stride: usize, bits: i32, data: &[f32]) {
    for i in 0..count {
        let v = &data[i * 4..i * 4 + 4];

        let f: [i32; 4] = [0, 1, 2, 3].map(|k| mo::quantize_snorm(v[k], bits));

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

fn quantize_color(v: f32, bytebits: i32, bits: i32) -> i32 {
    let result = mo::quantize_unorm(v, bytebits);

    // replicate the top bit into the low significant bits
    let mask = 1i32.wrapping_shl((bytebits - bits) as u32).wrapping_sub(1);

    (result & !mask) | (mask & (result >> (bytebits - 1)).wrapping_neg())
}

fn encode_color(destination: &mut [u8], count: usize, stride: usize, bits: i32, data: &[f32]) {
    for i in 0..count {
        let c = &data[i * 4..i * 4 + 4];

        if stride == 4 {
            for k in 0..4 {
                destination[i * 4 + k] = quantize_color(c[k], 8, bits) as u8;
            }
        } else {
            for k in 0..4 {
                let v = quantize_color(c[k], 16, bits) as u16;
                destination[i * 8 + k * 2..i * 8 + k * 2 + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
}

fn write_vertex_stream_raw(bin: &mut Vec<u8>, stream: &Stream, type_: Type, components: usize) -> StreamFormat {
    for a in &stream.data {
        push_f32s(bin, &a[..components]);
    }

    StreamFormat::new(type_, ComponentType::R32f, false, 4 * components, Filter::None)
}

/// Exponential filter encoding of `components` floats per element, appended to `bin`.
pub(crate) fn append_filter_exp(bin: &mut Vec<u8>, data: &[Attr], components: usize, bits: i32, mode: mo::EncodeExpMode) {
    let stride = 4 * components;
    let floats: Vec<f32> = data.iter().flat_map(|a| a[..components].iter().copied()).collect();

    let offset = bin.len();
    bin.resize(offset + data.len() * stride, 0);
    let _ = mo::encode_filter_exp(&mut bin[offset..], data.len(), stride, bits, &floats, mode);
}

fn write_vertex_stream_float(
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
            let v: Vec<f32> = a[..components].iter().map(|&x| quantize_float(x, bits)).collect();
            push_f32s(bin, &v);
        }
    }

    StreamFormat::new(type_, ComponentType::R32f, false, 4 * components, filter)
}

pub fn write_vertex_stream(
    bin: &mut Vec<u8>,
    stream: &Stream,
    qp: &QuantizationPosition,
    qt: &QuantizationTexture,
    settings: &Settings,
    filters: bool,
) -> StreamFormat {
    use mo::EncodeExpMode as M;

    match stream.type_ {
        AttributeType::Position => {
            if !settings.quantize {
                return write_vertex_stream_raw(bin, stream, Type::Vec3, 3);
            }

            if settings.pos_float {
                let mode = if settings.compressmore { M::SharedComponent } else { M::Separate };
                return write_vertex_stream_float(bin, stream, Type::Vec3, 3, settings.compress && filters, qp.bits, mode);
            }

            if stream.target == 0 {
                let pos_rscale = if qp.scale == 0.0 { 0.0 } else { 1.0 / qp.scale };

                for a in &stream.data {
                    let v = [
                        mo::quantize_unorm((a[0] - qp.offset[0]) * pos_rscale, qp.bits) as u16,
                        mo::quantize_unorm((a[1] - qp.offset[1]) * pos_rscale, qp.bits) as u16,
                        mo::quantize_unorm((a[2] - qp.offset[2]) * pos_rscale, qp.bits) as u16,
                        0,
                    ];
                    push_u16s(bin, &v);
                }

                StreamFormat::new(Type::Vec3, ComponentType::R16u, qp.normalized, 8, Filter::None)
            } else {
                let pos_rscale = if qp.scale == 0.0 {
                    0.0
                } else {
                    1.0 / qp.scale * if qp.normalized { 32767.0 / 65535.0 } else { 1.0 }
                };

                let q = |v: f32| (if v >= 0.0 { 1 } else { -1 }) * mo::quantize_unorm(v.abs() * pos_rscale, qp.bits);

                let mut maxv = 0;

                for a in &stream.data {
                    for k in 0..3 {
                        maxv = maxv.max(mo::quantize_unorm(a[k].abs() * pos_rscale, qp.bits));
                    }
                }

                if maxv <= 127 && !qp.normalized {
                    for a in &stream.data {
                        bin.extend_from_slice(&[q(a[0]) as i8 as u8, q(a[1]) as i8 as u8, q(a[2]) as i8 as u8, 0]);
                    }

                    StreamFormat::new(Type::Vec3, ComponentType::R8, false, 4, Filter::None)
                } else {
                    for a in &stream.data {
                        push_i16s(bin, &[q(a[0]) as i16, q(a[1]) as i16, q(a[2]) as i16, 0]);
                    }

                    StreamFormat::new(Type::Vec3, ComponentType::R16, qp.normalized, 8, Filter::None)
                }
            }
        }
        AttributeType::Texcoord => {
            if !settings.quantize {
                return write_vertex_stream_raw(bin, stream, Type::Vec2, 2);
            }

            // expand the encoded range to ensure it covers [0..1) interval
            // this can slightly reduce precision but we should not need more precision inside 0..1, and this significantly improves compressed size when using encodeExpOne
            if settings.tex_float {
                return write_vertex_stream_float(bin, stream, Type::Vec2, 2, settings.compress && filters, qt.bits, M::Clamped);
            }

            let uv_rscale = [
                if qt.scale[0] == 0.0 { 0.0 } else { 1.0 / qt.scale[0] },
                if qt.scale[1] == 0.0 { 0.0 } else { 1.0 / qt.scale[1] },
            ];

            for a in &stream.data {
                let v = [
                    mo::quantize_unorm((a[0] - qt.offset[0]) * uv_rscale[0], qt.bits) as u16,
                    mo::quantize_unorm((a[1] - qt.offset[1]) * uv_rscale[1], qt.bits) as u16,
                ];
                push_u16s(bin, &v);
            }

            StreamFormat::new(Type::Vec2, ComponentType::R16u, qt.normalized, 4, Filter::None)
        }
        AttributeType::Normal => {
            if !settings.quantize {
                return write_vertex_stream_raw(bin, stream, Type::Vec3, 3);
            }

            // expand the encoded range to ensure it covers [0..1) interval
            // for unit length vectors, SharedComponent results in a slightly smaller encoding vs Clamped
            // for morph deltas, we could use SharedComponent but it may preserve noisy small inputs redundantly
            if settings.nrm_float {
                let mode = if settings.compressmore && stream.target == 0 {
                    M::SharedComponent
                } else {
                    M::Clamped
                };
                return write_vertex_stream_float(bin, stream, Type::Vec3, 3, settings.compress && filters, settings.nrm_bits, mode);
            }

            let oct = filters && settings.compressmore && stream.target == 0;
            let bits = settings.nrm_bits;

            let filter = if oct { Filter::Oct } else { Filter::None };

            let offset = bin.len();
            let stride = if bits > 8 { 8 } else { 4 };
            bin.resize(offset + stream.data.len() * stride, 0);

            let floats = attr_floats(&stream.data);
            if oct {
                let _ = mo::encode_filter_oct(&mut bin[offset..], stream.data.len(), stride, bits, floats);
            } else {
                encode_snorm(&mut bin[offset..], stream.data.len(), stride, bits, floats);
            }

            let component_type = if bits > 8 { ComponentType::R16 } else { ComponentType::R8 };
            StreamFormat::new(Type::Vec3, component_type, true, stride, filter)
        }
        AttributeType::Tangent => {
            if !settings.quantize {
                return write_vertex_stream_raw(bin, stream, Type::Vec4, 4);
            }

            let oct = filters && settings.compressmore && stream.target == 0;
            let bits = if settings.nrm_bits > 8 { 8 } else { settings.nrm_bits };

            let filter = if oct { Filter::Oct } else { Filter::None };

            let offset = bin.len();
            let stride = 4;
            bin.resize(offset + stream.data.len() * stride, 0);

            let floats = attr_floats(&stream.data);
            if oct {
                let _ = mo::encode_filter_oct(&mut bin[offset..], stream.data.len(), stride, bits, floats);
            } else {
                encode_snorm(&mut bin[offset..], stream.data.len(), stride, bits, floats);
            }

            let type_ = if stream.target == 0 { Type::Vec4 } else { Type::Vec3 };
            StreamFormat::new(type_, ComponentType::R8, true, 4, filter)
        }
        AttributeType::Color => {
            let col = filters && settings.compresskhr && settings.compressmore;
            let bits = settings.col_bits;

            let filter = if col { Filter::Color } else { Filter::None };

            let offset = bin.len();
            let stride = if bits > 8 { 8 } else { 4 };
            bin.resize(offset + stream.data.len() * stride, 0);

            let floats = attr_floats(&stream.data);
            if col {
                let _ = mo::encode_filter_color(&mut bin[offset..], stream.data.len(), stride, bits, floats);
            } else {
                encode_color(&mut bin[offset..], stream.data.len(), stride, bits, floats);
            }

            if bits > 8 {
                StreamFormat::new(Type::Vec4, ComponentType::R16u, true, 8, filter)
            } else {
                StreamFormat::new(Type::Vec4, ComponentType::R8u, true, 4, filter)
            }
        }
        AttributeType::Weights => {
            for a in &stream.data {
                let ws = a[0] + a[1] + a[2] + a[3];
                let wsi = if ws == 0.0 { 0.0 } else { 1.0 / ws };

                let mut v = [0, 1, 2, 3].map(|k| mo::quantize_unorm(a[k] * wsi, 8) as u8);

                if wsi != 0.0 {
                    renormalize_weights(&mut v);
                }

                bin.extend_from_slice(&v);
            }

            StreamFormat::new(Type::Vec4, ComponentType::R8u, true, 4, Filter::None)
        }
        AttributeType::Joints => {
            let mut maxj: u32 = 0;

            for a in &stream.data {
                for &f in a {
                    maxj = maxj.max(f2i32(f) as u32);
                }
            }

            if maxj <= 255 {
                for a in &stream.data {
                    bin.extend_from_slice(&a.map(f2u8));
                }

                StreamFormat::new(Type::Vec4, ComponentType::R8u, false, 4, Filter::None)
            } else {
                for a in &stream.data {
                    push_u16s(bin, &a.map(f2u16));
                }

                StreamFormat::new(Type::Vec4, ComponentType::R16u, false, 8, Filter::None)
            }
        }
        AttributeType::Custom => {
            // note: _custom is equivalent to _ID, as such the data contains scalar integers
            if !settings.compressmore || !filters {
                return write_vertex_stream_raw(bin, stream, Type::Scalar, 1);
            }

            let maxv = stream.data.iter().map(|a| f2i32(a[0]) as u32).max().unwrap_or(0);

            // exp encoding uses a signed mantissa with only 23 significant bits; input glTF encoding may encode indices losslessly up to 2^24
            if maxv >= (1 << 23) {
                return write_vertex_stream_raw(bin, stream, Type::Scalar, 1);
            }

            for a in &stream.data {
                // exp encoding of integers in [0..2^23-1] range is equivalent to the integer itself
                bin.extend_from_slice(&f2u32(a[0]).to_le_bytes());
            }

            StreamFormat::new(Type::Scalar, ComponentType::R32f, false, 4, Filter::Exp)
        }
        AttributeType::Invalid => write_vertex_stream_raw(bin, stream, Type::Vec4, 4),
    }
}

pub fn write_index_stream(bin: &mut Vec<u8>, stream: &[u32]) -> StreamFormat {
    let maxi = stream.iter().copied().max().unwrap_or(0);

    // save 16-bit indices if we can; note that we can't use restart index (65535)
    if maxi < 65535 {
        for &i in stream {
            bin.extend_from_slice(&(i as u16).to_le_bytes());
        }

        StreamFormat::new(Type::Scalar, ComponentType::R16u, false, 2, Filter::None)
    } else {
        for &i in stream {
            bin.extend_from_slice(&i.to_le_bytes());
        }

        StreamFormat::new(Type::Scalar, ComponentType::R32u, false, 4, Filter::None)
    }
}

pub fn write_time_stream(bin: &mut Vec<u8>, data: &[f32]) -> StreamFormat {
    push_f32s(bin, data);

    StreamFormat::new(Type::Scalar, ComponentType::R32f, false, 4, Filter::None)
}

pub fn write_keyframe_stream(
    bin: &mut Vec<u8>,
    type_: AnimationPathType,
    data: &[Attr],
    settings: &Settings,
    has_tangents: bool,
) -> StreamFormat {
    match type_ {
        AnimationPathType::Rotation => {
            let filter = if settings.compressmore && !has_tangents {
                Filter::Quat
            } else {
                Filter::None
            };

            let offset = bin.len();
            let stride = 8;
            bin.resize(offset + data.len() * stride, 0);

            let floats = attr_floats(data);
            if filter == Filter::Quat {
                let _ = mo::encode_filter_quat(&mut bin[offset..], data.len(), stride, settings.rot_bits, floats);
            } else {
                encode_snorm(&mut bin[offset..], data.len(), stride, 16, floats);
            }

            StreamFormat::new(Type::Vec4, ComponentType::R16, true, 8, filter)
        }
        AnimationPathType::Weights => {
            for a in data {
                bin.push(mo::quantize_unorm(a[0], 8) as u8);
            }

            StreamFormat::new(Type::Scalar, ComponentType::R8u, true, 1, Filter::None)
        }
        AnimationPathType::Translation | AnimationPathType::Scale => {
            let filter = if settings.compressmore { Filter::Exp } else { Filter::None };
            let bits = if type_ == AnimationPathType::Translation {
                settings.trn_bits
            } else {
                settings.scl_bits
            };

            if filter == Filter::Exp {
                append_filter_exp(bin, data, 3, bits, mo::EncodeExpMode::SharedVector);
            } else {
                for a in data {
                    push_f32s(bin, &a[..3]);
                }
            }

            StreamFormat::new(Type::Vec3, ComponentType::R32f, false, 12, filter)
        }
        AnimationPathType::Invalid => {
            for a in data {
                push_f32s(bin, a);
            }

            StreamFormat::new(Type::Vec4, ComponentType::R32f, false, 16, Filter::None)
        }
    }
}

pub fn compress_vertex_stream(bin: &mut Vec<u8>, data: &[u8], count: usize, stride: usize, level: i32) {
    let Ok(bound) = mo::encode_vertex_buffer_bound(count, stride) else {
        return;
    };

    let mut compressed = vec![0u8; bound];
    let size = mo::encode_vertex_buffer_level(&mut compressed, bound, data, count, stride, level, i32::from(level > 0)).unwrap_or(0);

    bin.extend_from_slice(&compressed[..size]);
}

fn bytes_to_u16(data: &[u8]) -> Vec<u16> {
    data.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect()
}

fn bytes_to_u32(data: &[u8]) -> Vec<u32> {
    data.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect()
}

pub fn compress_index_stream(bin: &mut Vec<u8>, data: &[u8], count: usize, stride: usize) {
    let Ok(bound) = mo::encode_index_buffer_bound(count, count) else {
        return;
    };

    let mut compressed = vec![0u8; bound];

    let size = if stride == 2 {
        mo::encode_index_buffer(&mut compressed, bound, &bytes_to_u16(data), count)
    } else {
        mo::encode_index_buffer(&mut compressed, bound, &bytes_to_u32(data), count)
    }
    .unwrap_or(0);

    bin.extend_from_slice(&compressed[..size]);
}

pub fn compress_index_sequence(bin: &mut Vec<u8>, data: &[u8], count: usize, stride: usize) {
    let bound = mo::encode_index_sequence_bound(count, count);

    let mut compressed = vec![0u8; bound];

    let size = if stride == 2 {
        mo::encode_index_sequence(&mut compressed, bound, &bytes_to_u16(data), count)
    } else {
        mo::encode_index_sequence(&mut compressed, bound, &bytes_to_u32(data), count)
    }
    .unwrap_or(0);

    bin.extend_from_slice(&compressed[..size]);
}
