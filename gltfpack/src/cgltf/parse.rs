//! `cgltf_parse_json_*`: glTF JSON → [`Data`], ported function by function.
//! Functions return the next token index or a negative `CGLTF_ERROR_*` code, like the C implementation.

use std::collections::HashSet;

use super::jsmn::{JsmnParser, JsmnTok, JsmnType, jsmn_parse};
use super::types::*;
use crate::cstd;

pub(crate) const CGLTF_ERROR_JSON: i32 = -1;
pub(crate) const CGLTF_ERROR_NOMEM: i32 = -2;
pub(crate) const CGLTF_ERROR_LEGACY: i32 = -3;

macro_rules! check_toktype {
    ($tok:expr, $type_:expr) => {
        if $tok.type_ != $type_ {
            return CGLTF_ERROR_JSON;
        }
    };
}

// checking size for 0 verifies that a value follows the key
macro_rules! check_key {
    ($tok:expr) => {
        if $tok.type_ != JsmnType::String || $tok.size == 0 {
            return CGLTF_ERROR_JSON;
        }
    };
}

macro_rules! check {
    ($i:expr) => {
        if $i < 0 {
            return $i;
        }
    };
}

/// `CGLTF_PTRINDEX`: JSON index + 1, wrapping like the C pointer arithmetic (0 = no reference).
pub(crate) fn ptrindex(index: i32) -> usize {
    (index as isize as usize).wrapping_add(1)
}

fn ptrindex_opt(index: i32) -> Option<usize> {
    match ptrindex(index) {
        0 => None,
        raw => Some(raw),
    }
}

struct Parser<'a> {
    tokens: &'a [JsmnTok],
    json: &'a [u8],
    /// Arrays already allocated (C checks `if (*out_array) return CGLTF_ERROR_JSON` to reject duplicate keys).
    allocated: HashSet<usize>,
}

impl Parser<'_> {
    /// Token access; out of range reads return an `UNDEFINED` token like the C sentinel.
    fn tok(&self, i: i32) -> JsmnTok {
        usize::try_from(i)
            .ok()
            .and_then(|i| self.tokens.get(i))
            .copied()
            .unwrap_or_default()
    }

    fn text(&self, tok: JsmnTok) -> &[u8] {
        let start = usize::try_from(tok.start).unwrap_or(0);
        let end = usize::try_from(tok.end).unwrap_or(0);
        self.json.get(start..end).unwrap_or(&[])
    }

    /// `cgltf_json_strcmp(...) == 0`
    fn streq(&self, i: i32, s: &str) -> bool {
        let tok = self.tok(i);
        tok.type_ == JsmnType::String && self.text(tok) == s.as_bytes()
    }

    /// Token text truncated to 127 bytes, like the `char tmp[128]` copies in C.
    fn number_text(&self, tok: JsmnTok) -> &[u8] {
        let text = self.text(tok);
        &text[..text.len().min(127)]
    }

    fn to_int(&self, i: i32) -> i32 {
        let tok = self.tok(i);
        if tok.type_ != JsmnType::Primitive {
            return CGLTF_ERROR_JSON;
        }
        cstd::atoi(self.number_text(tok))
    }

    fn to_size(&self, i: i32) -> CgltfSize {
        let tok = self.tok(i);
        if tok.type_ != JsmnType::Primitive {
            return 0;
        }
        let res = cstd::atoll(self.number_text(tok));
        if res < 0 { 0 } else { res as CgltfSize }
    }

    fn to_float(&self, i: i32) -> f32 {
        let tok = self.tok(i);
        if tok.type_ != JsmnType::Primitive {
            return CGLTF_ERROR_JSON as f32;
        }
        cstd::atof(self.number_text(tok)) as f32
    }

    fn to_bool(&self, i: i32) -> bool {
        self.text(self.tok(i)) == b"true"
    }

    fn skip(&self, mut i: i32) -> i32 {
        let mut end = i + 1;

        while i < end {
            let tok = self.tok(i);
            match tok.type_ {
                JsmnType::Object => end += tok.size * 2,
                JsmnType::Array => end += tok.size,
                JsmnType::Primitive | JsmnType::String => {}
                JsmnType::Undefined => return -1,
            }
            i += 1;
        }

        i
    }

    fn string_value(&self, tok: JsmnTok) -> String {
        String::from_utf8_lossy(self.text(tok)).into_owned()
    }

    fn float_array(&self, mut i: i32, out: &mut [f32], size: i32) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Array);
        if self.tok(i).size != size {
            return CGLTF_ERROR_JSON;
        }
        i += 1;
        for j in 0..size as usize {
            check_toktype!(self.tok(i), JsmnType::Primitive);
            if let Some(o) = out.get_mut(j) {
                *o = self.to_float(i);
            }
            i += 1;
        }
        i
    }

    fn string(&self, i: i32, out: &mut Option<String>) -> i32 {
        check_toktype!(self.tok(i), JsmnType::String);
        if out.is_some() {
            return CGLTF_ERROR_JSON;
        }
        *out = Some(self.string_value(self.tok(i)));
        i + 1
    }

    /// Marks `v` as allocated; false if it already was (duplicate key).
    /// Only ever pass fields of the parsed `Data` (stable addresses), never temporaries.
    fn claim<T>(&mut self, v: &Vec<T>) -> bool {
        self.allocated.insert(v as *const Vec<T> as usize)
    }

    fn array<T: Default + Clone>(&mut self, i: i32, out: &mut Vec<T>) -> i32 {
        let tok = self.tok(i);
        if tok.type_ != JsmnType::Array {
            return if tok.type_ == JsmnType::Object {
                CGLTF_ERROR_LEGACY
            } else {
                CGLTF_ERROR_JSON
            };
        }
        if !self.claim(out) {
            return CGLTF_ERROR_JSON;
        }
        *out = vec![T::default(); usize::try_from(tok.size).unwrap_or(0)];
        i + 1
    }

    /// `cgltf_parse_json_array` into a nullable array (`Option<Vec>` models the C NULL check directly).
    fn array_opt<T: Default + Clone>(&mut self, i: i32, out: &mut Option<Vec<T>>) -> i32 {
        let tok = self.tok(i);
        if tok.type_ != JsmnType::Array {
            return if tok.type_ == JsmnType::Object {
                CGLTF_ERROR_LEGACY
            } else {
                CGLTF_ERROR_JSON
            };
        }
        if out.is_some() {
            return CGLTF_ERROR_JSON;
        }
        *out = Some(vec![T::default(); usize::try_from(tok.size).unwrap_or(0)]);
        i + 1
    }

    /// Allocates `size` empty strings and parses them (shared tail of `cgltf_parse_json_string_array`).
    fn strings(&self, mut i: i32, size: i32) -> Result<(i32, Vec<String>), i32> {
        let mut strings = vec![None; usize::try_from(size).unwrap_or(0)];
        for s in strings.iter_mut() {
            i = self.string(i, s);
            if i < 0 {
                return Err(i);
            }
        }
        Ok((i, strings.into_iter().flatten().collect()))
    }

    fn string_array(&mut self, i: i32, out: &mut Vec<String>) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Array);
        if !self.claim(out) {
            return CGLTF_ERROR_JSON;
        }
        match self.strings(i + 1, self.tok(i).size) {
            Ok((i, strings)) => {
                *out = strings;
                i
            }
            Err(e) => e,
        }
    }

    fn string_array_opt(&mut self, i: i32, out: &mut Option<Vec<String>>) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Array);
        if out.is_some() {
            return CGLTF_ERROR_JSON;
        }
        *out = Some(Vec::new());
        match self.strings(i + 1, self.tok(i).size) {
            Ok((i, strings)) => {
                *out = Some(strings);
                i
            }
            Err(e) => e,
        }
    }

    fn attribute_list(&mut self, mut i: i32, out: &mut Vec<Attribute>) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        if !self.claim(out) {
            return CGLTF_ERROR_JSON;
        }

        let count = usize::try_from(self.tok(i).size).unwrap_or(0);
        *out = vec![Attribute::default(); count];
        i += 1;

        for attr in out.iter_mut() {
            check_key!(self.tok(i));

            i = self.string(i, &mut attr.name);
            if i < 0 {
                return CGLTF_ERROR_JSON;
            }

            let (type_, index) = parse_attribute_type(attr.name.as_deref().unwrap_or(""), attr.index);
            attr.type_ = type_;
            attr.index = index;

            attr.data = ptrindex(self.to_int(i));
            i += 1;
        }

        i
    }

    fn extras(&self, i: i32, out: &mut Extras) -> i32 {
        if out.data.is_some() {
            return CGLTF_ERROR_JSON;
        }

        let tok = self.tok(i);

        // fill deprecated fields for now, this will be removed in the future
        out.start_offset = tok.start as CgltfSize;
        out.end_offset = tok.end as CgltfSize;

        out.data = Some(self.string_value(tok));

        self.skip(i)
    }

    fn unprocessed_extension(&self, mut i: i32, out: &mut Extension) -> i32 {
        check_toktype!(self.tok(i), JsmnType::String);
        check_toktype!(self.tok(i + 1), JsmnType::Object);
        if out.name.is_some() {
            return CGLTF_ERROR_JSON;
        }

        out.name = Some(self.string_value(self.tok(i)));
        i += 1;

        out.data = Some(self.string_value(self.tok(i)));

        self.skip(i)
    }

    /// Starts an `extensions` object (C: the `CGLTF_CHECK_TOKTYPE` + duplicate check + allocation preamble).
    /// `i` points at the `extensions` key; returns the index of the first extension key and the extension count.
    fn begin_extensions(&mut self, mut i: i32, out: &Vec<Extension>) -> Result<(i32, i32), i32> {
        i += 1;

        if self.tok(i).type_ != JsmnType::Object {
            return Err(CGLTF_ERROR_JSON);
        }
        if !self.claim(out) {
            return Err(CGLTF_ERROR_JSON);
        }

        let size = self.tok(i).size;
        Ok((i + 1, size))
    }

    fn unprocessed_extensions(&mut self, i: i32, out: &mut Vec<Extension>) -> i32 {
        let (mut i, size) = match self.begin_extensions(i, out) {
            Ok(v) => v,
            Err(e) => return e,
        };

        for _ in 0..size {
            check_key!(self.tok(i));

            let mut extension = Extension::default();
            i = self.unprocessed_extension(i, &mut extension);
            out.push(extension);
            check!(i);
        }
        i
    }

    /// Parses an unprocessed extension entry inside a hand-written `extensions` loop.
    fn push_unprocessed_extension(&self, i: i32, out: &mut Vec<Extension>) -> i32 {
        let mut extension = Extension::default();
        let i = self.unprocessed_extension(i, &mut extension);
        out.push(extension);
        i
    }

    fn draco_mesh_compression(&mut self, mut i: i32, out: &mut DracoMeshCompression) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "attributes") {
                i = self.attribute_list(i + 1, &mut out.attributes);
            } else if self.streq(i, "bufferView") {
                i += 1;
                out.buffer_view = ptrindex(self.to_int(i));
                i += 1;
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn mesh_gpu_instancing(&mut self, mut i: i32, out: &mut MeshGpuInstancing) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "attributes") {
                i = self.attribute_list(i + 1, &mut out.attributes);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn material_mapping_data(&self, mut i: i32, mut out: Option<&mut [MaterialMapping]>, offset: &mut CgltfSize) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Array);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_toktype!(self.tok(i), JsmnType::Object);

            let obj_size = self.tok(i).size;
            i += 1;

            let mut material = -1;
            let mut variants_tok = -1;
            let mut extras_tok = -1;

            for _ in 0..obj_size {
                check_key!(self.tok(i));

                if self.streq(i, "material") {
                    i += 1;
                    material = self.to_int(i);
                    i += 1;
                } else if self.streq(i, "variants") {
                    variants_tok = i + 1;
                    check_toktype!(self.tok(variants_tok), JsmnType::Array);

                    i = self.skip(i + 1);
                } else if self.streq(i, "extras") {
                    extras_tok = i + 1;
                    i = self.skip(extras_tok);
                } else {
                    i = self.skip(i + 1);
                }

                check!(i);
            }

            if material < 0 || variants_tok < 0 {
                return CGLTF_ERROR_JSON;
            }

            let variants_size = self.tok(variants_tok).size;

            if let Some(mappings) = out.as_deref_mut() {
                for k in 0..variants_size {
                    let variant = self.to_int(variants_tok + 1 + k);
                    if variant < 0 {
                        return variant;
                    }

                    let Some(mapping) = mappings.get_mut(*offset) else {
                        return CGLTF_ERROR_JSON;
                    };

                    mapping.material = ptrindex(material);
                    mapping.variant = variant as CgltfSize;

                    if extras_tok >= 0 {
                        let e = self.extras(extras_tok, &mut mapping.extras);
                        if e < 0 {
                            return e;
                        }
                    }

                    *offset += 1;
                }
            } else {
                *offset += usize::try_from(variants_size).unwrap_or(0);
            }
        }

        i
    }

    fn material_mappings(&mut self, mut i: i32, out_prim: &mut Primitive) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "mappings") {
                if !self.claim(&out_prim.mappings) {
                    return CGLTF_ERROR_JSON;
                }

                let mut mappings_offset = 0;
                let k = self.material_mapping_data(i + 1, None, &mut mappings_offset);
                if k < 0 {
                    return k;
                }

                out_prim.mappings = vec![MaterialMapping::default(); mappings_offset];

                mappings_offset = 0;
                i = self.material_mapping_data(i + 1, Some(&mut out_prim.mappings), &mut mappings_offset);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn primitive(&mut self, mut i: i32, out_prim: &mut Primitive) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        out_prim.type_ = PrimitiveType::Triangles;

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "mode") {
                i += 1;
                out_prim.type_ = json_to_primitive_type(self.to_int(i));
                i += 1;
            } else if self.streq(i, "indices") {
                i += 1;
                out_prim.indices = ptrindex_opt(self.to_int(i));
                i += 1;
            } else if self.streq(i, "material") {
                i += 1;
                out_prim.material = ptrindex_opt(self.to_int(i));
                i += 1;
            } else if self.streq(i, "attributes") {
                i = self.attribute_list(i + 1, &mut out_prim.attributes);
            } else if self.streq(i, "targets") {
                i = self.array(i + 1, &mut out_prim.targets);
                check!(i);

                for target in out_prim.targets.iter_mut() {
                    i = self.attribute_list(i, &mut target.attributes);
                    check!(i);
                }
            } else if self.streq(i, "extras") {
                i = self.extras(i + 1, &mut out_prim.extras);
            } else if self.streq(i, "extensions") {
                let (start, extensions_size) = match self.begin_extensions(i, &out_prim.extensions) {
                    Ok(v) => v,
                    Err(e) => return e,
                };
                i = start;

                for _ in 0..extensions_size {
                    check_key!(self.tok(i));

                    if self.streq(i, "KHR_draco_mesh_compression") {
                        out_prim.has_draco_mesh_compression = true;
                        i = self.draco_mesh_compression(i + 1, &mut out_prim.draco_mesh_compression);
                    } else if self.streq(i, "KHR_materials_variants") {
                        i = self.material_mappings(i + 1, out_prim);
                    } else {
                        i = self.push_unprocessed_extension(i, &mut out_prim.extensions);
                    }

                    check!(i);
                }
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn mesh(&mut self, mut i: i32, out_mesh: &mut Mesh) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "name") {
                i = self.string(i + 1, &mut out_mesh.name);
            } else if self.streq(i, "primitives") {
                i = self.array(i + 1, &mut out_mesh.primitives);
                check!(i);

                for prim in out_mesh.primitives.iter_mut() {
                    i = self.primitive(i, prim);
                    check!(i);
                }
            } else if self.streq(i, "weights") {
                i = self.array_opt(i + 1, &mut out_mesh.weights);
                check!(i);

                let weights = out_mesh.weights.get_or_insert_default();
                let count = weights.len() as i32;
                i = self.float_array(i - 1, weights, count);
            } else if self.streq(i, "extras") {
                i += 1;

                let tok = self.tok(i);
                out_mesh.extras.start_offset = tok.start as CgltfSize;
                out_mesh.extras.end_offset = tok.end as CgltfSize;

                if tok.type_ == JsmnType::Object {
                    let extras_size = tok.size;
                    i += 1;

                    for _ in 0..extras_size {
                        check_key!(self.tok(i));

                        if self.streq(i, "targetNames") && self.tok(i + 1).type_ == JsmnType::Array {
                            i = self.string_array_opt(i + 1, &mut out_mesh.target_names);
                        } else {
                            i = self.skip(i + 1);
                        }

                        check!(i);
                    }
                } else {
                    i = self.skip(i);
                }
            } else if self.streq(i, "extensions") {
                i = self.unprocessed_extensions(i, &mut out_mesh.extensions);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn accessor_sparse(&self, mut i: i32, out_sparse: &mut AccessorSparse) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "count") {
                i += 1;
                out_sparse.count = self.to_size(i);
                i += 1;
            } else if self.streq(i, "indices") {
                i += 1;
                check_toktype!(self.tok(i), JsmnType::Object);

                let indices_size = self.tok(i).size;
                i += 1;

                for _ in 0..indices_size {
                    check_key!(self.tok(i));

                    if self.streq(i, "bufferView") {
                        i += 1;
                        out_sparse.indices_buffer_view = ptrindex(self.to_int(i));
                        i += 1;
                    } else if self.streq(i, "byteOffset") {
                        i += 1;
                        out_sparse.indices_byte_offset = self.to_size(i);
                        i += 1;
                    } else if self.streq(i, "componentType") {
                        i += 1;
                        out_sparse.indices_component_type = json_to_component_type(self.to_int(i));
                        i += 1;
                    } else {
                        i = self.skip(i + 1);
                    }

                    check!(i);
                }
            } else if self.streq(i, "values") {
                i += 1;
                check_toktype!(self.tok(i), JsmnType::Object);

                let values_size = self.tok(i).size;
                i += 1;

                for _ in 0..values_size {
                    check_key!(self.tok(i));

                    if self.streq(i, "bufferView") {
                        i += 1;
                        out_sparse.values_buffer_view = ptrindex(self.to_int(i));
                        i += 1;
                    } else if self.streq(i, "byteOffset") {
                        i += 1;
                        out_sparse.values_byte_offset = self.to_size(i);
                        i += 1;
                    } else {
                        i = self.skip(i + 1);
                    }

                    check!(i);
                }
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn accessor(&mut self, mut i: i32, out: &mut Accessor) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "name") {
                i = self.string(i + 1, &mut out.name);
            } else if self.streq(i, "bufferView") {
                i += 1;
                out.buffer_view = ptrindex_opt(self.to_int(i));
                i += 1;
            } else if self.streq(i, "byteOffset") {
                i += 1;
                out.offset = self.to_size(i);
                i += 1;
            } else if self.streq(i, "componentType") {
                i += 1;
                out.component_type = json_to_component_type(self.to_int(i));
                i += 1;
            } else if self.streq(i, "normalized") {
                i += 1;
                out.normalized = self.to_bool(i);
                i += 1;
            } else if self.streq(i, "count") {
                i += 1;
                out.count = self.to_size(i);
                i += 1;
            } else if self.streq(i, "type") {
                i += 1;
                if self.streq(i, "SCALAR") {
                    out.type_ = Type::Scalar;
                } else if self.streq(i, "VEC2") {
                    out.type_ = Type::Vec2;
                } else if self.streq(i, "VEC3") {
                    out.type_ = Type::Vec3;
                } else if self.streq(i, "VEC4") {
                    out.type_ = Type::Vec4;
                } else if self.streq(i, "MAT2") {
                    out.type_ = Type::Mat2;
                } else if self.streq(i, "MAT3") {
                    out.type_ = Type::Mat3;
                } else if self.streq(i, "MAT4") {
                    out.type_ = Type::Mat4;
                }
                i += 1;
            } else if self.streq(i, "min") {
                i += 1;
                out.has_min = true;
                // note: we can't parse the precise number of elements since type may not have been computed yet
                let min_size = self.tok(i).size.min(16);
                i = self.float_array(i, &mut out.min, min_size);
            } else if self.streq(i, "max") {
                i += 1;
                out.has_max = true;
                // note: we can't parse the precise number of elements since type may not have been computed yet
                let max_size = self.tok(i).size.min(16);
                i = self.float_array(i, &mut out.max, max_size);
            } else if self.streq(i, "sparse") {
                out.is_sparse = true;
                i = self.accessor_sparse(i + 1, &mut out.sparse);
            } else if self.streq(i, "extras") {
                i = self.extras(i + 1, &mut out.extras);
            } else if self.streq(i, "extensions") {
                i = self.unprocessed_extensions(i, &mut out.extensions);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn texture_transform(&self, mut i: i32, out: &mut TextureTransform) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "offset") {
                i = self.float_array(i + 1, &mut out.offset, 2);
            } else if self.streq(i, "rotation") {
                i += 1;
                out.rotation = self.to_float(i);
                i += 1;
            } else if self.streq(i, "scale") {
                i = self.float_array(i + 1, &mut out.scale, 2);
            } else if self.streq(i, "texCoord") {
                i += 1;
                out.has_texcoord = true;
                out.texcoord = self.to_int(i);
                i += 1;
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn texture_view(&self, mut i: i32, out: &mut TextureView) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        out.scale = 1.0;
        out.transform.scale = [1.0; 2];

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "index") {
                i += 1;
                out.texture = ptrindex_opt(self.to_int(i));
                i += 1;
            } else if self.streq(i, "texCoord") {
                i += 1;
                out.texcoord = self.to_int(i);
                i += 1;
            } else if self.streq(i, "scale") || self.streq(i, "strength") {
                i += 1;
                out.scale = self.to_float(i);
                i += 1;
            } else if self.streq(i, "extensions") {
                i += 1;

                check_toktype!(self.tok(i), JsmnType::Object);
                let extensions_size = self.tok(i).size;

                i += 1;

                for _ in 0..extensions_size {
                    check_key!(self.tok(i));

                    if self.streq(i, "KHR_texture_transform") {
                        out.has_transform = true;
                        i = self.texture_transform(i + 1, &mut out.transform);
                    } else {
                        i = self.skip(i + 1);
                    }

                    check!(i);
                }
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn pbr_metallic_roughness(&self, mut i: i32, out: &mut PbrMetallicRoughness) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "metallicFactor") {
                i += 1;
                out.metallic_factor = self.to_float(i);
                i += 1;
            } else if self.streq(i, "roughnessFactor") {
                i += 1;
                out.roughness_factor = self.to_float(i);
                i += 1;
            } else if self.streq(i, "baseColorFactor") {
                i = self.float_array(i + 1, &mut out.base_color_factor, 4);
            } else if self.streq(i, "baseColorTexture") {
                i = self.texture_view(i + 1, &mut out.base_color_texture);
            } else if self.streq(i, "metallicRoughnessTexture") {
                i = self.texture_view(i + 1, &mut out.metallic_roughness_texture);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn pbr_specular_glossiness(&self, mut i: i32, out: &mut PbrSpecularGlossiness) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);
        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "diffuseFactor") {
                i = self.float_array(i + 1, &mut out.diffuse_factor, 4);
            } else if self.streq(i, "specularFactor") {
                i = self.float_array(i + 1, &mut out.specular_factor, 3);
            } else if self.streq(i, "glossinessFactor") {
                i += 1;
                out.glossiness_factor = self.to_float(i);
                i += 1;
            } else if self.streq(i, "diffuseTexture") {
                i = self.texture_view(i + 1, &mut out.diffuse_texture);
            } else if self.streq(i, "specularGlossinessTexture") {
                i = self.texture_view(i + 1, &mut out.specular_glossiness_texture);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn clearcoat(&self, mut i: i32, out: &mut Clearcoat) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);
        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "clearcoatFactor") {
                i += 1;
                out.clearcoat_factor = self.to_float(i);
                i += 1;
            } else if self.streq(i, "clearcoatRoughnessFactor") {
                i += 1;
                out.clearcoat_roughness_factor = self.to_float(i);
                i += 1;
            } else if self.streq(i, "clearcoatTexture") {
                i = self.texture_view(i + 1, &mut out.clearcoat_texture);
            } else if self.streq(i, "clearcoatRoughnessTexture") {
                i = self.texture_view(i + 1, &mut out.clearcoat_roughness_texture);
            } else if self.streq(i, "clearcoatNormalTexture") {
                i = self.texture_view(i + 1, &mut out.clearcoat_normal_texture);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn ior(&self, mut i: i32, out: &mut Ior) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);
        let size = self.tok(i).size;
        i += 1;

        // Default values
        out.ior = 1.5;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "ior") {
                i += 1;
                out.ior = self.to_float(i);
                i += 1;
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn specular(&self, mut i: i32, out: &mut Specular) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);
        let size = self.tok(i).size;
        i += 1;

        // Default values
        out.specular_factor = 1.0;
        out.specular_color_factor = [1.0; 3];

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "specularFactor") {
                i += 1;
                out.specular_factor = self.to_float(i);
                i += 1;
            } else if self.streq(i, "specularColorFactor") {
                i = self.float_array(i + 1, &mut out.specular_color_factor, 3);
            } else if self.streq(i, "specularTexture") {
                i = self.texture_view(i + 1, &mut out.specular_texture);
            } else if self.streq(i, "specularColorTexture") {
                i = self.texture_view(i + 1, &mut out.specular_color_texture);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn transmission(&self, mut i: i32, out: &mut Transmission) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);
        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "transmissionFactor") {
                i += 1;
                out.transmission_factor = self.to_float(i);
                i += 1;
            } else if self.streq(i, "transmissionTexture") {
                i = self.texture_view(i + 1, &mut out.transmission_texture);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn volume(&self, mut i: i32, out: &mut Volume) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);
        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "thicknessFactor") {
                i += 1;
                out.thickness_factor = self.to_float(i);
                i += 1;
            } else if self.streq(i, "thicknessTexture") {
                i = self.texture_view(i + 1, &mut out.thickness_texture);
            } else if self.streq(i, "attenuationColor") {
                i = self.float_array(i + 1, &mut out.attenuation_color, 3);
            } else if self.streq(i, "attenuationDistance") {
                i += 1;
                out.attenuation_distance = self.to_float(i);
                i += 1;
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn sheen(&self, mut i: i32, out: &mut Sheen) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);
        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "sheenColorFactor") {
                i = self.float_array(i + 1, &mut out.sheen_color_factor, 3);
            } else if self.streq(i, "sheenColorTexture") {
                i = self.texture_view(i + 1, &mut out.sheen_color_texture);
            } else if self.streq(i, "sheenRoughnessFactor") {
                i += 1;
                out.sheen_roughness_factor = self.to_float(i);
                i += 1;
            } else if self.streq(i, "sheenRoughnessTexture") {
                i = self.texture_view(i + 1, &mut out.sheen_roughness_texture);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn emissive_strength(&self, mut i: i32, out: &mut EmissiveStrength) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);
        let size = self.tok(i).size;
        i += 1;

        // Default
        out.emissive_strength = 1.0;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "emissiveStrength") {
                i += 1;
                out.emissive_strength = self.to_float(i);
                i += 1;
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn iridescence(&self, mut i: i32, out: &mut Iridescence) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);
        let size = self.tok(i).size;
        i += 1;

        // Default
        out.iridescence_ior = 1.3;
        out.iridescence_thickness_min = 100.0;
        out.iridescence_thickness_max = 400.0;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "iridescenceFactor") {
                i += 1;
                out.iridescence_factor = self.to_float(i);
                i += 1;
            } else if self.streq(i, "iridescenceTexture") {
                i = self.texture_view(i + 1, &mut out.iridescence_texture);
            } else if self.streq(i, "iridescenceIor") {
                i += 1;
                out.iridescence_ior = self.to_float(i);
                i += 1;
            } else if self.streq(i, "iridescenceThicknessMinimum") {
                i += 1;
                out.iridescence_thickness_min = self.to_float(i);
                i += 1;
            } else if self.streq(i, "iridescenceThicknessMaximum") {
                i += 1;
                out.iridescence_thickness_max = self.to_float(i);
                i += 1;
            } else if self.streq(i, "iridescenceThicknessTexture") {
                i = self.texture_view(i + 1, &mut out.iridescence_thickness_texture);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn diffuse_transmission(&self, mut i: i32, out: &mut DiffuseTransmission) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);
        let size = self.tok(i).size;
        i += 1;

        // Defaults
        out.diffuse_transmission_color_factor = [1.0; 3];
        out.diffuse_transmission_factor = 0.0;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "diffuseTransmissionFactor") {
                i += 1;
                out.diffuse_transmission_factor = self.to_float(i);
                i += 1;
            } else if self.streq(i, "diffuseTransmissionTexture") {
                i = self.texture_view(i + 1, &mut out.diffuse_transmission_texture);
            } else if self.streq(i, "diffuseTransmissionColorFactor") {
                i = self.float_array(i + 1, &mut out.diffuse_transmission_color_factor, 3);
            } else if self.streq(i, "diffuseTransmissionColorTexture") {
                i = self.texture_view(i + 1, &mut out.diffuse_transmission_color_texture);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn anisotropy(&self, mut i: i32, out: &mut Anisotropy) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);
        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "anisotropyStrength") {
                i += 1;
                out.anisotropy_strength = self.to_float(i);
                i += 1;
            } else if self.streq(i, "anisotropyRotation") {
                i += 1;
                out.anisotropy_rotation = self.to_float(i);
                i += 1;
            } else if self.streq(i, "anisotropyTexture") {
                i = self.texture_view(i + 1, &mut out.anisotropy_texture);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn dispersion(&self, mut i: i32, out: &mut Dispersion) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);
        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "dispersion") {
                i += 1;
                out.dispersion = self.to_float(i);
                i += 1;
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn image(&mut self, mut i: i32, out: &mut Image) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "uri") {
                i = self.string(i + 1, &mut out.uri);
            } else if self.streq(i, "bufferView") {
                i += 1;
                out.buffer_view = ptrindex_opt(self.to_int(i));
                i += 1;
            } else if self.streq(i, "mimeType") {
                i = self.string(i + 1, &mut out.mime_type);
            } else if self.streq(i, "name") {
                i = self.string(i + 1, &mut out.name);
            } else if self.streq(i, "extras") {
                i = self.extras(i + 1, &mut out.extras);
            } else if self.streq(i, "extensions") {
                i = self.unprocessed_extensions(i, &mut out.extensions);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn sampler(&mut self, mut i: i32, out: &mut Sampler) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        out.wrap_s = WRAP_MODE_REPEAT;
        out.wrap_t = WRAP_MODE_REPEAT;

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "name") {
                i = self.string(i + 1, &mut out.name);
            } else if self.streq(i, "magFilter") {
                i += 1;
                out.mag_filter = self.to_int(i);
                i += 1;
            } else if self.streq(i, "minFilter") {
                i += 1;
                out.min_filter = self.to_int(i);
                i += 1;
            } else if self.streq(i, "wrapS") {
                i += 1;
                out.wrap_s = self.to_int(i);
                i += 1;
            } else if self.streq(i, "wrapT") {
                i += 1;
                out.wrap_t = self.to_int(i);
                i += 1;
            } else if self.streq(i, "extras") {
                i = self.extras(i + 1, &mut out.extras);
            } else if self.streq(i, "extensions") {
                i = self.unprocessed_extensions(i, &mut out.extensions);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    /// `KHR_texture_basisu`/`EXT_texture_webp` body: `{ "source": N, ... }`.
    fn texture_source_extension(&self, mut i: i32, out: &mut Option<usize>) -> i32 {
        i += 1;
        check_toktype!(self.tok(i), JsmnType::Object);
        let num_properties = self.tok(i).size;
        i += 1;

        for _ in 0..num_properties {
            check_key!(self.tok(i));

            if self.streq(i, "source") {
                i += 1;
                *out = ptrindex_opt(self.to_int(i));
                i += 1;
            } else {
                i = self.skip(i + 1);
            }
            check!(i);
        }

        i
    }

    fn texture(&mut self, mut i: i32, out: &mut Texture) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "name") {
                i = self.string(i + 1, &mut out.name);
            } else if self.streq(i, "sampler") {
                i += 1;
                out.sampler = ptrindex_opt(self.to_int(i));
                i += 1;
            } else if self.streq(i, "source") {
                i += 1;
                out.image = ptrindex_opt(self.to_int(i));
                i += 1;
            } else if self.streq(i, "extras") {
                i = self.extras(i + 1, &mut out.extras);
            } else if self.streq(i, "extensions") {
                let (start, extensions_size) = match self.begin_extensions(i, &out.extensions) {
                    Ok(v) => v,
                    Err(e) => return e,
                };
                i = start;

                for _ in 0..extensions_size {
                    check_key!(self.tok(i));

                    if self.streq(i, "KHR_texture_basisu") {
                        out.has_basisu = true;
                        i = self.texture_source_extension(i, &mut out.basisu_image);
                    } else if self.streq(i, "EXT_texture_webp") {
                        out.has_webp = true;
                        i = self.texture_source_extension(i, &mut out.webp_image);
                    } else {
                        i = self.push_unprocessed_extension(i, &mut out.extensions);
                    }

                    check!(i);
                }
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn material(&mut self, mut i: i32, out: &mut Material) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        out.pbr_metallic_roughness.base_color_factor = [1.0; 4];
        out.pbr_metallic_roughness.metallic_factor = 1.0;
        out.pbr_metallic_roughness.roughness_factor = 1.0;

        out.pbr_specular_glossiness.diffuse_factor = [1.0; 4];
        out.pbr_specular_glossiness.specular_factor = [1.0; 3];
        out.pbr_specular_glossiness.glossiness_factor = 1.0;

        out.volume.attenuation_color = [1.0; 3];
        out.volume.attenuation_distance = f32::MAX;

        out.alpha_cutoff = 0.5;

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "name") {
                i = self.string(i + 1, &mut out.name);
            } else if self.streq(i, "pbrMetallicRoughness") {
                out.has_pbr_metallic_roughness = true;
                i = self.pbr_metallic_roughness(i + 1, &mut out.pbr_metallic_roughness);
            } else if self.streq(i, "emissiveFactor") {
                i = self.float_array(i + 1, &mut out.emissive_factor, 3);
            } else if self.streq(i, "normalTexture") {
                i = self.texture_view(i + 1, &mut out.normal_texture);
            } else if self.streq(i, "occlusionTexture") {
                i = self.texture_view(i + 1, &mut out.occlusion_texture);
            } else if self.streq(i, "emissiveTexture") {
                i = self.texture_view(i + 1, &mut out.emissive_texture);
            } else if self.streq(i, "alphaMode") {
                i += 1;
                if self.streq(i, "OPAQUE") {
                    out.alpha_mode = AlphaMode::Opaque;
                } else if self.streq(i, "MASK") {
                    out.alpha_mode = AlphaMode::Mask;
                } else if self.streq(i, "BLEND") {
                    out.alpha_mode = AlphaMode::Blend;
                }
                i += 1;
            } else if self.streq(i, "alphaCutoff") {
                i += 1;
                out.alpha_cutoff = self.to_float(i);
                i += 1;
            } else if self.streq(i, "doubleSided") {
                i += 1;
                out.double_sided = self.to_bool(i);
                i += 1;
            } else if self.streq(i, "extras") {
                i = self.extras(i + 1, &mut out.extras);
            } else if self.streq(i, "extensions") {
                let (start, extensions_size) = match self.begin_extensions(i, &out.extensions) {
                    Ok(v) => v,
                    Err(e) => return e,
                };
                i = start;

                for _ in 0..extensions_size {
                    check_key!(self.tok(i));

                    if self.streq(i, "KHR_materials_pbrSpecularGlossiness") {
                        out.has_pbr_specular_glossiness = true;
                        i = self.pbr_specular_glossiness(i + 1, &mut out.pbr_specular_glossiness);
                    } else if self.streq(i, "KHR_materials_unlit") {
                        out.unlit = true;
                        i = self.skip(i + 1);
                    } else if self.streq(i, "KHR_materials_clearcoat") {
                        out.has_clearcoat = true;
                        i = self.clearcoat(i + 1, &mut out.clearcoat);
                    } else if self.streq(i, "KHR_materials_ior") {
                        out.has_ior = true;
                        i = self.ior(i + 1, &mut out.ior);
                    } else if self.streq(i, "KHR_materials_specular") {
                        out.has_specular = true;
                        i = self.specular(i + 1, &mut out.specular);
                    } else if self.streq(i, "KHR_materials_transmission") {
                        out.has_transmission = true;
                        i = self.transmission(i + 1, &mut out.transmission);
                    } else if self.streq(i, "KHR_materials_volume") {
                        out.has_volume = true;
                        i = self.volume(i + 1, &mut out.volume);
                    } else if self.streq(i, "KHR_materials_sheen") {
                        out.has_sheen = true;
                        i = self.sheen(i + 1, &mut out.sheen);
                    } else if self.streq(i, "KHR_materials_emissive_strength") {
                        out.has_emissive_strength = true;
                        i = self.emissive_strength(i + 1, &mut out.emissive_strength);
                    } else if self.streq(i, "KHR_materials_iridescence") {
                        out.has_iridescence = true;
                        i = self.iridescence(i + 1, &mut out.iridescence);
                    } else if self.streq(i, "KHR_materials_diffuse_transmission") {
                        out.has_diffuse_transmission = true;
                        i = self.diffuse_transmission(i + 1, &mut out.diffuse_transmission);
                    } else if self.streq(i, "KHR_materials_anisotropy") {
                        out.has_anisotropy = true;
                        i = self.anisotropy(i + 1, &mut out.anisotropy);
                    } else if self.streq(i, "KHR_materials_dispersion") {
                        out.has_dispersion = true;
                        i = self.dispersion(i + 1, &mut out.dispersion);
                    } else {
                        i = self.push_unprocessed_extension(i, &mut out.extensions);
                    }

                    check!(i);
                }
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn meshopt_compression(&self, mut i: i32, out: &mut MeshoptCompression) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "buffer") {
                i += 1;
                out.buffer = ptrindex(self.to_int(i));
                i += 1;
            } else if self.streq(i, "byteOffset") {
                i += 1;
                out.offset = self.to_size(i);
                i += 1;
            } else if self.streq(i, "byteLength") {
                i += 1;
                out.size = self.to_size(i);
                i += 1;
            } else if self.streq(i, "byteStride") {
                i += 1;
                out.stride = self.to_size(i);
                i += 1;
            } else if self.streq(i, "count") {
                i += 1;
                out.count = self.to_size(i);
                i += 1;
            } else if self.streq(i, "mode") {
                i += 1;
                if self.streq(i, "ATTRIBUTES") {
                    out.mode = MeshoptCompressionMode::Attributes;
                } else if self.streq(i, "TRIANGLES") {
                    out.mode = MeshoptCompressionMode::Triangles;
                } else if self.streq(i, "INDICES") {
                    out.mode = MeshoptCompressionMode::Indices;
                }
                i += 1;
            } else if self.streq(i, "filter") {
                i += 1;
                if self.streq(i, "NONE") {
                    out.filter = MeshoptCompressionFilter::None;
                } else if self.streq(i, "OCTAHEDRAL") {
                    out.filter = MeshoptCompressionFilter::Octahedral;
                } else if self.streq(i, "QUATERNION") {
                    out.filter = MeshoptCompressionFilter::Quaternion;
                } else if self.streq(i, "EXPONENTIAL") {
                    out.filter = MeshoptCompressionFilter::Exponential;
                } else if self.streq(i, "COLOR") {
                    out.filter = MeshoptCompressionFilter::Color;
                }
                i += 1;
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn buffer_view(&mut self, mut i: i32, out: &mut BufferView) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "name") {
                i = self.string(i + 1, &mut out.name);
            } else if self.streq(i, "buffer") {
                i += 1;
                out.buffer = ptrindex(self.to_int(i));
                i += 1;
            } else if self.streq(i, "byteOffset") {
                i += 1;
                out.offset = self.to_size(i);
                i += 1;
            } else if self.streq(i, "byteLength") {
                i += 1;
                out.size = self.to_size(i);
                i += 1;
            } else if self.streq(i, "byteStride") {
                i += 1;
                out.stride = self.to_size(i);
                i += 1;
            } else if self.streq(i, "target") {
                i += 1;
                out.type_ = match self.to_int(i) {
                    34962 => BufferViewType::Vertices,
                    34963 => BufferViewType::Indices,
                    _ => BufferViewType::Invalid,
                };
                i += 1;
            } else if self.streq(i, "extras") {
                i = self.extras(i + 1, &mut out.extras);
            } else if self.streq(i, "extensions") {
                let (start, extensions_size) = match self.begin_extensions(i, &out.extensions) {
                    Ok(v) => v,
                    Err(e) => return e,
                };
                i = start;

                for _ in 0..extensions_size {
                    check_key!(self.tok(i));

                    if self.streq(i, "EXT_meshopt_compression") {
                        out.has_meshopt_compression = true;
                        i = self.meshopt_compression(i + 1, &mut out.meshopt_compression);
                    } else if self.streq(i, "KHR_meshopt_compression") {
                        out.has_meshopt_compression = true;
                        out.meshopt_compression.is_khr = true;
                        i = self.meshopt_compression(i + 1, &mut out.meshopt_compression);
                    } else {
                        i = self.push_unprocessed_extension(i, &mut out.extensions);
                    }

                    check!(i);
                }
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn buffer(&mut self, mut i: i32, out: &mut Buffer) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "name") {
                i = self.string(i + 1, &mut out.name);
            } else if self.streq(i, "byteLength") {
                i += 1;
                out.size = self.to_size(i);
                i += 1;
            } else if self.streq(i, "uri") {
                i = self.string(i + 1, &mut out.uri);
            } else if self.streq(i, "extras") {
                i = self.extras(i + 1, &mut out.extras);
            } else if self.streq(i, "extensions") {
                i = self.unprocessed_extensions(i, &mut out.extensions);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    /// Array of node references (`joints`, `children`, scene `nodes`).
    fn node_refs(&mut self, mut i: i32, out: &mut Vec<usize>) -> i32 {
        i = self.array(i, out);
        check!(i);

        for v in out.iter_mut() {
            *v = ptrindex(self.to_int(i));
            i += 1;
        }

        i
    }

    fn skin(&mut self, mut i: i32, out: &mut Skin) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "name") {
                i = self.string(i + 1, &mut out.name);
            } else if self.streq(i, "joints") {
                i = self.node_refs(i + 1, &mut out.joints);
            } else if self.streq(i, "skeleton") {
                i += 1;
                check_toktype!(self.tok(i), JsmnType::Primitive);
                out.skeleton = ptrindex_opt(self.to_int(i));
                i += 1;
            } else if self.streq(i, "inverseBindMatrices") {
                i += 1;
                check_toktype!(self.tok(i), JsmnType::Primitive);
                out.inverse_bind_matrices = ptrindex_opt(self.to_int(i));
                i += 1;
            } else if self.streq(i, "extras") {
                i = self.extras(i + 1, &mut out.extras);
            } else if self.streq(i, "extensions") {
                i = self.unprocessed_extensions(i, &mut out.extensions);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn camera(&mut self, mut i: i32, out: &mut Camera) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "name") {
                i = self.string(i + 1, &mut out.name);
            } else if self.streq(i, "perspective") {
                i += 1;

                check_toktype!(self.tok(i), JsmnType::Object);

                let data_size = self.tok(i).size;
                i += 1;

                if out.type_ != CameraType::Invalid {
                    return CGLTF_ERROR_JSON;
                }

                out.type_ = CameraType::Perspective;

                for _ in 0..data_size {
                    check_key!(self.tok(i));

                    let p = &mut out.perspective;
                    if self.streq(i, "aspectRatio") {
                        i += 1;
                        p.has_aspect_ratio = true;
                        p.aspect_ratio = self.to_float(i);
                        i += 1;
                    } else if self.streq(i, "yfov") {
                        i += 1;
                        p.yfov = self.to_float(i);
                        i += 1;
                    } else if self.streq(i, "zfar") {
                        i += 1;
                        p.has_zfar = true;
                        p.zfar = self.to_float(i);
                        i += 1;
                    } else if self.streq(i, "znear") {
                        i += 1;
                        p.znear = self.to_float(i);
                        i += 1;
                    } else if self.streq(i, "extras") {
                        i = self.extras(i + 1, &mut p.extras);
                    } else {
                        i = self.skip(i + 1);
                    }

                    check!(i);
                }
            } else if self.streq(i, "orthographic") {
                i += 1;

                check_toktype!(self.tok(i), JsmnType::Object);

                let data_size = self.tok(i).size;
                i += 1;

                if out.type_ != CameraType::Invalid {
                    return CGLTF_ERROR_JSON;
                }

                out.type_ = CameraType::Orthographic;

                for _ in 0..data_size {
                    check_key!(self.tok(i));

                    let o = &mut out.orthographic;
                    if self.streq(i, "xmag") {
                        i += 1;
                        o.xmag = self.to_float(i);
                        i += 1;
                    } else if self.streq(i, "ymag") {
                        i += 1;
                        o.ymag = self.to_float(i);
                        i += 1;
                    } else if self.streq(i, "zfar") {
                        i += 1;
                        o.zfar = self.to_float(i);
                        i += 1;
                    } else if self.streq(i, "znear") {
                        i += 1;
                        o.znear = self.to_float(i);
                        i += 1;
                    } else if self.streq(i, "extras") {
                        i = self.extras(i + 1, &mut o.extras);
                    } else {
                        i = self.skip(i + 1);
                    }

                    check!(i);
                }
            } else if self.streq(i, "extras") {
                i = self.extras(i + 1, &mut out.extras);
            } else if self.streq(i, "extensions") {
                i = self.unprocessed_extensions(i, &mut out.extensions);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn light(&mut self, mut i: i32, out: &mut Light) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        out.color = [1.0; 3];
        out.intensity = 1.0;

        out.spot_inner_cone_angle = 0.0;
        out.spot_outer_cone_angle = std::f32::consts::PI / 4.0;

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "name") {
                i = self.string(i + 1, &mut out.name);
            } else if self.streq(i, "color") {
                i = self.float_array(i + 1, &mut out.color, 3);
            } else if self.streq(i, "intensity") {
                i += 1;
                out.intensity = self.to_float(i);
                i += 1;
            } else if self.streq(i, "type") {
                i += 1;
                if self.streq(i, "directional") {
                    out.type_ = LightType::Directional;
                } else if self.streq(i, "point") {
                    out.type_ = LightType::Point;
                } else if self.streq(i, "spot") {
                    out.type_ = LightType::Spot;
                }
                i += 1;
            } else if self.streq(i, "range") {
                i += 1;
                out.range = self.to_float(i);
                i += 1;
            } else if self.streq(i, "spot") {
                i += 1;

                check_toktype!(self.tok(i), JsmnType::Object);

                let data_size = self.tok(i).size;
                i += 1;

                for _ in 0..data_size {
                    check_key!(self.tok(i));

                    if self.streq(i, "innerConeAngle") {
                        i += 1;
                        out.spot_inner_cone_angle = self.to_float(i);
                        i += 1;
                    } else if self.streq(i, "outerConeAngle") {
                        i += 1;
                        out.spot_outer_cone_angle = self.to_float(i);
                        i += 1;
                    } else {
                        i = self.skip(i + 1);
                    }

                    check!(i);
                }
            } else if self.streq(i, "extras") {
                i = self.extras(i + 1, &mut out.extras);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn node(&mut self, mut i: i32, out: &mut Node) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        out.rotation[3] = 1.0;
        out.scale = [1.0; 3];
        out.matrix[0] = 1.0;
        out.matrix[5] = 1.0;
        out.matrix[10] = 1.0;
        out.matrix[15] = 1.0;

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "name") {
                i = self.string(i + 1, &mut out.name);
            } else if self.streq(i, "children") {
                i = self.node_refs(i + 1, &mut out.children);
            } else if self.streq(i, "mesh") {
                i += 1;
                check_toktype!(self.tok(i), JsmnType::Primitive);
                out.mesh = ptrindex_opt(self.to_int(i));
                i += 1;
            } else if self.streq(i, "skin") {
                i += 1;
                check_toktype!(self.tok(i), JsmnType::Primitive);
                out.skin = ptrindex_opt(self.to_int(i));
                i += 1;
            } else if self.streq(i, "camera") {
                i += 1;
                check_toktype!(self.tok(i), JsmnType::Primitive);
                out.camera = ptrindex_opt(self.to_int(i));
                i += 1;
            } else if self.streq(i, "translation") {
                out.has_translation = true;
                i = self.float_array(i + 1, &mut out.translation, 3);
            } else if self.streq(i, "rotation") {
                out.has_rotation = true;
                i = self.float_array(i + 1, &mut out.rotation, 4);
            } else if self.streq(i, "scale") {
                out.has_scale = true;
                i = self.float_array(i + 1, &mut out.scale, 3);
            } else if self.streq(i, "matrix") {
                out.has_matrix = true;
                i = self.float_array(i + 1, &mut out.matrix, 16);
            } else if self.streq(i, "weights") {
                i = self.array_opt(i + 1, &mut out.weights);
                check!(i);

                let weights = out.weights.get_or_insert_default();
                let count = weights.len() as i32;
                i = self.float_array(i - 1, weights, count);
            } else if self.streq(i, "extras") {
                i = self.extras(i + 1, &mut out.extras);
            } else if self.streq(i, "extensions") {
                let (start, extensions_size) = match self.begin_extensions(i, &out.extensions) {
                    Ok(v) => v,
                    Err(e) => return e,
                };
                i = start;

                for _ in 0..extensions_size {
                    check_key!(self.tok(i));

                    if self.streq(i, "KHR_lights_punctual") {
                        i += 1;

                        check_toktype!(self.tok(i), JsmnType::Object);

                        let data_size = self.tok(i).size;
                        i += 1;

                        for _ in 0..data_size {
                            check_key!(self.tok(i));

                            if self.streq(i, "light") {
                                i += 1;
                                check_toktype!(self.tok(i), JsmnType::Primitive);
                                out.light = ptrindex_opt(self.to_int(i));
                                i += 1;
                            } else {
                                i = self.skip(i + 1);
                            }

                            check!(i);
                        }
                    } else if self.streq(i, "EXT_mesh_gpu_instancing") {
                        out.has_mesh_gpu_instancing = true;
                        i = self.mesh_gpu_instancing(i + 1, &mut out.mesh_gpu_instancing);
                    } else {
                        i = self.push_unprocessed_extension(i, &mut out.extensions);
                    }

                    check!(i);
                }
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn scene(&mut self, mut i: i32, out: &mut Scene) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "name") {
                i = self.string(i + 1, &mut out.name);
            } else if self.streq(i, "nodes") {
                i = self.node_refs(i + 1, &mut out.nodes);
            } else if self.streq(i, "extras") {
                i = self.extras(i + 1, &mut out.extras);
            } else if self.streq(i, "extensions") {
                i = self.unprocessed_extensions(i, &mut out.extensions);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn animation_sampler(&mut self, mut i: i32, out: &mut AnimationSampler) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "input") {
                i += 1;
                out.input = ptrindex(self.to_int(i));
                i += 1;
            } else if self.streq(i, "output") {
                i += 1;
                out.output = ptrindex(self.to_int(i));
                i += 1;
            } else if self.streq(i, "interpolation") {
                i += 1;
                if self.streq(i, "LINEAR") {
                    out.interpolation = InterpolationType::Linear;
                } else if self.streq(i, "STEP") {
                    out.interpolation = InterpolationType::Step;
                } else if self.streq(i, "CUBICSPLINE") {
                    out.interpolation = InterpolationType::CubicSpline;
                }
                i += 1;
            } else if self.streq(i, "extras") {
                i = self.extras(i + 1, &mut out.extras);
            } else if self.streq(i, "extensions") {
                i = self.unprocessed_extensions(i, &mut out.extensions);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn animation_channel(&mut self, mut i: i32, out: &mut AnimationChannel) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "sampler") {
                i += 1;
                out.sampler = ptrindex(self.to_int(i));
                i += 1;
            } else if self.streq(i, "target") {
                i += 1;

                check_toktype!(self.tok(i), JsmnType::Object);

                let target_size = self.tok(i).size;
                i += 1;

                for _ in 0..target_size {
                    check_key!(self.tok(i));

                    if self.streq(i, "node") {
                        i += 1;
                        out.target_node = ptrindex_opt(self.to_int(i));
                        i += 1;
                    } else if self.streq(i, "path") {
                        i += 1;
                        if self.streq(i, "translation") {
                            out.target_path = AnimationPathType::Translation;
                        } else if self.streq(i, "rotation") {
                            out.target_path = AnimationPathType::Rotation;
                        } else if self.streq(i, "scale") {
                            out.target_path = AnimationPathType::Scale;
                        } else if self.streq(i, "weights") {
                            out.target_path = AnimationPathType::Weights;
                        }
                        i += 1;
                    } else if self.streq(i, "extras") {
                        i = self.extras(i + 1, &mut out.extras);
                    } else if self.streq(i, "extensions") {
                        i = self.unprocessed_extensions(i, &mut out.extensions);
                    } else {
                        i = self.skip(i + 1);
                    }

                    check!(i);
                }
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn animation(&mut self, mut i: i32, out: &mut Animation) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "name") {
                i = self.string(i + 1, &mut out.name);
            } else if self.streq(i, "samplers") {
                i = self.array(i + 1, &mut out.samplers);
                check!(i);

                for sampler in out.samplers.iter_mut() {
                    i = self.animation_sampler(i, sampler);
                    check!(i);
                }
            } else if self.streq(i, "channels") {
                i = self.array(i + 1, &mut out.channels);
                check!(i);

                for channel in out.channels.iter_mut() {
                    i = self.animation_channel(i, channel);
                    check!(i);
                }
            } else if self.streq(i, "extras") {
                i = self.extras(i + 1, &mut out.extras);
            } else if self.streq(i, "extensions") {
                i = self.unprocessed_extensions(i, &mut out.extensions);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn variant(&mut self, mut i: i32, out: &mut MaterialVariant) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "name") {
                i = self.string(i + 1, &mut out.name);
            } else if self.streq(i, "extras") {
                i = self.extras(i + 1, &mut out.extras);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }

    fn asset(&mut self, mut i: i32, out: &mut Asset) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "copyright") {
                i = self.string(i + 1, &mut out.copyright);
            } else if self.streq(i, "generator") {
                i = self.string(i + 1, &mut out.generator);
            } else if self.streq(i, "version") {
                i = self.string(i + 1, &mut out.version);
            } else if self.streq(i, "minVersion") {
                i = self.string(i + 1, &mut out.min_version);
            } else if self.streq(i, "extras") {
                i = self.extras(i + 1, &mut out.extras);
            } else if self.streq(i, "extensions") {
                i = self.unprocessed_extensions(i, &mut out.extensions);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        if let Some(version) = &out.version
            && cstd::atof(version.as_bytes()) < 2.0
        {
            return CGLTF_ERROR_LEGACY;
        }

        i
    }

    /// `cgltf_parse_json_<objects>`: array of objects at the root.
    fn objects<T: Default + Clone>(&mut self, mut i: i32, out: &mut Vec<T>, parse: fn(&mut Self, i32, &mut T) -> i32) -> i32 {
        i = self.array(i, out);
        check!(i);

        for item in out.iter_mut() {
            i = parse(self, i, item);
            check!(i);
        }
        i
    }

    fn root(&mut self, mut i: i32, out: &mut Data) -> i32 {
        check_toktype!(self.tok(i), JsmnType::Object);

        let size = self.tok(i).size;
        i += 1;

        for _ in 0..size {
            check_key!(self.tok(i));

            if self.streq(i, "asset") {
                i = self.asset(i + 1, &mut out.asset);
            } else if self.streq(i, "meshes") {
                i = self.objects(i + 1, &mut out.meshes, Self::mesh);
            } else if self.streq(i, "accessors") {
                i = self.objects(i + 1, &mut out.accessors, Self::accessor);
            } else if self.streq(i, "bufferViews") {
                i = self.objects(i + 1, &mut out.buffer_views, Self::buffer_view);
            } else if self.streq(i, "buffers") {
                i = self.objects(i + 1, &mut out.buffers, Self::buffer);
            } else if self.streq(i, "materials") {
                i = self.objects(i + 1, &mut out.materials, Self::material);
            } else if self.streq(i, "images") {
                i = self.objects(i + 1, &mut out.images, Self::image);
            } else if self.streq(i, "textures") {
                i = self.objects(i + 1, &mut out.textures, Self::texture);
            } else if self.streq(i, "samplers") {
                i = self.objects(i + 1, &mut out.samplers, Self::sampler);
            } else if self.streq(i, "skins") {
                i = self.objects(i + 1, &mut out.skins, Self::skin);
            } else if self.streq(i, "cameras") {
                i = self.objects(i + 1, &mut out.cameras, Self::camera);
            } else if self.streq(i, "nodes") {
                i = self.objects(i + 1, &mut out.nodes, Self::node);
            } else if self.streq(i, "scenes") {
                i = self.objects(i + 1, &mut out.scenes, Self::scene);
            } else if self.streq(i, "scene") {
                i += 1;
                out.scene = ptrindex_opt(self.to_int(i));
                i += 1;
            } else if self.streq(i, "animations") {
                i = self.objects(i + 1, &mut out.animations, Self::animation);
            } else if self.streq(i, "extras") {
                i = self.extras(i + 1, &mut out.extras);
            } else if self.streq(i, "extensions") {
                let (start, extensions_size) = match self.begin_extensions(i, &out.data_extensions) {
                    Ok(v) => v,
                    Err(e) => return e,
                };
                i = start;

                for _ in 0..extensions_size {
                    check_key!(self.tok(i));

                    if self.streq(i, "KHR_lights_punctual") {
                        i += 1;

                        check_toktype!(self.tok(i), JsmnType::Object);

                        let data_size = self.tok(i).size;
                        i += 1;

                        for _ in 0..data_size {
                            check_key!(self.tok(i));

                            if self.streq(i, "lights") {
                                i = self.objects(i + 1, &mut out.lights, Self::light);
                            } else {
                                i = self.skip(i + 1);
                            }

                            check!(i);
                        }
                    } else if self.streq(i, "KHR_materials_variants") {
                        i += 1;

                        check_toktype!(self.tok(i), JsmnType::Object);

                        let data_size = self.tok(i).size;
                        i += 1;

                        for _ in 0..data_size {
                            check_key!(self.tok(i));

                            if self.streq(i, "variants") {
                                i = self.objects(i + 1, &mut out.variants, Self::variant);
                            } else {
                                i = self.skip(i + 1);
                            }

                            check!(i);
                        }
                    } else {
                        i = self.push_unprocessed_extension(i, &mut out.data_extensions);
                    }

                    check!(i);
                }
            } else if self.streq(i, "extensionsUsed") {
                i = self.string_array(i + 1, &mut out.extensions_used);
            } else if self.streq(i, "extensionsRequired") {
                i = self.string_array(i + 1, &mut out.extensions_required);
            } else {
                i = self.skip(i + 1);
            }

            check!(i);
        }

        i
    }
}

fn parse_attribute_type(name: &str, index: i32) -> (AttributeType, i32) {
    let bytes = name.as_bytes();

    if bytes.first() == Some(&b'_') {
        return (AttributeType::Custom, index);
    }

    let us = bytes.iter().position(|&c| c == b'_');
    let prefix = &bytes[..us.unwrap_or(bytes.len())];

    let type_ = match prefix {
        b"POSITION" => AttributeType::Position,
        b"NORMAL" => AttributeType::Normal,
        b"TANGENT" => AttributeType::Tangent,
        b"TEXCOORD" => AttributeType::Texcoord,
        b"COLOR" => AttributeType::Color,
        b"JOINTS" => AttributeType::Joints,
        b"WEIGHTS" => AttributeType::Weights,
        _ => AttributeType::Invalid,
    };

    if let Some(us) = us
        && type_ != AttributeType::Invalid
    {
        let index = cstd::atoi(&bytes[us + 1..]);
        if index < 0 {
            return (AttributeType::Invalid, 0);
        }
        return (type_, index);
    }

    (type_, index)
}

fn json_to_primitive_type(type_: i32) -> PrimitiveType {
    match type_ {
        0 => PrimitiveType::Points,
        1 => PrimitiveType::Lines,
        2 => PrimitiveType::LineLoop,
        3 => PrimitiveType::LineStrip,
        4 => PrimitiveType::Triangles,
        5 => PrimitiveType::TriangleStrip,
        6 => PrimitiveType::TriangleFan,
        _ => PrimitiveType::Invalid,
    }
}

fn json_to_component_type(type_: i32) -> ComponentType {
    match type_ {
        5120 => ComponentType::R8,
        5121 => ComponentType::R8u,
        5122 => ComponentType::R16,
        5123 => ComponentType::R16u,
        5125 => ComponentType::R32u,
        5126 => ComponentType::R32f,
        _ => ComponentType::Invalid,
    }
}

/// `CGLTF_PTRFIXUP`: raw `CGLTF_PTRINDEX` value → index, rejecting out of range references.
fn fixup(var: &mut Option<usize>, size: usize) -> Result<(), ()> {
    if let Some(raw) = *var {
        if raw > size {
            return Err(());
        }
        *var = Some(raw - 1);
    }
    Ok(())
}

/// `CGLTF_PTRFIXUP_REQ`
fn fixup_req(var: &mut usize, size: usize) -> Result<(), ()> {
    if *var == 0 || *var > size {
        return Err(());
    }
    *var -= 1;
    Ok(())
}

fn fixup_pointers(data: &mut Data) -> Result<(), ()> {
    let accessors_count = data.accessors.len();
    let materials_count = data.materials.len();
    let buffer_views_count = data.buffer_views.len();
    let buffers_count = data.buffers.len();
    let images_count = data.images.len();
    let samplers_count = data.samplers.len();
    let textures_count = data.textures.len();
    let nodes_count = data.nodes.len();
    let meshes_count = data.meshes.len();
    let skins_count = data.skins.len();
    let cameras_count = data.cameras.len();
    let lights_count = data.lights.len();
    let scenes_count = data.scenes.len();

    for mesh in data.meshes.iter_mut() {
        for prim in mesh.primitives.iter_mut() {
            fixup(&mut prim.indices, accessors_count)?;
            fixup(&mut prim.material, materials_count)?;

            for attr in prim.attributes.iter_mut() {
                fixup_req(&mut attr.data, accessors_count)?;
            }

            for target in prim.targets.iter_mut() {
                for attr in target.attributes.iter_mut() {
                    fixup_req(&mut attr.data, accessors_count)?;
                }
            }

            if prim.has_draco_mesh_compression {
                fixup_req(&mut prim.draco_mesh_compression.buffer_view, buffer_views_count)?;
                for attr in prim.draco_mesh_compression.attributes.iter_mut() {
                    fixup_req(&mut attr.data, accessors_count)?;
                }
            }

            for mapping in prim.mappings.iter_mut() {
                fixup_req(&mut mapping.material, materials_count)?;
            }
        }
    }

    for i in 0..data.accessors.len() {
        let accessor = &mut data.accessors[i];
        fixup(&mut accessor.buffer_view, buffer_views_count)?;

        if accessor.is_sparse {
            fixup_req(&mut accessor.sparse.indices_buffer_view, buffer_views_count)?;
            fixup_req(&mut accessor.sparse.values_buffer_view, buffer_views_count)?;
        }

        if let Some(bv) = accessor.buffer_view {
            accessor.stride = data.buffer_views[bv].stride;
        }

        if accessor.stride == 0 {
            accessor.stride = super::calc_size(accessor.type_, accessor.component_type);
        }
    }

    for texture in data.textures.iter_mut() {
        fixup(&mut texture.image, images_count)?;
        fixup(&mut texture.basisu_image, images_count)?;
        fixup(&mut texture.webp_image, images_count)?;
        fixup(&mut texture.sampler, samplers_count)?;
    }

    for image in data.images.iter_mut() {
        fixup(&mut image.buffer_view, buffer_views_count)?;
    }

    for material in data.materials.iter_mut() {
        let views: [&mut TextureView; 21] = [
            &mut material.normal_texture,
            &mut material.emissive_texture,
            &mut material.occlusion_texture,
            &mut material.pbr_metallic_roughness.base_color_texture,
            &mut material.pbr_metallic_roughness.metallic_roughness_texture,
            &mut material.pbr_specular_glossiness.diffuse_texture,
            &mut material.pbr_specular_glossiness.specular_glossiness_texture,
            &mut material.clearcoat.clearcoat_texture,
            &mut material.clearcoat.clearcoat_roughness_texture,
            &mut material.clearcoat.clearcoat_normal_texture,
            &mut material.specular.specular_texture,
            &mut material.specular.specular_color_texture,
            &mut material.transmission.transmission_texture,
            &mut material.volume.thickness_texture,
            &mut material.sheen.sheen_color_texture,
            &mut material.sheen.sheen_roughness_texture,
            &mut material.iridescence.iridescence_texture,
            &mut material.iridescence.iridescence_thickness_texture,
            &mut material.diffuse_transmission.diffuse_transmission_texture,
            &mut material.diffuse_transmission.diffuse_transmission_color_texture,
            &mut material.anisotropy.anisotropy_texture,
        ];

        for view in views {
            fixup(&mut view.texture, textures_count)?;
        }
    }

    for view in data.buffer_views.iter_mut() {
        fixup_req(&mut view.buffer, buffers_count)?;

        if view.has_meshopt_compression {
            fixup_req(&mut view.meshopt_compression.buffer, buffers_count)?;
        }
    }

    for skin in data.skins.iter_mut() {
        for joint in skin.joints.iter_mut() {
            fixup_req(joint, nodes_count)?;
        }

        fixup(&mut skin.skeleton, nodes_count)?;
        fixup(&mut skin.inverse_bind_matrices, accessors_count)?;
    }

    for i in 0..data.nodes.len() {
        for j in 0..data.nodes[i].children.len() {
            fixup_req(&mut data.nodes[i].children[j], nodes_count)?;

            let child = data.nodes[i].children[j];
            if data.nodes[child].parent.is_some() {
                return Err(());
            }

            data.nodes[child].parent = Some(i);
        }

        let node = &mut data.nodes[i];
        fixup(&mut node.mesh, meshes_count)?;
        fixup(&mut node.skin, skins_count)?;
        fixup(&mut node.camera, cameras_count)?;
        fixup(&mut node.light, lights_count)?;

        if node.has_mesh_gpu_instancing {
            for attr in node.mesh_gpu_instancing.attributes.iter_mut() {
                fixup_req(&mut attr.data, accessors_count)?;
            }
        }
    }

    for i in 0..data.scenes.len() {
        for j in 0..data.scenes[i].nodes.len() {
            fixup_req(&mut data.scenes[i].nodes[j], nodes_count)?;

            if data.nodes[data.scenes[i].nodes[j]].parent.is_some() {
                return Err(());
            }
        }
    }

    fixup(&mut data.scene, scenes_count)?;

    for animation in data.animations.iter_mut() {
        for sampler in animation.samplers.iter_mut() {
            fixup_req(&mut sampler.input, accessors_count)?;
            fixup_req(&mut sampler.output, accessors_count)?;
        }

        let samplers_count = animation.samplers.len();
        for channel in animation.channels.iter_mut() {
            fixup_req(&mut channel.sampler, samplers_count)?;
            fixup(&mut channel.target_node, nodes_count)?;
        }
    }

    Ok(())
}

/// `cgltf_parse_json`
pub(crate) fn parse_json(options: &mut Options, json_chunk: &[u8]) -> Result<Data, CgltfResult> {
    let size = json_chunk.len();

    if options.json_token_count == 0 {
        let mut parser = JsmnParser::new();
        let token_count = jsmn_parse(&mut parser, json_chunk, size, None);

        if token_count <= 0 {
            return Err(CgltfResult::InvalidJson);
        }

        options.json_token_count = token_count as CgltfSize;
    }

    let mut tokens = vec![JsmnTok::default(); options.json_token_count + 1];

    let mut parser = JsmnParser::new();
    let token_count = {
        let count = options.json_token_count;
        jsmn_parse(&mut parser, json_chunk, size, Some(&mut tokens[..count]))
    };

    if token_count <= 0 {
        return Err(CgltfResult::InvalidJson);
    }

    // this makes sure that we always have an UNDEFINED token at the end of the stream
    // for invalid JSON inputs this makes sure we don't perform out of bound reads of token data
    tokens.truncate(token_count as usize);

    let mut data = Data::default();

    let i = {
        let mut ctx = Parser {
            tokens: &tokens,
            json: json_chunk,
            allocated: HashSet::new(),
        };
        ctx.root(0, &mut data)
    };

    if i < 0 {
        return Err(match i {
            CGLTF_ERROR_NOMEM => CgltfResult::OutOfMemory,
            CGLTF_ERROR_LEGACY => CgltfResult::LegacyGltf,
            _ => CgltfResult::InvalidGltf,
        });
    }

    if fixup_pointers(&mut data).is_err() {
        return Err(CgltfResult::InvalidGltf);
    }

    data.json = Some(json_chunk.to_vec());

    Ok(data)
}
