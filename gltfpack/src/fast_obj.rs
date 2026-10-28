//! Port of fast_obj 1.3 (https://github.com/thisistherk/fast_obj), the OBJ/MTL reader used by gltfpack.
//!
//! Names and paths are kept as raw bytes (C strings may hold any byte); reads use the same 64KB chunking as the C
//! reader so that overlong lines stop parsing at the same place.

#[derive(Debug, Clone, Default)]
pub struct FastObjTexture {
    /// Texture name from .mtl file
    pub name: Option<Vec<u8>>,
    /// Resolved path to texture
    pub path: Option<Vec<u8>>,
}

#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct FastObjMaterial {
    /// Material name
    pub name: Option<Vec<u8>>,

    /// Ambient
    pub Ka: [f32; 3],
    /// Diffuse
    pub Kd: [f32; 3],
    /// Specular
    pub Ks: [f32; 3],
    /// Emission
    pub Ke: [f32; 3],
    /// Transmittance
    pub Kt: [f32; 3],
    /// Shininess
    pub Ns: f32,
    /// Index of refraction
    pub Ni: f32,
    /// Transmission filter
    pub Tf: [f32; 3],
    /// Disolve (alpha)
    pub d: f32,
    /// Illumination model
    pub illum: i32,

    /// Set for materials that don't come from the associated mtllib
    pub fallback: i32,

    /// Texture map indices in `FastObjMesh::textures`
    pub map_Ka: u32,
    pub map_Kd: u32,
    pub map_Ks: u32,
    pub map_Ke: u32,
    pub map_Kt: u32,
    pub map_Ns: u32,
    pub map_Ni: u32,
    pub map_d: u32,
    pub map_bump: u32,
}

/// `mtl_default`
impl Default for FastObjMaterial {
    fn default() -> Self {
        Self {
            name: None,
            Ka: [0.0; 3],
            Kd: [1.0; 3],
            Ks: [0.0; 3],
            Ke: [0.0; 3],
            Kt: [0.0; 3],
            Ns: 1.0,
            Ni: 1.0,
            Tf: [1.0; 3],
            d: 1.0,
            illum: 1,
            fallback: 0,
            map_Ka: 0,
            map_Kd: 0,
            map_Ks: 0,
            map_Ke: 0,
            map_Kt: 0,
            map_Ns: 0,
            map_Ni: 0,
            map_d: 0,
            map_bump: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FastObjIndex {
    pub p: u32,
    pub t: u32,
    pub n: u32,
}

#[derive(Debug, Clone, Default)]
pub struct FastObjGroup {
    /// Group name
    pub name: Option<Vec<u8>>,
    /// Number of faces
    pub face_count: u32,
    /// First face in `FastObjMesh` face_* arrays
    pub face_offset: u32,
    /// First index in `FastObjMesh::indices`
    pub index_offset: u32,
}

/// Note: a dummy zero-initialized value is added to the first index of the positions, texcoords, normals and
/// textures arrays. Hence, valid indices into these arrays start from 1, with an index of 0 indicating that the
/// attribute is not present.
#[derive(Debug, Clone, Default)]
pub struct FastObjMesh {
    /// Vertex data
    pub position_count: u32,
    pub positions: Vec<f32>,

    pub texcoord_count: u32,
    pub texcoords: Vec<f32>,

    pub normal_count: u32,
    pub normals: Vec<f32>,

    pub color_count: u32,
    pub colors: Vec<f32>,

    /// Face data: one element for each face
    pub face_count: u32,
    pub face_vertices: Vec<u32>,
    pub face_materials: Vec<u32>,
    /// empty unless the file has line elements (C: NULL)
    pub face_lines: Vec<u8>,

    /// Index data: one element for each face vertex
    pub index_count: u32,
    pub indices: Vec<FastObjIndex>,

    /// Materials
    pub material_count: u32,
    pub materials: Vec<FastObjMaterial>,

    /// Texture maps
    pub texture_count: u32,
    pub textures: Vec<FastObjTexture>,

    /// Mesh objects ('o' tag in .obj file)
    pub object_count: u32,
    pub objects: Vec<FastObjGroup>,

    /// Mesh groups ('g' tag in .obj file)
    pub group_count: u32,
    pub groups: Vec<FastObjGroup>,
}

#[cfg(windows)]
const FAST_OBJ_SEPARATOR: u8 = b'\\';
#[cfg(windows)]
const FAST_OBJ_OTHER_SEP: u8 = b'/';
#[cfg(not(windows))]
const FAST_OBJ_SEPARATOR: u8 = b'/';
#[cfg(not(windows))]
const FAST_OBJ_OTHER_SEP: u8 = b'\\';

/// Size of buffer to read into
const BUFFER_SIZE: usize = 65536;

/// Max supported power when parsing float
const MAX_POWER: u32 = 20;

const POWER_10_POS: [f64; MAX_POWER as usize] = [
    1.0e0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8, 1.0e9, 1.0e10, 1.0e11, 1.0e12, 1.0e13, 1.0e14, 1.0e15, 1.0e16, 1.0e17,
    1.0e18, 1.0e19,
];

const POWER_10_NEG: [f64; MAX_POWER as usize] = [
    1.0e0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12, 1.0e-13, 1.0e-14, 1.0e-15,
    1.0e-16, 1.0e-17, 1.0e-18, 1.0e-19,
];

struct FastObjData {
    /// Final mesh
    mesh: FastObjMesh,

    /// Current object/group
    object: FastObjGroup,
    group: FastObjGroup,

    /// Current material index
    material: u32,

    /// Current line in file
    line: u32,

    /// Base path for materials/textures
    base: Option<Vec<u8>>,
}

/// Byte at `i`; past the end reads as a newline (every C buffer is newline-terminated).
fn at(s: &[u8], i: usize) -> u8 {
    s.get(i).copied().unwrap_or(b'\n')
}

/// C `strlen` on a byte string.
fn c_str(s: &[u8]) -> &[u8] {
    &s[..s.iter().position(|&c| c == 0).unwrap_or(s.len())]
}

fn string_concat(a: Option<&[u8]>, s: &[u8]) -> Vec<u8> {
    let mut p = a.map(|a| c_str(a).to_vec()).unwrap_or_default();
    p.extend_from_slice(s);
    p
}

fn string_equal(a: &[u8], s: &[u8]) -> bool {
    c_str(a) == s
}

fn string_fix_separators(s: &mut [u8]) {
    for c in s.iter_mut() {
        if *c == 0 {
            break;
        }
        if *c == FAST_OBJ_OTHER_SEP {
            *c = FAST_OBJ_SEPARATOR;
        }
    }
}

fn is_whitespace(c: u8) -> bool {
    c == b' ' || c == b'\t' || c == b'\r'
}

fn is_newline(c: u8) -> bool {
    c == b'\n'
}

fn is_digit(c: u8) -> bool {
    c.is_ascii_digit()
}

fn is_exponent(c: u8) -> bool {
    c == b'e' || c == b'E'
}

fn skip_name(b: &[u8], mut ptr: usize) -> usize {
    let s = ptr;

    while !is_newline(at(b, ptr)) {
        ptr += 1;
    }

    while ptr > s && is_whitespace(at(b, ptr - 1)) {
        ptr -= 1;
    }

    ptr
}

fn skip_whitespace(b: &[u8], mut ptr: usize) -> usize {
    while is_whitespace(at(b, ptr)) {
        ptr += 1;
    }

    ptr
}

fn skip_line(b: &[u8], mut ptr: usize) -> usize {
    loop {
        let c = at(b, ptr);
        ptr += 1;
        if is_newline(c) || ptr > b.len() {
            break;
        }
    }

    ptr
}

fn flush_object(data: &mut FastObjData) {
    // Add object if not empty
    let object = std::mem::take(&mut data.object);
    if object.face_count > 0 {
        data.mesh.objects.push(object);
    }

    // Reset for more data
    data.object.face_offset = data.mesh.face_vertices.len() as u32;
    data.object.index_offset = data.mesh.indices.len() as u32;
}

fn flush_group(data: &mut FastObjData) {
    // Add group if not empty
    let group = std::mem::take(&mut data.group);
    if group.face_count > 0 {
        data.mesh.groups.push(group);
    }

    // Reset for more data
    data.group.face_offset = data.mesh.face_vertices.len() as u32;
    data.group.index_offset = data.mesh.indices.len() as u32;
}

fn parse_int(b: &[u8], mut ptr: usize) -> (usize, i32) {
    let sign: i32 = if at(b, ptr) == b'-' {
        ptr += 1;
        -1
    } else {
        1
    };

    let mut num: i32 = 0;
    while is_digit(at(b, ptr)) {
        num = num.wrapping_mul(10).wrapping_add(i32::from(at(b, ptr) - b'0'));
        ptr += 1;
    }

    (ptr, sign.wrapping_mul(num))
}

fn parse_float(b: &[u8], ptr: usize) -> (usize, f32) {
    let mut ptr = skip_whitespace(b, ptr);

    let sign = match at(b, ptr) {
        b'+' => {
            ptr += 1;
            1.0
        }
        b'-' => {
            ptr += 1;
            -1.0
        }
        _ => 1.0,
    };

    let mut num = 0.0f64;
    while is_digit(at(b, ptr)) {
        num = 10.0 * num + f64::from(at(b, ptr) - b'0');
        ptr += 1;
    }

    if at(b, ptr) == b'.' {
        ptr += 1;
    }

    let mut fra = 0.0f64;
    let mut div = 1.0f64;

    while is_digit(at(b, ptr)) {
        fra = 10.0 * fra + f64::from(at(b, ptr) - b'0');
        div *= 10.0;
        ptr += 1;
    }

    num += fra / div;

    if is_exponent(at(b, ptr)) {
        ptr += 1;

        let powers = match at(b, ptr) {
            b'+' => {
                ptr += 1;
                &POWER_10_POS
            }
            b'-' => {
                ptr += 1;
                &POWER_10_NEG
            }
            _ => &POWER_10_POS,
        };

        let mut eval: u32 = 0;
        while is_digit(at(b, ptr)) {
            eval = eval.wrapping_mul(10).wrapping_add(u32::from(at(b, ptr) - b'0'));
            ptr += 1;
        }

        num *= if eval >= MAX_POWER { 0.0 } else { powers[eval as usize] };
    }

    (ptr, (sign * num) as f32)
}

fn parse_vertex(data: &mut FastObjData, b: &[u8], mut ptr: usize) -> usize {
    for _ in 0..3 {
        let (p, v) = parse_float(b, ptr);
        ptr = p;
        data.mesh.positions.push(v);
    }

    ptr = skip_whitespace(b, ptr);
    if !is_newline(at(b, ptr)) {
        // Fill the colors array until it matches the size of the positions array
        let target = data.mesh.positions.len() - 3;
        while data.mesh.colors.len() < target {
            data.mesh.colors.push(1.0);
        }

        for _ in 0..3 {
            let (p, v) = parse_float(b, ptr);
            ptr = p;
            data.mesh.colors.push(v);
        }
    }

    ptr
}

fn parse_texcoord(data: &mut FastObjData, b: &[u8], mut ptr: usize) -> usize {
    for _ in 0..2 {
        let (p, v) = parse_float(b, ptr);
        ptr = p;
        data.mesh.texcoords.push(v);
    }

    ptr
}

fn parse_normal(data: &mut FastObjData, b: &[u8], mut ptr: usize) -> usize {
    for _ in 0..3 {
        let (p, v) = parse_float(b, ptr);
        ptr = p;
        data.mesh.normals.push(v);
    }

    ptr
}

/// Relative (negative) or absolute OBJ index, with C unsigned wraparound.
fn resolve_index(v: i32, count: usize) -> u32 {
    if v < 0 {
        (count as u32).wrapping_sub(v.wrapping_neg() as u32)
    } else {
        v as u32
    }
}

fn parse_face(data: &mut FastObjData, b: &[u8], ptr: usize, line: u8) -> usize {
    let mut ptr = skip_whitespace(b, ptr);

    let mut count: u32 = 0;
    while !is_newline(at(b, ptr)) {
        let mut t = 0;
        let mut n = 0;

        let (p, v) = parse_int(b, ptr);
        ptr = p;
        if at(b, ptr) == b'/' {
            ptr += 1;
            if at(b, ptr) != b'/' {
                (ptr, t) = parse_int(b, ptr);
            }

            if at(b, ptr) == b'/' {
                ptr += 1;
                (ptr, n) = parse_int(b, ptr);
            }
        }

        if v == 0 {
            // Skip lines with no valid vertex index
            return ptr;
        }

        let vn = FastObjIndex {
            p: resolve_index(v, data.mesh.positions.len() / 3),
            t: resolve_index(t, data.mesh.texcoords.len() / 2),
            n: resolve_index(n, data.mesh.normals.len() / 3),
        };

        data.mesh.indices.push(vn);
        count += 1;

        ptr = skip_whitespace(b, ptr);
    }

    data.mesh.face_vertices.push(count);
    data.mesh.face_materials.push(data.material);

    if line != 0 || !data.mesh.face_lines.is_empty() {
        // when line info exists, ensure it uses aligned indexing with other face data
        while data.mesh.face_lines.len() + 1 < data.mesh.face_vertices.len() {
            data.mesh.face_lines.push(0);
        }

        data.mesh.face_lines.push(line);
    }

    data.group.face_count += 1;
    data.object.face_count += 1;

    ptr
}

fn parse_object(data: &mut FastObjData, b: &[u8], ptr: usize) -> usize {
    let s = skip_whitespace(b, ptr);
    let e = skip_name(b, s);

    flush_object(data);
    data.object.name = Some(b[s..e].to_vec());

    e
}

fn parse_group(data: &mut FastObjData, b: &[u8], ptr: usize) -> usize {
    let s = skip_whitespace(b, ptr);
    let e = skip_name(b, s);

    flush_group(data);
    data.group.name = Some(b[s..e].to_vec());

    e
}

fn parse_usemtl(data: &mut FastObjData, b: &[u8], ptr: usize) -> usize {
    // Parse the material name
    let s = skip_whitespace(b, ptr);
    let e = skip_name(b, s);
    let name = &b[s..e];

    // Find an existing material with the same name
    let idx = data
        .mesh
        .materials
        .iter()
        .position(|m| m.name.as_deref().is_some_and(|n| string_equal(n, name)));

    // If doesn't exist, create a default one with this name
    // Note: this case happens when OBJ doesn't have its MTL
    let idx = match idx {
        Some(idx) => idx,
        None => {
            data.mesh.materials.push(FastObjMaterial {
                name: Some(name.to_vec()),
                fallback: 1,
                ..Default::default()
            });
            data.mesh.materials.len() - 1
        }
    };

    data.material = idx as u32;

    e
}

fn read_mtl_triple(b: &[u8], p: usize, v: &mut [f32; 3]) -> usize {
    let (p, x) = parse_float(b, p);
    let (p, y) = parse_float(b, p);
    let (p, z) = parse_float(b, p);
    *v = [x, y, z];
    p
}

fn read_mtl_single(b: &[u8], p: usize, v: &mut f32) -> usize {
    let (p, x) = parse_float(b, p);
    *v = x;
    p
}

fn read_map(data: &mut FastObjData, b: &[u8], ptr: usize, idx: &mut u32) -> usize {
    let ptr = skip_whitespace(b, ptr);

    // Don't support options at present
    if at(b, ptr) == b'-' {
        return ptr;
    }

    // Read name
    let s = ptr;
    let e = skip_name(b, ptr);
    let name = &b[s..e];

    // Try to find an existing texture map with the same name
    *idx = 1; // skip dummy at index 0
    while (*idx as usize) < data.mesh.textures.len() {
        let map = &data.mesh.textures[*idx as usize];
        if map.name.as_deref().is_some_and(|n| string_equal(n, name)) {
            break;
        }

        *idx += 1;
    }

    // Add it to the texture array if it didn't already exist
    if *idx as usize == data.mesh.textures.len() {
        let mut path = string_concat(data.base.as_deref(), name);
        string_fix_separators(&mut path);
        data.mesh.textures.push(FastObjTexture {
            name: Some(name.to_vec()),
            path: Some(path),
        });
    }

    e
}

fn matches_word(b: &[u8], p: usize, word: &[u8]) -> bool {
    word.iter().enumerate().all(|(k, &c)| at(b, p + k) == c)
}

fn read_mtllib(data: &mut FastObjData, file: &[u8]) {
    // Read entire file
    let mut contents = file.to_vec();
    let l = contents.len();
    contents.push(b'\n');
    let b = &contents[..];

    let mut mtl = FastObjMaterial::default();

    let mut found_d = false;

    let mut p = 0;
    let e = l;
    while p < e {
        p = skip_whitespace(b, p);

        match at(b, p) {
            b'n' => {
                p += 1;
                if matches_word(b, p, b"ewmtl") && is_whitespace(at(b, p + 5)) {
                    // Push previous material (if there is one)
                    if mtl.name.is_some() {
                        data.mesh.materials.push(std::mem::take(&mut mtl));
                    }

                    // Read name
                    p += 5;

                    while is_whitespace(at(b, p)) {
                        p += 1;
                    }

                    let s = p;
                    p = skip_name(b, p);

                    mtl.name = Some(b[s..p].to_vec());
                }
            }
            b'K' => match at(b, p + 1) {
                b'a' => p = read_mtl_triple(b, p + 2, &mut mtl.Ka),
                b'd' => p = read_mtl_triple(b, p + 2, &mut mtl.Kd),
                b's' => p = read_mtl_triple(b, p + 2, &mut mtl.Ks),
                b'e' => p = read_mtl_triple(b, p + 2, &mut mtl.Ke),
                b't' => p = read_mtl_triple(b, p + 2, &mut mtl.Kt),
                _ => {}
            },
            b'N' => match at(b, p + 1) {
                b's' => p = read_mtl_single(b, p + 2, &mut mtl.Ns),
                b'i' => p = read_mtl_single(b, p + 2, &mut mtl.Ni),
                _ => {}
            },
            b'T' => {
                if at(b, p + 1) == b'r' {
                    let mut tr = 0.0;
                    p = read_mtl_single(b, p + 2, &mut tr);
                    if !found_d {
                        // Ignore Tr if we've already read d
                        mtl.d = 1.0 - tr;
                    }
                } else if at(b, p + 1) == b'f' {
                    p = read_mtl_triple(b, p + 2, &mut mtl.Tf);
                }
            }
            b'd' => {
                if is_whitespace(at(b, p + 1)) {
                    p = read_mtl_single(b, p + 1, &mut mtl.d);
                    found_d = true;
                }
            }
            b'i' => {
                p += 1;
                if matches_word(b, p, b"llum") && is_whitespace(at(b, p + 4)) {
                    let (np, v) = parse_int(b, p + 4);
                    p = np;
                    mtl.illum = v;
                }
            }
            b'm' => {
                p += 1;
                if matches_word(b, p, b"ap_") {
                    p += 3;
                    if at(b, p) == b'K' {
                        p += 1;
                        if is_whitespace(at(b, p + 1)) {
                            match at(b, p) {
                                b'a' => p = read_map(data, b, p + 1, &mut mtl.map_Ka),
                                b'd' => p = read_map(data, b, p + 1, &mut mtl.map_Kd),
                                b's' => p = read_map(data, b, p + 1, &mut mtl.map_Ks),
                                b'e' => p = read_map(data, b, p + 1, &mut mtl.map_Ke),
                                b't' => p = read_map(data, b, p + 1, &mut mtl.map_Kt),
                                _ => {}
                            }
                        }
                    } else if at(b, p) == b'N' {
                        p += 1;
                        if is_whitespace(at(b, p + 1)) {
                            match at(b, p) {
                                b's' => p = read_map(data, b, p + 1, &mut mtl.map_Ns),
                                b'i' => p = read_map(data, b, p + 1, &mut mtl.map_Ni),
                                _ => {}
                            }
                        }
                    } else if at(b, p) == b'd' {
                        p += 1;
                        if is_whitespace(at(b, p)) {
                            p = read_map(data, b, p, &mut mtl.map_d);
                        }
                    } else if (at(b, p) == b'b' || at(b, p) == b'B') && matches_word(b, p + 1, b"ump") && is_whitespace(at(b, p + 4)) {
                        p = read_map(data, b, p + 4, &mut mtl.map_bump);
                    }
                }
            }
            _ => {}
        }

        p = skip_line(b, p);
    }

    // Push final material
    if mtl.name.is_some() {
        data.mesh.materials.push(mtl);
    }
}

fn path_string(path: &[u8]) -> String {
    String::from_utf8_lossy(c_str(path)).into_owned()
}

fn parse_mtllib(data: &mut FastObjData, b: &[u8], ptr: usize) -> usize {
    let s = skip_whitespace(b, ptr);
    let e = skip_name(b, s);

    let mut lib = string_concat(data.base.as_deref(), &b[s..e]);
    string_fix_separators(&mut lib);

    if let Ok(file) = std::fs::read(path_string(&lib)) {
        read_mtllib(data, &file);
    }

    e
}

fn parse_buffer(data: &mut FastObjData, b: &[u8], ptr: usize, end: usize) {
    let mut p = ptr;
    while p < end {
        p = skip_whitespace(b, p);

        match at(b, p) {
            b'v' => {
                p += 1;

                let c = at(b, p);
                p += 1;
                match c {
                    b' ' | b'\t' => p = parse_vertex(data, b, p),
                    b't' => p = parse_texcoord(data, b, p),
                    b'n' => p = parse_normal(data, b, p),
                    // roll p++ back in case *p was a newline
                    _ => p -= 1,
                }
            }
            b'f' | b'l' | b'o' | b'g' => {
                let tag = at(b, p);
                p += 1;

                let c = at(b, p);
                p += 1;
                match (c, tag) {
                    (b' ' | b'\t', b'f') => p = parse_face(data, b, p, 0),
                    (b' ' | b'\t', b'l') => p = parse_face(data, b, p, 1),
                    (b' ' | b'\t', b'o') => p = parse_object(data, b, p),
                    (b' ' | b'\t', _) => p = parse_group(data, b, p),
                    // roll p++ back in case *p was a newline
                    _ => p -= 1,
                }
            }
            b'm' => {
                p += 1;
                if matches_word(b, p, b"tllib") && is_whitespace(at(b, p + 5)) {
                    p = parse_mtllib(data, b, p + 5);
                }
            }
            b'u' => {
                p += 1;
                if matches_word(b, p, b"semtl") && is_whitespace(at(b, p + 5)) {
                    p = parse_usemtl(data, b, p + 5);
                }
            }
            _ => {}
        }

        p = skip_line(b, p);

        data.line += 1;
    }

    if !data.mesh.colors.is_empty() {
        // Fill the remaining slots in the colors array
        while data.mesh.colors.len() < data.mesh.positions.len() {
            data.mesh.colors.push(1.0);
        }
    }
}

/// `fast_obj_destroy`: releases the mesh.
pub fn fast_obj_destroy(mesh: FastObjMesh) {
    drop(mesh);
}

/// `fast_obj_read`: returns `None` if the file can't be opened.
pub fn fast_obj_read(path: &str) -> Option<FastObjMesh> {
    let file = std::fs::read(path).ok()?;

    let mut m = FastObjMesh::default();

    // Add dummy position/texcoord/normal/texture
    m.positions.extend_from_slice(&[0.0, 0.0, 0.0]);
    m.texcoords.extend_from_slice(&[0.0, 0.0]);
    m.normals.extend_from_slice(&[0.0, 0.0, 1.0]);
    m.textures.push(FastObjTexture::default());

    let path_bytes = path.as_bytes();

    // Find base path for materials/textures: use the last separator in the path
    let sep = path_bytes.iter().rposition(|&c| c == FAST_OBJ_SEPARATOR || c == FAST_OBJ_OTHER_SEP);

    let mut data = FastObjData {
        mesh: m,
        object: FastObjGroup::default(),
        group: FastObjGroup::default(),
        material: 0,
        line: 1,
        base: sep.map(|sep| path_bytes[..=sep].to_vec()),
    };

    // Create buffer for reading file
    let mut buffer = vec![0u8; 2 * BUFFER_SIZE + 1];
    let mut file_pos = 0;

    let mut start = 0;
    loop {
        // Read another buffer's worth from file
        let mut read = BUFFER_SIZE.min(file.len() - file_pos);
        buffer[start..start + read].copy_from_slice(&file[file_pos..file_pos + read]);
        file_pos += read;

        if read == 0 && start == 0 {
            break;
        }

        // Ensure buffer ends in a newline
        if read < BUFFER_SIZE && (read == 0 || buffer[start + read - 1] != b'\n') {
            buffer[start + read] = b'\n';
            read += 1;
        }

        let end = start + read;
        if end == 0 {
            break;
        }

        // Find last new line
        let mut last = end;
        while last > 0 {
            last -= 1;
            if buffer[last] == b'\n' {
                break;
            }
        }

        // Check there actually is a new line
        if buffer[last] != b'\n' {
            break;
        }

        last += 1;

        // Process buffer
        parse_buffer(&mut data, &buffer[..last], 0, last);

        // Copy overflow for next buffer
        buffer.copy_within(last..end, 0);
        start = end - last;
    }

    // Flush final object/group
    flush_object(&mut data);
    flush_group(&mut data);

    let mut m = data.mesh;

    m.position_count = (m.positions.len() / 3) as u32;
    m.texcoord_count = (m.texcoords.len() / 2) as u32;
    m.normal_count = (m.normals.len() / 3) as u32;
    m.color_count = (m.colors.len() / 3) as u32;
    m.face_count = m.face_vertices.len() as u32;
    m.index_count = m.indices.len() as u32;
    m.material_count = m.materials.len() as u32;
    m.texture_count = m.textures.len() as u32;
    m.object_count = m.objects.len() as u32;
    m.group_count = m.groups.len() as u32;

    Some(m)
}
