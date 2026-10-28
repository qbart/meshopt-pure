//! Port of micro_ply (https://github.com/maluoi/header-libs), the ASCII / binary little-endian PLY reader used by
//! gltfpack. Reads past the end of the input (C undefined behavior) make `ply_read` fail instead.

use crate::cstd;

pub const PLY_PROP_POSITION_X: &str = "x";
pub const PLY_PROP_POSITION_Y: &str = "y";
pub const PLY_PROP_POSITION_Z: &str = "z";
pub const PLY_PROP_NORMAL_X: &str = "nx";
pub const PLY_PROP_NORMAL_Y: &str = "ny";
pub const PLY_PROP_NORMAL_Z: &str = "nz";
pub const PLY_PROP_TEXCOORD_X: &str = "s";
pub const PLY_PROP_TEXCOORD_Y: &str = "t";
pub const PLY_PROP_COLOR_R: &str = "red";
pub const PLY_PROP_COLOR_G: &str = "green";
pub const PLY_PROP_COLOR_B: &str = "blue";
pub const PLY_PROP_COLOR_A: &str = "alpha";
pub const PLY_PROP_COLOR_DIFF_R: &str = "diffuse_red";
pub const PLY_PROP_COLOR_DIFF_G: &str = "diffuse_green";
pub const PLY_PROP_COLOR_DIFF_B: &str = "diffuse_blue";
pub const PLY_PROP_COLOR_DIFF_A: &str = "diffuse_alpha";
pub const PLY_PROP_INDICES: &str = "vertex_index";
pub const PLY_ELEMENT_VERTICES: &str = "vertex";
pub const PLY_ELEMENT_FACES: &str = "face";

/// `ply_prop_` values
pub const PLY_PROP_INT: u8 = 1;
pub const PLY_PROP_UINT: u8 = 2;
pub const PLY_PROP_DECIMAL: u8 = 3;

#[derive(Debug, Clone, Default)]
pub struct PlyProp {
    pub bytes: u8,
    /// follows `PLY_PROP_*`
    pub type_: u8,
    pub list_bytes: u8,
    pub list_type: u8,
    pub offset: u16,
    /// at most 31 bytes (C: `char name[32]`)
    pub name: String,
}

#[derive(Debug, Clone, Default)]
pub struct PlyElement {
    /// at most 63 bytes (C: `char name[64]`)
    pub name: String,
    pub count: i32,
    pub properties: Vec<PlyProp>,
    pub data: Vec<u8>,
    pub data_stride: i32,
    /// `None` unless the element is a single list property (C: NULL)
    pub list_data: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Default)]
pub struct PlyFile {
    pub elements: Vec<PlyElement>,
}

impl PlyFile {
    /// C `count`
    pub fn count(&self) -> i32 {
        self.elements.len() as i32
    }
}

/// `ply_map_t`
#[derive(Debug, Clone)]
pub struct PlyMap<'a> {
    pub name: &'a str,
    pub to_type: u8,
    pub to_size: u8,
    pub to_offset: u16,
    pub default_val: Option<&'a [u8]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyFmt {
    Ascii,
    BinaryLe,
    BinaryBe,
}

fn read_n<const N: usize>(src: &[u8]) -> [u8; N] {
    let mut b = [0u8; N];
    for (d, s) in b.iter_mut().zip(src) {
        *d = *s;
    }
    b
}

/// `_ply_convert`: converts one value between PLY property types; `dest`/`src` are sized `dest_size`/`src_size`.
fn ply_convert_value(dest: &mut [u8], dest_size: u8, dest_type: u8, src: &[u8], src_size: u8, src_type: u8) {
    let (ds, ss) = (usize::from(dest_size), usize::from(src_size));
    if dest.len() < ds || src.len() < ss {
        return;
    }

    if dest_size == src_size && src_type == dest_type {
        dest[..ds].copy_from_slice(&src[..ds]);
        return;
    }

    let mut dval = 0.0f64;
    let mut ival = 0i64;
    if src_type == PLY_PROP_DECIMAL {
        if src_size == 4 {
            dval = f64::from(f32::from_le_bytes(read_n(src)));
        } else {
            dval = f64::from_le_bytes(read_n(src));
        }
    } else if src_type == PLY_PROP_INT {
        match src_size {
            1 => ival = i64::from(i8::from_le_bytes(read_n(src))),
            2 => ival = i64::from(i16::from_le_bytes(read_n(src))),
            4 => ival = i64::from(i32::from_le_bytes(read_n(src))),
            8 => ival = i64::from_le_bytes(read_n(src)),
            _ => {}
        }
    } else {
        match src_size {
            1 => ival = i64::from(u8::from_le_bytes(read_n(src))),
            2 => ival = i64::from(u16::from_le_bytes(read_n(src))),
            4 => ival = i64::from(u32::from_le_bytes(read_n(src))),
            8 => ival = u64::from_le_bytes(read_n(src)) as i64,
            _ => {}
        }
    }

    if dest_type == PLY_PROP_DECIMAL {
        if src_type != PLY_PROP_DECIMAL {
            dval = ival as f64;
        }
        if dest_size == 4 {
            dest[..4].copy_from_slice(&(dval as f32).to_le_bytes());
        } else {
            dest[..8].copy_from_slice(&dval.to_le_bytes());
        }
    } else {
        // int and uint truncate to the same bits, so only size matters
        if src_type == PLY_PROP_DECIMAL {
            ival = dval as i64;
        }
        match dest_size {
            1 => dest[..1].copy_from_slice(&(ival as u8).to_le_bytes()),
            2 => dest[..2].copy_from_slice(&(ival as u16).to_le_bytes()),
            4 => dest[..4].copy_from_slice(&(ival as u32).to_le_bytes()),
            _ => {}
        }
    }
}

/// C string view starting at `pos`: bytes up to the first NUL (or the end of the data).
fn cstr(data: &[u8], pos: usize) -> &[u8] {
    let tail = data.get(pos..).unwrap_or(&[]);
    &tail[..tail.iter().position(|&c| c == 0).unwrap_or(tail.len())]
}

fn starts_with(data: &[u8], pos: usize, prefix: &str) -> bool {
    cstr(data, pos).starts_with(prefix.as_bytes())
}

/// `strchr(data + pos, '\n')`
fn find_newline(data: &[u8], pos: usize) -> Option<usize> {
    cstr(data, pos).iter().position(|&c| c == b'\n').map(|i| pos + i)
}

/// `_ply_get_word`: reads at most `out_size - 1` bytes up to a separator.
fn get_word(data: &[u8], pos: usize, out_size: usize) -> Vec<u8> {
    cstr(data, pos)
        .iter()
        .take_while(|&&c| c != b' ' && c != b'\t' && c != b'\n' && c != b'\r')
        .take(out_size - 1)
        .copied()
        .collect()
}

fn type_info(word: &[u8]) -> Option<(u8, u8)> {
    Some(match word {
        b"char" => (PLY_PROP_INT, 1),
        b"uchar" => (PLY_PROP_UINT, 1),
        b"short" => (PLY_PROP_INT, 2),
        b"ushort" => (PLY_PROP_UINT, 2),
        b"int" => (PLY_PROP_INT, 4),
        b"uint" => (PLY_PROP_UINT, 4),
        b"float" => (PLY_PROP_DECIMAL, 4),
        b"double" => (PLY_PROP_DECIMAL, 8),
        _ => return None,
    })
}

fn word_string(word: &[u8]) -> String {
    String::from_utf8_lossy(word).into_owned()
}

/// Allocates `n` zero bytes, failing like `malloc` would for absurd sizes.
fn alloc(n: usize) -> Option<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).ok()?;
    v.resize(n, 0);
    Some(v)
}

/// Converts an ASCII token into a property value of the given type (C: `atof`/`atol` + `_ply_convert`).
fn convert_ascii(dest: &mut [u8], bytes: u8, type_: u8, word: &[u8]) {
    if type_ == PLY_PROP_DECIMAL {
        let val = cstd::atof(word);
        ply_convert_value(dest, bytes, type_, &val.to_le_bytes(), 8, PLY_PROP_DECIMAL);
    } else if type_ == PLY_PROP_INT {
        let val = cstd::strtol(word).0;
        ply_convert_value(dest, bytes, type_, &val.to_le_bytes(), 8, PLY_PROP_INT);
    } else {
        let val = cstd::strtol(word).0 as u64;
        ply_convert_value(dest, bytes, type_, &val.to_le_bytes(), 8, PLY_PROP_UINT);
    }
}

/// `ply_read`
pub fn ply_read(file: &[u8]) -> Option<PlyFile> {
    // Check file signature
    if !starts_with(file, 0, "ply") {
        return None;
    }

    // File data
    let mut format = PlyFmt::Ascii;
    let mut out_file = PlyFile::default();

    // Read the header
    let mut line = find_newline(file, 0);
    loop {
        let mut pos = line? + 1;

        if starts_with(file, pos, "format ") {
            let word = get_word(file, pos + 7, 128);
            match &word[..] {
                b"ascii" => format = PlyFmt::Ascii,
                b"binary_little_endian" => format = PlyFmt::BinaryLe,
                b"binary_big_endian" => format = PlyFmt::BinaryBe,
                _ => {}
            }
        } else if starts_with(file, pos, "comment ") {
        } else if starts_with(file, pos, "element ") {
            let name = get_word(file, pos + 8, 64);
            let word = get_word(file, pos + 9 + name.len(), 128);

            out_file.elements.push(PlyElement {
                name: word_string(&name),
                count: cstd::atoi(&word),
                ..Default::default()
            });
        } else if starts_with(file, pos, "property ") {
            let mut prop = PlyProp::default();
            let word = get_word(file, pos + 9, 128);

            if word == b"list" {
                let mut off = 10 + word.len();
                let word = get_word(file, pos + off, 128);
                if let Some((t, b)) = type_info(&word) {
                    (prop.type_, prop.bytes) = (t, b);
                }
                off += word.len() + 1;
                let word = get_word(file, pos + off, 128);
                if let Some((t, b)) = type_info(&word) {
                    (prop.list_type, prop.list_bytes) = (t, b);
                }
                off += word.len() + 1;
                prop.name = word_string(&get_word(file, pos + off, 32));
            } else {
                if let Some((t, b)) = type_info(&word) {
                    (prop.type_, prop.bytes) = (t, b);
                }
                prop.name = word_string(&get_word(file, pos + 10 + word.len(), 32));
            }

            // C indexes elements[count - 1]; a property before any element is undefined behavior
            let el = out_file.elements.last_mut()?;
            prop.offset = el.data_stride as u16;
            el.data_stride += i32::from(prop.bytes);
            el.properties.push(prop);
        } else if starts_with(file, pos, "end_header") {
            pos = find_newline(file, pos)? + 1;
            line = Some(pos);
            break;
        }
        line = find_newline(file, pos);
    }

    if format == PlyFmt::BinaryBe {
        return None;
    }

    let mut line = line?;

    // Parse the data
    for el in out_file.elements.iter_mut() {
        let total = (el.data_stride as isize as usize).checked_mul(el.count as isize as usize)?;
        el.data = alloc(total)?;

        let property_count = el.properties.len();

        // If it's a list type
        if property_count == 1 && el.properties[0].list_type != 0 {
            let p = el.properties[0].clone();
            let pbytes = usize::from(p.bytes);
            let lbytes = usize::from(p.list_bytes);

            let mut list_data: Vec<u8> = Vec::new();
            let mut data_pos = 0usize;

            if format == PlyFmt::Ascii {
                for _ in 0..el.count {
                    let mut off = 0;
                    let word = get_word(file, line + off, 128);
                    off += word.len() + 1;
                    let count = cstd::atoi(&word);
                    ply_convert_value(
                        el.data.get_mut(data_pos..)?,
                        p.bytes,
                        p.type_,
                        &count.to_le_bytes(),
                        4,
                        PLY_PROP_INT,
                    );
                    for _ in 0..count {
                        let word = get_word(file, line + off, 128);
                        off += word.len() + 1;
                        let start = list_data.len();
                        list_data.resize(start + lbytes, 0);
                        convert_ascii(&mut list_data[start..], p.list_bytes, p.list_type, &word);
                    }
                    line = find_newline(file, line)? + 1;
                    data_pos += pbytes;
                }
            } else {
                for _ in 0..el.count {
                    // Get the count of items in this element
                    let src = file.get(line..line + pbytes)?;
                    el.data.get_mut(data_pos..data_pos + pbytes)?.copy_from_slice(src);
                    let mut count = [0u8; 4];
                    ply_convert_value(&mut count, 4, PLY_PROP_INT, src, p.bytes, p.type_);
                    let count = i32::from_le_bytes(count);
                    line += pbytes;
                    data_pos += pbytes;

                    // Copy the elements
                    let n = lbytes.checked_mul(usize::try_from(count).ok()?)?;
                    list_data.extend_from_slice(file.get(line..line.checked_add(n)?)?);
                    line += n;
                }
            }

            el.list_data = Some(list_data);
        } else if format == PlyFmt::Ascii {
            let stride = el.data_stride as usize;
            for e in 0..el.count as usize {
                let mut off = 0;
                for p in &el.properties {
                    let word = get_word(file, line + off, 128);
                    off += word.len() + 1;
                    let start = e * stride + usize::from(p.offset);
                    convert_ascii(el.data.get_mut(start..)?, p.bytes, p.type_, &word);
                }
                line = find_newline(file, line)? + 1;
            }
        } else {
            let copy = el.data.len();
            el.data.copy_from_slice(file.get(line..line.checked_add(copy)?)?);
            line += copy;
        }
    }

    Some(out_file)
}

/// `ply_convert`: converts the named element into `format_stride`-sized items laid out by `to_format`.
/// Returns the converted bytes and the item count (`None` if the element doesn't exist).
pub fn ply_convert(file: &PlyFile, element_name: &str, to_format: &[PlyMap], format_stride: i32) -> Option<(Vec<u8>, i32)> {
    // Find the elements we want to convert by name
    let elements = file.elements.iter().find(|e| e.name == element_name)?;
    let stride = usize::try_from(format_stride).ok()?;

    match &elements.list_data {
        None => {
            // Map the element properties to the out format
            let map: Vec<Option<&PlyProp>> = to_format
                .iter()
                .map(|f| elements.properties.iter().find(|p| p.name == f.name))
                .collect();

            // Now convert and copy each item
            let count = usize::try_from(elements.count).ok()?;
            let mut out = alloc(count.checked_mul(stride)?)?;
            let src_stride = elements.data_stride as usize;

            for i in 0..count {
                let dest = &mut out[i * stride..];
                let src = elements.data.get(i * src_stride..).unwrap_or(&[]);
                for (f, m) in to_format.iter().zip(&map) {
                    let to = usize::from(f.to_offset);
                    match m {
                        None => {
                            if let Some(d) = f.default_val {
                                let n = usize::from(f.to_size).min(d.len());
                                if let Some(dst) = dest.get_mut(to..to + n) {
                                    dst.copy_from_slice(&d[..n]);
                                }
                            }
                        }
                        Some(prop) => {
                            let s = src.get(usize::from(prop.offset)..).unwrap_or(&[]);
                            if let Some(dst) = dest.get_mut(to..) {
                                ply_convert_value(dst, f.to_size, f.to_type, s, prop.bytes, prop.type_);
                            }
                        }
                    }
                }
            }

            Some((out, elements.count))
        }
        Some(list_data) => {
            let p0 = elements.properties.first()?;
            let (src_size, src_type) = (p0.bytes, p0.type_);
            let (src_ind_size, src_ind_type) = (usize::from(p0.list_bytes), p0.list_type);
            let f0 = to_format.first()?;
            let to_size = usize::from(f0.to_size);

            let read_ct = |i: usize| {
                let mut ct = [0u8; 4];
                let s = elements.data.get(i * usize::from(src_size)..).unwrap_or(&[]);
                ply_convert_value(&mut ct, 4, PLY_PROP_INT, s, src_size, src_type);
                i32::from_le_bytes(ct)
            };

            // Count how many total indices there will be
            let element_count = usize::try_from(elements.count).ok()?;
            let mut count: i32 = 0;
            for i in 0..element_count {
                count = count.wrapping_add(3 + (read_ct(i) - 3) * 3);
            }

            let mut out = alloc(usize::try_from(count).ok()?.checked_mul(stride)?)?;

            let mut src_ind = 0usize;
            let mut dest = 0usize;
            for i in 0..element_count {
                let ct = read_ct(i);
                for x in 0..(ct - 2).max(0) as usize {
                    // fan triangulation: corners 0, x + 1, x + 2
                    for k in 0..3 {
                        let corner = if k == 0 { 0 } else { x + k };
                        let s = list_data.get(src_ind + corner * src_ind_size..).unwrap_or(&[]);
                        if let Some(d) = out.get_mut(dest + (x * 3 + k) * to_size..) {
                            ply_convert_value(d, f0.to_size, f0.to_type, s, src_ind_size as u8, src_ind_type);
                        }
                    }
                }
                src_ind += src_ind_size * ct.max(0) as usize;
                dest = dest.wrapping_add((stride as isize * (3 + (ct as isize - 3) * 3)) as usize);
            }

            Some((out, count))
        }
    }
}

/// `ply_free`
pub fn ply_free(file: PlyFile) {
    drop(file);
}
