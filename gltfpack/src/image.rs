//! Image helpers (`image.cpp`): reading image data, header parsing, alpha detection and resizing rules.

use crate::cgltf;
use crate::fileio::{get_extension, get_full_path, read_file};

const MIME_TYPES: [(&str, &str); 5] = [
    ("image/jpeg", ".jpg"),
    ("image/jpeg", ".jpeg"),
    ("image/png", ".png"),
    ("image/ktx2", ".ktx2"),
    ("image/webp", ".webp"),
];

const PNG_SIGNATURE: &[u8] = b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a";

fn infer_mime_type(path: &str) -> &'static str {
    let ext = get_extension(path);

    MIME_TYPES.iter().find(|(_, e)| ext == *e).map_or("", |(m, _)| m)
}

/// `std::string::compare(pos, len, str) == 0`
fn compare(data: &[u8], pos: usize, len: usize, s: &[u8]) -> bool {
    let end = (pos + len).min(data.len());
    data.get(pos..end) == Some(s)
}

fn parse_data_uri(uri: &str) -> Option<(String, Vec<u8>)> {
    if !uri.starts_with("data:") {
        return None;
    }

    let comma = uri.find(',')?;

    if comma < 7 || &uri[comma - 7..comma] != ";base64" {
        return None;
    }

    let base64 = &uri.as_bytes()[comma + 1..];
    let base64_size = base64.len();
    let mut size = base64_size - base64_size / 4;

    if base64_size >= 2 {
        size -= usize::from(base64[base64_size - 2] == b'=');
        size -= usize::from(base64[base64_size - 1] == b'=');
    }

    let data = cgltf::load_buffer_base64(size, base64).ok()?;

    Some((uri[5..comma - 7].to_string(), data))
}

fn fixup_mime_type(data: &[u8], mime_type: &str) -> String {
    if mime_type == "image/jpeg" && compare(data, 0, 8, PNG_SIGNATURE) {
        return "image/png".to_string();
    }

    if mime_type == "image/png" && compare(data, 0, 3, b"\xff\xd8\xff") {
        return "image/jpeg".to_string();
    }

    mime_type.to_string()
}

/// `readImage`: returns the image bytes and mime type.
pub fn read_image(data: &cgltf::Data, image: &cgltf::Image, input_path: Option<&str>) -> Option<(Vec<u8>, String)> {
    if let Some(uri) = &image.uri
        && let Some((mime_type, bytes)) = parse_data_uri(uri)
    {
        let mime_type = fixup_mime_type(&bytes, &mime_type);
        return Some((bytes, mime_type));
    }

    if let Some(view) = image.buffer_view.map(|v| &data.buffer_views[v])
        && let Some(buffer) = data.buffers.get(view.buffer).and_then(|b| b.data.as_deref())
        && let Some(image_mime) = &image.mime_type
    {
        let bytes = buffer.get(view.offset..view.offset.checked_add(view.size)?)?.to_vec();
        let mime_type = fixup_mime_type(&bytes, image_mime);
        return Some((bytes, mime_type));
    }

    if let Some(uri) = image.uri.as_deref().filter(|u| !u.is_empty())
        && let Some(input_path) = input_path
    {
        let path = cgltf::decode_uri(uri);
        // C++ truncates the decoded path at the first NUL
        let path = path.split('\0').next().unwrap_or("").to_string();

        let bytes = read_file(&get_full_path(&path, input_path))?;

        let mime_type = image.mime_type.clone().unwrap_or_else(|| infer_mime_type(&path).to_string());
        let mime_type = fixup_mime_type(&bytes, &mime_type);
        return Some((bytes, mime_type));
    }

    None
}

fn byte(data: &[u8], offset: usize) -> u32 {
    u32::from(data.get(offset).copied().unwrap_or(0))
}

fn read_int16_be(data: &[u8], offset: usize) -> i32 {
    (byte(data, offset) * 256 + byte(data, offset + 1)) as i32
}

fn read_int32_be(data: &[u8], offset: usize) -> i32 {
    ((byte(data, offset) << 24) | (byte(data, offset + 1) << 16) | (byte(data, offset + 2) << 8) | byte(data, offset + 3)) as i32
}

fn read_int32_le(data: &[u8], offset: usize) -> i32 {
    (byte(data, offset) | (byte(data, offset + 1) << 8) | (byte(data, offset + 2) << 16) | (byte(data, offset + 3) << 24)) as i32
}

// https://en.wikipedia.org/wiki/PNG#File_format
fn get_dimensions_png(data: &[u8]) -> Option<(i32, i32)> {
    if data.len() < 8 + 8 + 13 + 4 {
        return None;
    }

    if !compare(data, 0, 8, PNG_SIGNATURE) || !compare(data, 12, 4, b"IHDR") {
        return None;
    }

    Some((read_int32_be(data, 16), read_int32_be(data, 20)))
}

// https://en.wikipedia.org/wiki/JPEG_File_Interchange_Format#File_format_structure
fn get_dimensions_jpeg(data: &[u8]) -> Option<(i32, i32)> {
    let mut offset = 0usize;

    // note, this can stop parsing before reaching the end but we stop at SOF anyway
    while offset + 4 <= data.len() {
        if data[offset] != 0xff {
            return None;
        }

        let marker = data[offset + 1];

        if marker == 0xff {
            offset += 1;
            continue; // padding
        }

        // d0..d9 correspond to SOI, RSTn, EOI
        if marker == 0 || (0xd0..=0xd9).contains(&marker) {
            offset += 2;
            continue; // no payload
        }

        // c0/c2 correspond to SOF0/SOF2
        if marker == 0xc0 || marker == 0xc2 {
            if offset + 10 > data.len() {
                return None;
            }

            return Some((read_int16_be(data, offset + 7), read_int16_be(data, offset + 5)));
        }

        offset += 2 + read_int16_be(data, offset + 2) as usize;
    }

    None
}

// https://en.wikipedia.org/wiki/PNG#File_format
fn has_transparency_png(data: &[u8]) -> bool {
    if data.len() < 8 + 8 + 13 + 4 {
        return false;
    }

    if !compare(data, 0, 8, PNG_SIGNATURE) || !compare(data, 12, 4, b"IHDR") {
        return false;
    }

    let ctype = i32::from(data[25] as i8);

    if ctype != 3 {
        return ctype == 4 || ctype == 6;
    }

    let mut offset = 8usize; // reparse IHDR chunk for simplicity

    while offset + 12 <= data.len() {
        let length = read_int32_be(data, offset);

        if length < 0 {
            return false;
        }

        if compare(data, offset + 4, 4, b"tRNS") {
            return true;
        }

        offset += 12 + length as usize;
    }

    false
}

// https://github.khronos.org/KTX-Specification/ktxspec.v2.html
fn has_transparency_ktx2(data: &[u8]) -> bool {
    if data.len() < 12 + 17 * 4 {
        return false;
    }

    if !compare(data, 0, 12, b"\xabKTX 20\xbb\r\n\x1a\n") {
        return false;
    }

    let dfd_offset = read_int32_le(data, 48);
    let dfd_length = read_int32_le(data, 52);

    if dfd_length < 4 + 24 + 16 || dfd_offset as u32 as usize > data.len() || dfd_length as u32 as usize > data.len() - dfd_offset as usize
    {
        return false;
    }

    const KDF_DF_MODEL_ETC1S: i32 = 163;
    const KDF_DF_MODEL_UASTC: i32 = 166;
    const KHR_DF_CHANNEL_UASTC_RGBA: i32 = 3;
    const KHR_DF_CHANNEL_ETC1S_RGB: i32 = 0;
    const KHR_DF_CHANNEL_ETC1S_AAA: i32 = 15;

    let dfd = dfd_offset as usize;
    let color_model = read_int32_le(data, dfd + 12) & 0xff;
    let channel_type1 = (read_int32_le(data, dfd + 28) >> 24) & 0xf;
    let channel_type2 = if dfd_length >= 4 + 24 + 16 * 2 {
        (read_int32_le(data, dfd + 44) >> 24) & 0xf
    } else {
        channel_type1
    };

    if color_model == KDF_DF_MODEL_ETC1S {
        channel_type1 == KHR_DF_CHANNEL_ETC1S_RGB && channel_type2 == KHR_DF_CHANNEL_ETC1S_AAA
    } else if color_model == KDF_DF_MODEL_UASTC {
        channel_type1 == KHR_DF_CHANNEL_UASTC_RGBA
    } else {
        false
    }
}

// https://developers.google.com/speed/webp/docs/riff_container
fn has_transparency_webp(data: &[u8]) -> bool {
    if data.len() < 12 + 4 + 12 {
        return false;
    }

    if !compare(data, 0, 4, b"RIFF") || !compare(data, 8, 4, b"WEBP") {
        return false;
    }

    // WebP data may use VP8L, VP8X or VP8 format, but VP8 does not support transparency
    if compare(data, 12, 4, b"VP8L") {
        if data[20] != 0x2f {
            return false;
        }

        // width (14) | height (14) | alpha_is_used (1) | version_number(3)
        let header = read_int32_le(data, 21) as u32;
        (header & (1 << 28)) != 0
    } else if compare(data, 12, 4, b"VP8X") {
        // zero (2) | icc (1) | alpha (1) | exif (1) | xmp (1) | animation (1) | zero (1)
        (data[20] & (1 << 4)) != 0
    } else {
        false
    }
}

pub fn has_alpha(data: &[u8], mime_type: &str) -> bool {
    match mime_type {
        "image/png" => has_transparency_png(data),
        "image/ktx2" => has_transparency_ktx2(data),
        "image/webp" => has_transparency_webp(data),
        _ => false,
    }
}

/// `getDimensions`: (width, height) from PNG/JPEG headers.
pub fn get_dimensions(data: &[u8], mime_type: &str) -> Option<(i32, i32)> {
    match mime_type {
        "image/png" => get_dimensions_png(data),
        "image/jpeg" => get_dimensions_jpeg(data),
        _ => None,
    }
}

fn round_pow2(value: i32) -> i32 {
    let mut result: i32 = 1;

    while result < value {
        result = result.wrapping_shl(1);
    }

    // to prevent odd texture sizes from increasing the size too much, we round to nearest power of 2 above a certain size
    if value > 128 && result.wrapping_mul(3) / 4 > value {
        result >>= 1;
    }

    result
}

fn round_block(value: i32, pow2: bool) -> i32 {
    if value == 0 {
        return 4;
    }

    if pow2 && value > 4 {
        return round_pow2(value);
    }

    value.wrapping_add(3) & !3
}

pub fn adjust_dimensions(width: &mut i32, height: &mut i32, scale: f32, limit: i32, pow2: bool) {
    *width = crate::cxx::f2i32(*width as f32 * scale);
    *height = crate::cxx::f2i32(*height as f32 * scale);

    if limit != 0 && (*width > limit || *height > limit) {
        let limit_scale = limit as f32 / (if *width > *height { *width } else { *height }) as f32;

        *width = crate::cxx::f2i32(*width as f32 * limit_scale);
        *height = crate::cxx::f2i32(*height as f32 * limit_scale);
    }

    *width = round_block(*width, pow2);
    *height = round_block(*height, pow2);
}

pub fn mime_extension(mime_type: &str) -> &'static str {
    MIME_TYPES.iter().find(|(m, _)| *m == mime_type).map_or(".raw", |(_, e)| e)
}
