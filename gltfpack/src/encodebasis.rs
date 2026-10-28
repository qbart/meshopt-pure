//! KTX2/BasisU texture encoding (`encodebasis.cpp`).
//!
//! C++ gltfpack hands each image to `basisu::basis_compressor` with `m_create_ktx2_file`. The
//! pure Rust `ktx` crate ports the compressor core, so the parts it leaves out are done here,
//! following basisu 2.10: source image loading, y flip, resampling (`image_resample`), mipmap
//! generation (`generate_mipmaps`) and the KTX2 container (`create_ktx2_file`).

use ktx::basisu::basis_file::{BASIS_SLICE_DESC_SIZE, BasisFileHeader, BasisSliceDesc};
use ktx::basisu::comp::{BasisCompressor, BasisCompressorErrorCode, BasisCompressorParams, Image};
use ktx::basisu::resampler::{BoundaryOp, Resampler};
use ktx::basisu::uastc_enc::{
    C_PACK_UASTC_LEVEL_DEFAULT, C_PACK_UASTC_LEVEL_FASTER, C_PACK_UASTC_LEVEL_FASTEST, C_PACK_UASTC_LEVEL_MASK, C_PACK_UASTC_LEVEL_SLOWER,
    C_PACK_UASTC_LEVEL_VERY_SLOW,
};
use ktx::jpgd::{JpegDecoder, decompress_jpeg_image_from_memory};
use zstd_pure_rs::prelude as zstd;

use crate::cgltf::Data;
use crate::image::{adjust_dimensions, get_dimensions, read_image};
use crate::types::*;

/// Whether BasisU encoding is available (C++: `WITH_BASISU`).
pub const SUPPORTED: bool = true;

struct BasisSettings {
    etc1s_l: i32,
    etc1s_q: i32,
    uastc_l: usize,
    uastc_q: f32,
}

const fn bs(etc1s_l: i32, etc1s_q: i32, uastc_l: usize, uastc_q: f32) -> BasisSettings {
    BasisSettings {
        etc1s_l,
        etc1s_q,
        uastc_l,
        uastc_q,
    }
}

const BASIS_SETTINGS: [BasisSettings; 10] = [
    bs(1, 1, 0, 4.0),
    bs(1, 32, 0, 3.0),
    bs(1, 64, 1, 2.0),
    bs(1, 96, 1, 1.5),
    bs(1, 128, 1, 1.0), // quality arguments aligned with basisu defaults
    bs(1, 150, 1, 0.8),
    bs(1, 170, 1, 0.6),
    bs(1, 192, 1, 0.4), // gltfpack defaults
    bs(1, 224, 2, 0.2),
    bs(1, 255, 2, 0.0),
];

const MAX_DIMENSION: u32 = 16384;

/// `basisu::image` as RGBA rows; new images are filled with `g_black_color`.
#[derive(Clone)]
struct Rgba {
    width: u32,
    height: u32,
    pixels: Vec<[u8; 4]>,
}

impl Rgba {
    fn new(width: u32, height: u32) -> Rgba {
        Rgba {
            width,
            height,
            pixels: vec![[0, 0, 0, 255]; width as usize * height as usize],
        }
    }

    fn from_bytes(data: &[u8], width: u32, height: u32) -> Option<Rgba> {
        let count = (width as usize).checked_mul(height as usize)?;
        let bytes = data.get(..count.checked_mul(4)?)?;
        let pixels = bytes.as_chunks::<4>().0.to_vec();
        Some(Rgba { width, height, pixels })
    }

    fn has_alpha(&self) -> bool {
        self.pixels.iter().any(|p| p[3] < 255)
    }

    fn flip_y(&mut self) {
        let w = self.width as usize;
        let h = self.height as usize;
        for y in 0..h / 2 {
            let (top, bottom) = self.pixels.split_at_mut((h - 1 - y) * w);
            top[y * w..(y + 1) * w].swap_with_slice(&mut bottom[..w]);
        }
    }

    fn to_image(&self) -> Image {
        let mut image = Image::with_size(self.width, self.height);
        image.get_pixels_mut().copy_from_slice(&self.pixels);
        image
    }
}

/// `load_png` / `load_jpg` (4 components).
fn load_image(data: &[u8], mime_type: &str) -> Option<Rgba> {
    if mime_type == "image/png" {
        if data.is_empty() {
            return None;
        }
        let (pixels, width, height) = ktx::lodepng::lodepng_decode32(data).ok()?;
        Rgba::from_bytes(&pixels, width, height)
    } else {
        let (pixels, width, height, _) = decompress_jpeg_image_from_memory(data, 4, JpegDecoder::C_FLAG_LINEAR_CHROMA_FILTERING)?;
        Rgba::from_bytes(&pixels, u32::try_from(width).ok()?, u32::try_from(height).ok()?)
    }
}

fn saturate(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

fn linear_to_srgb(l: f32) -> f32 {
    if l < 0.0031308 {
        saturate(l * 12.92)
    } else {
        saturate(1.055 * l.powf(1.0 / 2.4) - 0.055)
    }
}

fn srgb_to_linear(s: f32) -> f32 {
    if s < 0.04045 {
        saturate(s * (1.0 / 12.92))
    } else {
        saturate(((s + 0.055) * (1.0 / 1.055)).powf(2.4))
    }
}

/// `image_resample` with `filter_scale = 1` and `first_comp = 0`: resamples `src` into `dst`.
fn image_resample(src: &Rgba, dst: &mut Rgba, srgb: bool, filter: &str, wrapping: bool, num_comps: usize) -> bool {
    const LINEAR_TO_SRGB_TABLE_SIZE: usize = 8192;

    let (src_w, src_h) = (src.width, src.height);
    let (dst_w, dst_h) = (dst.width, dst.height);

    if src_w.max(src_h) > MAX_DIMENSION || src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 || dst_w.max(dst_h) > MAX_DIMENSION {
        return false;
    }

    if src_w == dst_w && src_h == dst_h {
        *dst = src.clone();
        return true;
    }

    let mut srgb_to_linear_table = [0f32; 256];
    let mut linear_to_srgb_table = [0u8; LINEAR_TO_SRGB_TABLE_SIZE];
    if srgb {
        for (i, v) in srgb_to_linear_table.iter_mut().enumerate() {
            *v = srgb_to_linear(i as f32 * (1.0 / 255.0));
        }
        for (i, v) in linear_to_srgb_table.iter_mut().enumerate() {
            let s = linear_to_srgb(i as f32 * (1.0 / (LINEAR_TO_SRGB_TABLE_SIZE - 1) as f32));
            *v = ((255.0 * s + 0.5) as i32).clamp(0, 255) as u8;
        }
    }

    let boundary = if wrapping { BoundaryOp::Wrap } else { BoundaryOp::Clamp };
    let new_resampler = |clist_x, clist_y| {
        Resampler::new(
            src_w as i32,
            src_h as i32,
            dst_w as i32,
            dst_h as i32,
            boundary,
            0.0,
            1.0,
            Some(filter),
            clist_x,
            clist_y,
            1.0,
            1.0,
            0.0,
            0.0,
        )
    };

    let mut resamplers = vec![new_resampler(None, None)];
    let (clist_x, clist_y) = (resamplers[0].get_clist_x(), resamplers[0].get_clist_y());
    for _ in 1..num_comps {
        resamplers.push(new_resampler(clist_x.clone(), clist_y.clone()));
    }

    let mut samples = vec![vec![0f32; src_w as usize]; num_comps];
    let mut dst_y = 0usize;

    for src_y in 0..src_h as usize {
        let row = &src.pixels[src_y * src_w as usize..(src_y + 1) * src_w as usize];

        // Put source lines into resampler(s)
        for (x, p) in row.iter().enumerate() {
            for (c, s) in samples.iter_mut().enumerate() {
                let v = p[c];
                s[x] = if !srgb || c == 3 {
                    v as f32 * (1.0 / 255.0)
                } else {
                    srgb_to_linear_table[v as usize]
                };
            }
        }

        for (r, s) in resamplers.iter_mut().zip(&samples) {
            if !r.put_line(s) {
                return false;
            }
        }

        // Now retrieve any output lines
        'lines: loop {
            for (c, r) in resamplers.iter_mut().enumerate() {
                let Some(output) = r.get_line() else {
                    break 'lines;
                };

                let linear = !srgb || c == 3;
                let Some(out_row) = dst.pixels.get_mut(dst_y * dst_w as usize..(dst_y + 1) * dst_w as usize) else {
                    return false;
                };

                for (d, &v) in out_row.iter_mut().zip(output) {
                    d[c] = if linear {
                        ((255.0 * v + 0.5) as i32).clamp(0, 255) as u8
                    } else {
                        let j = ((LINEAR_TO_SRGB_TABLE_SIZE - 1) as f32 * v + 0.5) as i32;
                        linear_to_srgb_table[j.clamp(0, LINEAR_TO_SRGB_TABLE_SIZE as i32 - 1) as usize]
                    };
                }
            }

            dst_y += 1;
        }
    }

    true
}

/// `generate_mipmaps` (kaiser filter, wrapping, fast mips): levels 1.. of `img`.
fn generate_mipmaps(img: &Rgba, srgb: bool, has_alpha: bool) -> Option<Vec<Rgba>> {
    let mut total_levels = 1;
    let (mut w, mut h) = (img.width, img.height);
    while w.max(h) > 1 {
        w = (w >> 1).max(1);
        h = (h >> 1).max(1);
        total_levels += 1;
    }

    let mut mips: Vec<Rgba> = Vec::new();
    for level in 1..total_levels {
        let mut level_img = Rgba::new((img.width >> level).max(1), (img.height >> level).max(1));
        let source = if level > 1 { mips.last()? } else { img };

        if !image_resample(source, &mut level_img, srgb, "kaiser", true, if has_alpha { 4 } else { 3 }) {
            return None;
        }

        mips.push(level_img);
    }

    Some(mips)
}

struct Job {
    data: Vec<u8>,
    mime_type: String,
    width: i32,
    height: i32,
    uastc: bool,
    settings: &'static BasisSettings,
    srgb: bool,
    normal_map: bool,
    flipy: bool,
}

/// `fillParams` (compressor settings; the source images are supplied by `encode`).
fn fill_params(params: &mut BasisCompressorParams, job: &Job) {
    let bs = job.settings;

    if job.uastc {
        const LEVEL_FLAGS: [u32; 5] = [
            C_PACK_UASTC_LEVEL_FASTEST,
            C_PACK_UASTC_LEVEL_FASTER,
            C_PACK_UASTC_LEVEL_DEFAULT,
            C_PACK_UASTC_LEVEL_SLOWER,
            C_PACK_UASTC_LEVEL_VERY_SLOW,
        ];

        params.uastc = true;

        params.pack_uastc_ldr_4x4_flags &= !C_PACK_UASTC_LEVEL_MASK;
        params.pack_uastc_ldr_4x4_flags |= LEVEL_FLAGS[bs.uastc_l];

        params.rdo_uastc_ldr_4x4 = bs.uastc_q > 0.0;
        params.rdo_uastc_ldr_4x4_quality_scalar.set(bs.uastc_q);
        params.rdo_uastc_ldr_4x4_dict_size.set(1024);
    } else {
        params.etc1s_compression_level.set(bs.etc1s_l);
        params.quality_level = bs.etc1s_q;
        params.etc1s_max_endpoint_clusters = 0;
        params.etc1s_max_selector_clusters = 0;

        params.no_selector_rdo = job.normal_map;
        params.no_endpoint_rdo = job.normal_map;
    }

    params.perceptual = job.srgb;

    params.mip_gen = true;
    params.mip_srgb = job.srgb;

    params.ktx2_and_basis_srgb_transfer_function = job.srgb;

    params.status_output = false;

    // basis_parallel_compress runs every compressor on a single threaded job pool
    params.job_pool_threads = 1;
    params.multithreading = true;
}

/// `basis_compressor::read_source_images` for one source file with generated mipmaps.
fn read_source_images(job: &Job) -> Option<(Vec<Rgba>, bool)> {
    let mut file_image = load_image(&job.data, &job.mime_type)?;
    let has_alpha = file_image.has_alpha();

    if job.flipy {
        file_image.flip_y();
    }

    if job.width > 0 && job.height > 0 {
        let mut temp_img = Rgba::new((job.width as u32).min(MAX_DIMENSION), (job.height as u32).min(MAX_DIMENSION));
        // the result is ignored by basisu as well; a failed resample leaves a black image
        image_resample(&file_image, &mut temp_img, job.srgb, "box", false, 4);
        file_image = temp_img;
    }

    if file_image.width == 0 || file_image.height == 0 || file_image.width > MAX_DIMENSION || file_image.height > MAX_DIMENSION {
        return None;
    }

    let mips = generate_mipmaps(&file_image, job.srgb, has_alpha)?;

    let mut levels = vec![file_image];
    levels.extend(mips);
    Some((levels, has_alpha))
}

fn encode(job: &Job) -> Vec<u8> {
    let Some((levels, has_alpha)) = read_source_images(job) else {
        return b"error decoding source image".to_vec();
    };

    let mut params = BasisCompressorParams::default();
    fill_params(&mut params, job);

    // each mip level becomes a separate source image; alpha is decided on the source file like basisu
    params.source_images = levels.iter().map(Rgba::to_image).collect();
    params.force_alpha = has_alpha;
    params.check_for_alpha = has_alpha;

    let mut compressor = BasisCompressor::new();
    if !compressor.init(&params) {
        return b"error encoding image".to_vec();
    }

    match compressor.process() {
        BasisCompressorErrorCode::Success => {}
        BasisCompressorErrorCode::FailedReadingSourceImages => return b"error decoding source image".to_vec(),
        _ => return b"error encoding image".to_vec(),
    }

    let alpha = compressor.get_any_source_image_has_alpha();
    create_ktx2_file(compressor.get_output_basis_file(), levels.len(), job.uastc, alpha, job.srgb)
        .unwrap_or_else(|| b"error encoding image".to_vec())
}

const KTX2_IDENTIFIER: [u8; 12] = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A];
const KTX2_HEADER_SIZE: usize = 80;
const KTX2_LEVEL_INDEX_SIZE: usize = 24;

const KTX2_SS_BASISLZ: u32 = 1;
const KTX2_SS_ZSTANDARD: u32 = 2;
const KTX2_KHR_DF_TRANSFER_LINEAR: u32 = 1;
const KTX2_KHR_DF_TRANSFER_SRGB: u32 = 2;
const KTX2_DF_CHANNEL_UASTC_RGB: u32 = 0;
const KTX2_DF_CHANNEL_UASTC_RGBA: u32 = 3;

#[rustfmt::skip]
const KTX2_ETC1S_NONALPHA_DFD: [u8; 44] = [
    0x2C, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x2, 0x0, 0x28, 0x0, 0xA3, 0x1, 0x2, 0x0,
    0x3, 0x3, 0x0, 0x0, 0x8, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x3F, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xFF, 0xFF, 0xFF, 0xFF,
];

#[rustfmt::skip]
const KTX2_ETC1S_ALPHA_DFD: [u8; 60] = [
    0x3C, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x2, 0x0, 0x38, 0x0, 0xA3, 0x1, 0x2, 0x0,
    0x3, 0x3, 0x0, 0x0, 0x8, 0x8, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x3F, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xFF, 0xFF, 0xFF, 0xFF, 0x40, 0x0, 0x3F, 0xF,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xFF, 0xFF, 0xFF, 0xFF,
];

#[rustfmt::skip]
const KTX2_UASTC_LDR_4X4_NONALPHA_DFD: [u8; 44] = [
    0x2C, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x2, 0x0, 0x28, 0x0, 0xA6, 0x1, 0x2, 0x0,
    0x3, 0x3, 0x0, 0x0, 0x10, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x7F, 0x4,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xFF, 0xFF, 0xFF, 0xFF,
];

#[rustfmt::skip]
const KTX2_UASTC_LDR_4X4_ALPHA_DFD: [u8; 44] = [
    0x2C, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x2, 0x0, 0x28, 0x0, 0xA6, 0x1, 0x2, 0x0,
    0x3, 0x3, 0x0, 0x0, 0x10, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x7F, 0x3,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xFF, 0xFF, 0xFF, 0xFF,
];

fn read_dword(data: &[u8], index: usize) -> u32 {
    u32::from_le_bytes([data[index * 4], data[index * 4 + 1], data[index * 4 + 2], data[index * 4 + 3]])
}

fn write_dword(data: &mut [u8], index: usize, v: u32) {
    data[index * 4..index * 4 + 4].copy_from_slice(&v.to_le_bytes());
}

/// `basis_compressor::get_dfd` (LDR ETC1S / UASTC 4x4).
fn get_dfd(uastc: bool, alpha: bool, srgb: bool) -> Vec<u8> {
    let mut dfd = match (uastc, alpha) {
        (true, true) => KTX2_UASTC_LDR_4X4_ALPHA_DFD.to_vec(),
        (true, false) => KTX2_UASTC_LDR_4X4_NONALPHA_DFD.to_vec(),
        (false, true) => KTX2_ETC1S_ALPHA_DFD.to_vec(),
        (false, false) => KTX2_ETC1S_NONALPHA_DFD.to_vec(),
    };

    // Write the transfer function (linear vs. sRGB)
    let mut dfd_bits = read_dword(&dfd, 3);
    dfd_bits &= !(0xFF << 16);
    dfd_bits |= (if srgb {
        KTX2_KHR_DF_TRANSFER_SRGB
    } else {
        KTX2_KHR_DF_TRANSFER_LINEAR
    }) << 16;
    write_dword(&mut dfd, 3, dfd_bits);

    // Fix up the DFD channel(s)
    let mut dfd_chan0 = read_dword(&dfd, 7);
    if uastc {
        dfd_chan0 &= !(0xF << 24);
        dfd_chan0 |= (if alpha {
            KTX2_DF_CHANNEL_UASTC_RGBA
        } else {
            KTX2_DF_CHANNEL_UASTC_RGB
        }) << 24;
    }
    write_dword(&mut dfd, 7, dfd_chan0);

    dfd
}

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// `basis_compressor::create_ktx2_file` for a single 2D image with `total_levels` mipmaps, built from
/// the `.basis` output (slices are ordered by level, with an alpha slice after each ETC1S color slice).
fn create_ktx2_file(basis: &[u8], total_levels: usize, uastc: bool, alpha: bool, srgb: bool) -> Option<Vec<u8>> {
    let header = BasisFileHeader::from_bytes(basis)?;
    let range = |ofs: u32, size: u32| basis.get(ofs as usize..(ofs as usize).checked_add(size as usize)?);

    let slices_per_level = if !uastc && alpha { 2 } else { 1 };
    if header.total_slices as usize != total_levels * slices_per_level {
        return None;
    }

    let mut slices = Vec::new();
    for i in 0..header.total_slices {
        let ofs = header.slice_desc_file_ofs.checked_add(i * BASIS_SLICE_DESC_SIZE)?;
        let desc = BasisSliceDesc::from_bytes(basis.get(ofs as usize..)?)?;
        slices.push((desc, range(desc.file_ofs, desc.file_size)?));
    }

    let base_width = slices[0].0.orig_width;
    let base_height = slices[0].0.orig_height;

    // This will append the texture data in the correct order
    let mut level_data_bytes: Vec<Vec<u8>> = vec![Vec::new(); total_levels];
    let mut slice_level_offsets = Vec::with_capacity(slices.len());
    for (i, (_, data)) in slices.iter().enumerate() {
        let level = &mut level_data_bytes[i / slices_per_level];
        slice_level_offsets.push(level.len() as u32);
        level.extend_from_slice(data);
    }

    let supercompression_scheme;
    let mut compressed_level_data_bytes = Vec::with_capacity(total_levels);
    let mut ktx2_global_data = Vec::new();

    if uastc {
        supercompression_scheme = KTX2_SS_ZSTANDARD;

        for level in &level_data_bytes {
            let mut compressed = vec![0u8; zstd::ZSTD_compressBound(level.len())];
            let result = zstd::ZSTD_compress(&mut compressed, level, 9);
            if zstd::ZSTD_isError(result) {
                return None;
            }
            compressed.truncate(result);
            compressed_level_data_bytes.push(compressed);
        }
    } else {
        supercompression_scheme = KTX2_SS_BASISLZ;
        compressed_level_data_bytes = level_data_bytes.clone();

        let endpoints = range(header.endpoint_cb_file_ofs, header.endpoint_cb_file_size)?;
        let selectors = range(header.selector_cb_file_ofs, header.selector_cb_file_size)?;
        let tables = range(header.tables_file_ofs, header.tables_file_size)?;

        // ktx2_etc1s_global_data_header
        ktx2_global_data.extend_from_slice(&(header.total_endpoints as u16).to_le_bytes());
        ktx2_global_data.extend_from_slice(&(header.total_selectors as u16).to_le_bytes());
        push_u32(&mut ktx2_global_data, endpoints.len() as u32);
        push_u32(&mut ktx2_global_data, selectors.len() as u32);
        push_u32(&mut ktx2_global_data, tables.len() as u32);
        push_u32(&mut ktx2_global_data, 0);

        // ktx2_etc1s_image_desc per level: flags, rgb offset/length, alpha offset/length
        let mut image_descs = vec![[0u32; 5]; total_levels];
        for (i, (_, data)) in slices.iter().enumerate() {
            let d = &mut image_descs[i / slices_per_level];
            if alpha && i % 2 == 1 {
                d[3] = slice_level_offsets[i];
                d[4] = data.len() as u32;
            } else {
                d[1] = slice_level_offsets[i];
                d[2] = data.len() as u32;
            }
        }
        for d in &image_descs {
            for &v in d {
                push_u32(&mut ktx2_global_data, v);
            }
        }

        ktx2_global_data.extend_from_slice(endpoints);
        ktx2_global_data.extend_from_slice(selectors);
        ktx2_global_data.extend_from_slice(tables);
    }

    // Key values: only KTXwriter; no alignment key is needed as the data is always supercompressed
    let mut key_value_data = Vec::new();
    let key = b"KTXwriter\0";
    let value = b"Basis Universal 2.10\0";
    push_u32(&mut key_value_data, (key.len() + value.len()) as u32);
    key_value_data.extend_from_slice(key);
    key_value_data.extend_from_slice(value);
    while !key_value_data.len().is_multiple_of(4) {
        key_value_data.push(0);
    }

    let dfd = get_dfd(uastc, alpha, srgb);

    let mut out = vec![0u8; KTX2_HEADER_SIZE + KTX2_LEVEL_INDEX_SIZE * total_levels];

    let dfd_byte_offset = out.len() as u32;
    out.extend_from_slice(&dfd);

    let kvd_byte_offset = out.len() as u32;
    out.extend_from_slice(&key_value_data);

    let mut sgd_byte_offset = 0u64;
    if !ktx2_global_data.is_empty() {
        while !out.len().is_multiple_of(8) {
            out.push(0);
        }
        sgd_byte_offset = out.len() as u64;
        out.extend_from_slice(&ktx2_global_data);
    }

    // Level data - write the smallest mipmap first.
    let mut level_index = vec![[0u64; 3]; total_levels];
    for level in (0..total_levels).rev() {
        level_index[level][0] = out.len() as u64;
        level_index[level][1] = compressed_level_data_bytes[level].len() as u64;
        if uastc {
            level_index[level][2] = level_data_bytes[level].len() as u64;
        }
        out.extend_from_slice(&compressed_level_data_bytes[level]);
    }

    let mut head = Vec::with_capacity(KTX2_HEADER_SIZE + KTX2_LEVEL_INDEX_SIZE * total_levels);
    head.extend_from_slice(&KTX2_IDENTIFIER);
    push_u32(&mut head, 0); // vk_format: VK_FORMAT_UNDEFINED
    push_u32(&mut head, 1); // type_size
    push_u32(&mut head, base_width);
    push_u32(&mut head, base_height);
    push_u32(&mut head, 0); // pixel_depth
    push_u32(&mut head, 0); // layer_count
    push_u32(&mut head, 1); // face_count
    push_u32(&mut head, total_levels as u32);
    push_u32(&mut head, supercompression_scheme);
    push_u32(&mut head, dfd_byte_offset);
    push_u32(&mut head, dfd.len() as u32);
    push_u32(&mut head, kvd_byte_offset);
    push_u32(&mut head, key_value_data.len() as u32);
    push_u64(&mut head, sgd_byte_offset);
    push_u64(&mut head, ktx2_global_data.len() as u64);
    for l in &level_index {
        for &v in l {
            push_u64(&mut head, v);
        }
    }
    out[..head.len()].copy_from_slice(&head);

    Some(out)
}

/// `prepareEncode`: `Err` carries the error string, `Ok(None)` skips the image.
fn prepare_encode(
    data: &Data,
    image: &crate::cgltf::Image,
    input_path: Option<&str>,
    info: &ImageInfo,
    settings: &Settings,
) -> Result<Option<Job>, &'static str> {
    let Some((img_data, mime_type)) = read_image(data, image, input_path) else {
        return Err("error reading source file");
    };

    if mime_type != "image/png" && mime_type != "image/jpeg" {
        return Ok(None);
    }

    let Some((mut width, mut height)) = get_dimensions(&img_data, &mime_type) else {
        return Err("error parsing image header");
    };

    let kind = info.kind as usize;
    adjust_dimensions(
        &mut width,
        &mut height,
        settings.texture_scale[kind],
        settings.texture_limit[kind],
        settings.texture_pow2,
    );

    let quality = settings.texture_quality[kind];
    let uastc = settings.texture_mode[kind] == TextureMode::Uastc;

    let Some(bs) = quality
        .checked_sub(1)
        .and_then(|q| usize::try_from(q).ok())
        .and_then(|q| BASIS_SETTINGS.get(q))
    else {
        return Err("error encoding image");
    };

    Ok(Some(Job {
        data: img_data,
        mime_type,
        width,
        height,
        uastc,
        settings: bs,
        srgb: info.srgb,
        normal_map: info.normal_map,
        flipy: settings.texture_flipy,
    }))
}

/// `encodeImagesBasis`: fills `encoded[i]` with KTX2 data or an `error...` message.
pub fn encode_images_basis(encoded: &mut [Vec<u8>], data: &Data, images: &[ImageInfo], input_path: Option<&str>, settings: &Settings) {
    let mut jobs = Vec::new();

    for (i, image) in data.images.iter().enumerate() {
        let info = &images[i];
        let mode = settings.texture_mode[info.kind as usize];

        if mode == TextureMode::Etc1s || mode == TextureMode::Uastc {
            match prepare_encode(data, image, input_path, info, settings) {
                Ok(Some(job)) => jobs.push((i, job)),
                Ok(None) => {} // encoding was skipped
                Err(error) => encoded[i] = error.as_bytes().to_vec(),
            }
        }
    }

    let num_threads = if settings.texture_jobs == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        settings.texture_jobs.max(1) as usize
    };

    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: Vec<(usize, Vec<u8>)> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..num_threads.min(jobs.len()))
            .map(|_| {
                s.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some((i, job)) = jobs.get(j) else {
                            break;
                        };
                        done.push((*i, encode(job)));
                    }
                    done
                })
            })
            .collect();

        workers.into_iter().flat_map(|w| w.join().unwrap_or_default()).collect()
    });

    for (i, result) in results {
        encoded[i] = result;
    }
}
