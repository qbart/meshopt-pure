//! Opacity micromap generation.
//!
//! Opacity micromaps use a "bird" traversal order which recursively subdivides the triangles:
//! <https://docs.vulkan.org/spec/latest/_images/micromap-subd.svg>
//! note that triangles 0 and 2 have the same winding as the source triangle, however triangles 1 (flipped)
//! and 3 (upright) have flipped winding; this is obvious from the level 2 subdivision in the diagram above

use crate::error::{Error, Result, ensure};
use crate::hash::{Hasher, hash_buckets, hash_lookup};
use crate::validate;

fn get_level_size(level: i32, states: i32) -> usize {
    // 1-bit 2-state or 2-bit 4-state per micro triangle, rounded up to whole bytes
    ((1usize << (level * 2)) * (states >> 1) as usize + 7) >> 3
}

struct Texture<'a> {
    data: &'a [u8],
    stride: usize,
    pitch: usize,
    width: u32,
    height: u32,
    widthf: f32,  // width * 256.f
    heightf: f32, // height * 256.f
}

fn sample_texture(texture: &Texture, u: f32, v: f32) -> f32 {
    // wrap texture coordinates; floor is expensive so only call it if we're outside of [0, 1] range (+eps)
    let u = if (u - 0.5).abs() > 0.5 { u - u.floor() } else { u };
    let v = if (v - 0.5).abs() > 0.5 { v - v.floor() } else { v };

    // convert from [0, 1] to 16.8 fixed point coordinates (rounded to nearest subpixel) with texel centers on integer grid
    let uf = (u * texture.widthf - 127.5) as i32;
    let vf = (v * texture.heightf - 127.5) as i32;

    // clamp to avoid extrapolation past left/top edge since we don't wrap across the edge
    let uf = uf.max(0);
    let vf = vf.max(0);

    // x/y are texel coordinates, rx/ry are subpixel offsets
    let x = uf >> 8;
    let y = vf >> 8;
    let rx = uf & 255;
    let ry = vf & 255;

    // safeguard: this should not happen but if it ever does, ensure the accesses are inbounds
    if x as u32 >= texture.width || y as u32 >= texture.height {
        return 0.0;
    }

    // clamp the offsets instead of wrapping for simplicity and performance
    let offset = y as usize * texture.pitch + x as usize * texture.stride;
    let offsetx = if x + 1 < texture.width as i32 { texture.stride } else { 0 };
    let offsety = if y + 1 < texture.height as i32 { texture.pitch } else { 0 };

    let a00 = i32::from(texture.data[offset]);
    let a10 = i32::from(texture.data[offset + offsetx]);
    let a01 = i32::from(texture.data[offset + offsety]);
    let a11 = i32::from(texture.data[offset + offsetx + offsety]);

    // bilinear interpolation in integer space: result is 8.16 fixed point
    let ax0 = a00 * 256 + (a10 - a00) * rx;
    let ax1 = a01 * 256 + (a11 - a01) * rx;
    let axy = ax0 * 256 + (ax1 - ax0) * ry;

    axy as f32 * (1.0 / (255.0 * 65536.0))
}

fn hash_update4u(mut h: u32, key: &[u8]) -> u32 {
    // MurmurHash2
    const M: u32 = 0x5bd1e995;
    const R: u32 = 24;

    for w in key.as_chunks::<4>().0 {
        let mut k = u32::from_le_bytes(*w);

        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);

        h = h.wrapping_mul(M);
        h ^= k;
    }

    h
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TriangleOmm {
    uvs: [i32; 6],
    level: i32,
}

struct TriangleOmmHasher<'a> {
    data: &'a [TriangleOmm],
}

impl Hasher<u32> for TriangleOmmHasher<'_> {
    fn hash(&self, index: &u32) -> usize {
        let tri = &self.data[*index as usize];

        let mut key = [0u8; 24];
        for (k, uv) in key.as_chunks_mut::<4>().0.iter_mut().zip(&tri.uvs) {
            *k = uv.to_le_bytes();
        }

        hash_update4u(tri.level as u32, &key) as usize
    }

    fn equal(&mut self, lhs: &u32, rhs: &u32) -> bool {
        self.data[*lhs as usize] == self.data[*rhs as usize]
    }
}

struct OmmHasher<'a> {
    data: &'a [u8],
    offsets: &'a [u32],
    levels: &'a [u8],
    states: i32,
}

impl OmmHasher<'_> {
    fn entry(&self, index: u32) -> &[u8] {
        let offset = self.offsets[index as usize] as usize;
        &self.data[offset..offset + get_level_size(i32::from(self.levels[index as usize]), self.states)]
    }
}

impl Hasher<u32> for OmmHasher<'_> {
    fn hash(&self, index: &u32) -> usize {
        let key = self.entry(*index);
        let size = key.len();

        let mut h = u32::from(self.levels[*index as usize]);

        // MurmurHash2 for large keys, simple fold for small; note that size is a power of two
        if size < 4 {
            h ^= u32::from(key[0]) | (u32::from(key[size - 1]) << 8);
        } else {
            h = hash_update4u(h, key);
        }

        // MurmurHash2 finalizer
        h ^= h >> 13;
        h = h.wrapping_mul(0x5bd1e995);
        h ^= h >> 15;
        h as usize
    }

    fn equal(&mut self, lhs: &u32, rhs: &u32) -> bool {
        self.levels[*lhs as usize] == self.levels[*rhs as usize] && self.entry(*lhs) == self.entry(*rhs)
    }
}

fn quantize_subpixel(v: f32, size: u32) -> i32 {
    (v * (size as i32 * 4) as f32 + if v >= 0.0 { 0.5 } else { -0.5 }) as i32
}

fn rasterize_edge(u0: f32, v0: f32, u1: f32, v1: f32, edgeres: i32, texture: &Texture) -> i32 {
    let edgestep = 1.0 / (edgeres + 1) as f32;

    let (ud, vd) = ((u1 - u0) * edgestep, (v1 - v0) * edgestep);
    let (mut u, mut v) = (u0, v0);

    let mut mask = 0;
    let mut count = 0;

    for i in 0..edgeres {
        u += ud;
        v += vd;

        let a = sample_texture(texture, u, v);
        mask |= i32::from(a >= 0.5) << i;
        count += i32::from(a >= 0.5);
    }

    mask | (count << 16)
}

fn rasterize_opacity0(states: i32, result: &mut [u8], index: usize, a0: f32, a1: f32, a2: f32, ac: f32, e: [i32; 3], edgeres: i32) {
    // basic coverage estimator from center and corner values; trained to minimize error
    let mut coverage = (a0 + a1 + a2) * 0.12 + ac * 0.64;

    if edgeres != 0 {
        let edgescale = 1.0 / edgeres as f32;

        // if we have edge samples, we can get a better coverage estimate by including them; trained to minimize error
        coverage = ac * 0.22 + ((e[0] >> 16) + (e[1] >> 16) + (e[2] >> 16)) as f32 * edgescale * 0.23 + (a0 + a1 + a2) * 0.03;
    }

    if states == 2 {
        result[index / 8] |= u8::from(coverage >= 0.5) << (index % 8);
        return;
    }

    let transp = i32::from(a0 < 0.5) & i32::from(a1 < 0.5) & i32::from(a2 < 0.5) & i32::from(ac < 0.5);
    let opaque = i32::from(a0 > 0.5) & i32::from(a1 > 0.5) & i32::from(a2 > 0.5) & i32::from(ac > 0.5);

    // treat state as known if thresholding of corners & centers against wider bounds is consistent
    // for unknown states, we currently use the same formula as the 2-state opacity for better consistency with forced 2-state
    let unknown = 2 + i32::from(coverage >= 0.5);
    let mut state = if (transp | opaque) != 0 { opaque } else { unknown };

    if edgeres != 0 && (transp | opaque) != 0 {
        // if we have edge samples, ensure they are consistent too, falling back to unknown if not
        let exp = if opaque != 0 { (1 << edgeres) - 1 } else { 0 };
        let eok = (e[0] & 0xffff) == exp && (e[1] & 0xffff) == exp && (e[2] & 0xffff) == exp;

        state = if eok { state } else { unknown };
    }

    result[index / 4] |= (state << ((index % 4) * 2)) as u8;
}

fn midpoint(a: &[f32; 3], b: &[f32; 3], texture: &Texture) -> [f32; 3] {
    let (u, v) = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0);
    [u, v, sample_texture(texture, u, v)]
}

fn rasterize_opacity1(
    states: i32,
    result: &mut [u8],
    index: usize,
    edgeres: i32,
    c0: &[f32; 3],
    c1: &[f32; 3],
    c2: &[f32; 3],
    texture: &Texture,
) {
    // compute each edge midpoint & sample
    let c01 = midpoint(c0, c1, texture);
    let c12 = midpoint(c1, c2, texture);
    let c20 = midpoint(c2, c0, texture);

    // corner tables for each edge, and corner + edge tables for each triangle
    // edges are numbered counter clockwise, 6 outer first, 3 inner last; triangle vertex and edge references are in triangle winding order
    const EDGES: [[usize; 2]; 9] = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [5, 1], [1, 3], [3, 5]];
    const TRIANGLES: [[usize; 6]; 4] = [[0, 1, 5, 0, 6, 5], [5, 3, 1, 8, 7, 6], [1, 2, 3, 1, 2, 7], [3, 5, 4, 8, 4, 3]];

    let points = [c0, &c01, c1, &c12, c2, &c20];

    let mut em = [0i32; 9];

    // sample additional points on the edges to improve state estimation
    if edgeres > 0 {
        for (m, e) in em.iter_mut().zip(&EDGES) {
            let (p, q) = (points[e[0]], points[e[1]]);
            *m = rasterize_edge(p[0], p[1], q[0], q[1], edgeres, texture);
        }
    }

    for (i, tri) in TRIANGLES.iter().enumerate() {
        let (p0, p1, p2) = (points[tri[0]], points[tri[1]], points[tri[2]]);

        // compute triangle center & sample
        let uc = (p0[0] + p1[0] + p2[0]) * (1.0 / 3.0);
        let vc = (p0[1] + p1[1] + p2[1]) * (1.0 / 3.0);
        let ac = sample_texture(texture, uc, vc);

        // rasterize opacity state based on alpha values in corners and center (and optionally edges)
        rasterize_opacity0(
            states,
            result,
            index * 4 + i,
            p0[2],
            p1[2],
            p2[2],
            ac,
            [em[tri[3]], em[tri[4]], em[tri[5]]],
            edgeres,
        );
    }
}

fn rasterize_opacity_rec(
    states: i32,
    result: &mut [u8],
    index: usize,
    level: i32,
    edgeres: i32,
    c0: &[f32; 3],
    c1: &[f32; 3],
    c2: &[f32; 3],
    texture: &Texture,
) {
    if level == 0 {
        // compute triangle center & sample
        let uc = (c0[0] + c1[0] + c2[0]) * (1.0 / 3.0);
        let vc = (c0[1] + c1[1] + c2[1]) * (1.0 / 3.0);
        let ac = sample_texture(texture, uc, vc);

        let mut e = [0i32; 3];

        if edgeres > 0 {
            // sample additional points on the edges to improve state estimation
            e[0] = rasterize_edge(c0[0], c0[1], c1[0], c1[1], edgeres, texture);
            e[1] = rasterize_edge(c1[0], c1[1], c2[0], c2[1], edgeres, texture);
            e[2] = rasterize_edge(c2[0], c2[1], c0[0], c0[1], edgeres, texture);
        }

        // rasterize opacity state based on alpha values in corners and center (and optionally edges)
        return rasterize_opacity0(states, result, index, c0[2], c1[2], c2[2], ac, e, edgeres);
    }

    // fast path: equivalent to recursive rasterization, but reuses edge data to reduce sample count
    if level == 1 && edgeres > 0 {
        return rasterize_opacity1(states, result, index, edgeres, c0, c1, c2, texture);
    }

    // compute each edge midpoint & sample
    let c01 = midpoint(c0, c1, texture);
    let c12 = midpoint(c1, c2, texture);
    let c20 = midpoint(c2, c0, texture);

    // recursively rasterize each triangle
    // note: triangles 1 and 3 have flipped winding, and 1 is flipped upside down
    rasterize_opacity_rec(states, result, index * 4, level - 1, edgeres, c0, &c01, &c20, texture);
    rasterize_opacity_rec(states, result, index * 4 + 1, level - 1, edgeres, &c20, &c12, &c01, texture);
    rasterize_opacity_rec(states, result, index * 4 + 2, level - 1, edgeres, &c01, c1, &c12, texture);
    rasterize_opacity_rec(states, result, index * 4 + 3, level - 1, edgeres, &c12, &c20, c2, texture);
}

fn get_special_index(data: &[u8], level: i32, states: i32) -> i32 {
    let first = i32::from(data[0]) & if states == 2 { 1 } else { 3 };
    let special = -(1 + first);

    // at level 0, every micromap can be converted to a special index
    if level == 0 {
        return special;
    }

    // at level 1 with 2 states, the byte is partially filled so we need a separate check
    if level == 1 && states == 2 {
        return if i32::from(data[0] & 15) == ((-first) & 15) { special } else { 0 };
    }

    // otherwise we need to check that all bytes are consistent with the first value and we can do this byte-wise
    let expected = first * if states == 2 { 0xff } else { 0x55 };
    let size = get_level_size(level, states);

    if data[..size].iter().all(|&d| i32::from(d) == expected) {
        special
    } else {
        0
    }
}

fn validate_texture_size(texture_width: u32, texture_height: u32) -> Result<()> {
    ensure(
        texture_width.wrapping_sub(1) < 16384 && texture_height.wrapping_sub(1) < 16384,
        Error::InvalidArgument("texture dimensions must be in 1..=16384"),
    )
}

/// Opacity micromap generator (measure).
/// Computes a subdivision level for each input triangle, as well as deduplicating the triangles that reference the same UVs to reduce rasterization requests.
/// Returns the number of OMM entries.
///
/// `levels` and `sources` must hold `index_count / 3` entries (worst case output); `omm_indices` must hold `index_count / 3` entries.
/// `max_level` is the maximum subdivision level (0..=12); `target_edge` can be 0, when > 0 subdivision is adaptive.
pub fn opacity_map_measure(
    levels: &mut [u8],
    sources: &mut [u32],
    omm_indices: &mut [i32],
    indices: &[u32],
    index_count: usize,
    vertex_uvs: &[f32],
    vertex_count: usize,
    vertex_uvs_stride: usize,
    texture_width: u32,
    texture_height: u32,
    max_level: i32,
    target_edge: f32,
) -> Result<usize> {
    validate::triangles(index_count)?;
    validate::floats(vertex_uvs, vertex_count, vertex_uvs_stride, 8)?;
    validate_texture_size(texture_width, texture_height)?;
    ensure((0..=12).contains(&max_level), Error::InvalidArgument("max_level must be in 0..=12"))?;
    ensure(target_edge >= 0.0, Error::InvalidArgument("target_edge must be >= 0"))?;
    let indices = validate::indices(indices, index_count, vertex_count)?;
    let face_count = index_count / 3;
    let levels = validate::output(levels, face_count)?;
    let sources = validate::output(sources, face_count)?;
    let omm_indices = validate::output(omm_indices, face_count)?;

    let vertex_stride_float = vertex_uvs_stride / 4;
    let texture_area = texture_width as f32 * texture_height as f32;

    // hash map used to deduplicate triangle rasterization requests based on UV
    let mut table = vec![!0u32; hash_buckets(face_count)];

    let mut triangles = vec![TriangleOmm::default(); face_count];

    let mut result = 0usize;

    for (i, tri) in indices.as_chunks::<3>().0.iter().enumerate() {
        let uv = |v: u32| {
            (
                vertex_uvs[v as usize * vertex_stride_float],
                vertex_uvs[v as usize * vertex_stride_float + 1],
            )
        };
        let ((u0, v0), (u1, v1), (u2, v2)) = (uv(tri[0]), uv(tri[1]), uv(tri[2]));

        let mut level = max_level;

        if target_edge > 0.0 {
            // compute ratio of edge length (in texels) to target and determine subdivision level
            let uvarea = ((u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0)).abs() * 0.5 * texture_area;
            let ratio = uvarea.sqrt() / target_edge;
            let levelf = (if ratio > 1.0 { ratio } else { 1.0 }).log2();

            // round to nearest and clamp
            level = (levelf + 0.5) as i32;
            level = level.max(0);
            level = level.min(max_level);
        }

        // deduplicate rasterization requests based on UV
        let uvs = [
            quantize_subpixel(u0, texture_width),
            quantize_subpixel(v0, texture_height),
            quantize_subpixel(u1, texture_width),
            quantize_subpixel(v1, texture_height),
            quantize_subpixel(u2, texture_width),
            quantize_subpixel(v2, texture_height),
        ];

        triangles[result] = TriangleOmm { uvs, level }; // speculatively write triangle data to give hasher a way to compare it

        let entry = hash_lookup(&table, &mut TriangleOmmHasher { data: &triangles }, &(result as u32), &!0u32);

        if table[entry] == !0u32 {
            table[entry] = result as u32;
            levels[result] = level as u8;
            sources[result] = i as u32;
            result += 1;
        }

        omm_indices[i] = table[entry] as i32;
    }

    Ok(result)
}

/// Size in bytes of the opacity data for one entry of the given level (0..=12) and states (2 or 4).
pub fn opacity_map_entry_size(level: i32, states: i32) -> Result<usize> {
    ensure((0..=12).contains(&level), Error::InvalidArgument("level must be in 0..=12"))?;
    ensure(states == 2 || states == 4, Error::InvalidArgument("states must be 2 or 4"))?;

    Ok(get_level_size(level, states))
}

/// Opacity micromap generator (rasterize).
/// Rasterizes opacity state for a single triangle entry by sampling the alpha texture, using bilinear filtering and 0.5 alpha cutoff.
///
/// `result` must hold [`opacity_map_entry_size`] bytes. `texture_data` starts at the alpha channel of the first pixel (UNORM8);
/// `texture_stride` is the distance between pixels (1..=4), `texture_pitch` the distance between rows, in bytes.
pub fn opacity_map_rasterize(
    result: &mut [u8],
    level: i32,
    states: i32,
    uv0: &[f32; 2],
    uv1: &[f32; 2],
    uv2: &[f32; 2],
    texture_data: &[u8],
    texture_stride: usize,
    texture_pitch: usize,
    texture_width: u32,
    texture_height: u32,
) -> Result<()> {
    let size = opacity_map_entry_size(level, states)?;
    validate_texture_size(texture_width, texture_height)?;
    ensure(
        (1..=4).contains(&texture_stride),
        Error::InvalidArgument("texture_stride must be in 1..=4"),
    )?;
    ensure(
        texture_pitch >= texture_stride * texture_width as usize,
        Error::InvalidArgument("texture_pitch must be >= stride * width"),
    )?;
    ensure(
        texture_data.len() > (texture_height as usize - 1) * texture_pitch + (texture_width as usize - 1) * texture_stride,
        Error::BufferTooSmall("texture_data"),
    )?;
    let result = validate::output(result, size)?;

    result.fill(0);

    let texture = Texture {
        data: texture_data,
        stride: texture_stride,
        pitch: texture_pitch,
        width: texture_width,
        height: texture_height,
        widthf: texture_width as i32 as f32 * 256.0,
        heightf: texture_height as i32 as f32 * 256.0,
    };

    // determine number of edge samples for conservative state estimation
    let texture_area = texture_width as i32 as f32 * texture_height as i32 as f32;
    let uvarea = ((uv1[0] - uv0[0]) * (uv2[1] - uv0[1]) - (uv2[0] - uv0[0]) * (uv1[1] - uv0[1])).abs() * 0.5 * texture_area;
    let uvedge = uvarea.sqrt() / (1 << level) as f32;

    // target ~2px distance between edge samples (assuming equilateral microtriangles)
    let edgeres = ((uvedge * 0.75) as i32).clamp(0, 7);

    // rasterize all micro triangles recursively, passing corner data down to reduce redundant sampling
    let c0 = [uv0[0], uv0[1], sample_texture(&texture, uv0[0], uv0[1])];
    let c1 = [uv1[0], uv1[1], sample_texture(&texture, uv1[0], uv1[1])];
    let c2 = [uv2[0], uv2[1], sample_texture(&texture, uv2[0], uv2[1])];

    rasterize_opacity_rec(states, result, 0, level, edgeres, &c0, &c1, &c2, &texture);

    Ok(())
}

/// Opacity micromap generator (compact).
/// Compacts and deduplicates opacity data, merging identical micromap entries and replacing micromap states with special indices (-4..-1) when possible.
/// Returns the number of OMM entries after compaction; the data array should be trimmed using the last offset/size.
pub fn opacity_map_compact(
    data: &mut [u8],
    data_size: usize,
    levels: &mut [u8],
    offsets: &mut [u32],
    omm_count: usize,
    omm_indices: &mut [i32],
    triangle_count: usize,
    states: i32,
) -> Result<usize> {
    ensure(states == 2 || states == 4, Error::InvalidArgument("states must be 2 or 4"))?;
    let data = validate::output(data, data_size)?;
    let levels = validate::output(levels, omm_count)?;
    let offsets = validate::output(offsets, omm_count)?;
    let omm_indices = validate::output(omm_indices, triangle_count)?;

    for (&level, &offset) in levels.iter().zip(offsets.iter()) {
        ensure(level <= 12, Error::InvalidArgument("levels must be in 0..=12"))?;
        ensure(
            offset as usize + get_level_size(i32::from(level), states) <= data_size,
            Error::BufferTooSmall("data"),
        )?;
    }
    ensure(
        omm_indices.iter().all(|&i| i < 0 || (i as usize) < omm_count),
        Error::IndexOutOfRange,
    )?;

    let data_old = data.to_vec();

    let mut table = vec![!0u32; hash_buckets(omm_count)];

    let mut remap = vec![0i32; omm_count];

    let mut next = 0usize;
    let mut offset = 0usize;

    for (i, r) in remap.iter_mut().enumerate() {
        let level = i32::from(levels[i]);

        let size = get_level_size(level, states);
        let old = &data_old[offsets[i] as usize..offsets[i] as usize + size];

        // try to convert to a special index if all micro-triangle states are the same
        let special = get_special_index(old, level, states);
        if special < 0 {
            *r = special;
            continue;
        }

        // speculatively write data to give hasher a way to compare it
        data[offset..offset + size].copy_from_slice(old);
        offsets[next] = offset as u32;
        levels[next] = level as u8;

        let entry = hash_lookup(
            &table,
            &mut OmmHasher {
                data,
                offsets,
                levels,
                states,
            },
            &(next as u32),
            &!0u32,
        );

        if table[entry] == !0u32 {
            table[entry] = next as u32;
            next += 1;
            offset += size;
        }

        *r = table[entry] as i32;
    }

    // remap triangle indices to new indices or special indices
    for index in omm_indices.iter_mut() {
        *index = if *index < 0 { *index } else { remap[*index as usize] };
    }

    Ok(next)
}
