// This work is based on:
// Matthaeus Chajdas. GeometryFX 1.2 - Cluster Culling. 2016
// Jack Ritter. An Efficient Bounding Sphere. 1990
// Thomas Larsson. Fast and Tight Fitting Bounding Spheres. 2008

use crate::error::{Error, Result, ensure};
use crate::quantization::quantize_snorm;
use crate::validate;

// This must be <= 256 since meshlet indices are stored as bytes
const MESHLET_MAX_VERTICES: usize = 256;

// A reasonable limit is around 2*max_vertices or less
const MESHLET_MAX_TRIANGLES: usize = 512;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Bounds {
    /// bounding sphere, useful for frustum and occlusion culling
    pub center: [f32; 3],
    pub radius: f32,

    /// normal cone, useful for backface culling
    pub cone_apex: [f32; 3],
    pub cone_axis: [f32; 3],
    /// = cos(angle/2)
    pub cone_cutoff: f32,

    /// normal cone axis and cutoff, stored in 8-bit SNORM format; decode using x/127.0
    pub cone_axis_s8: [i8; 3],
    pub cone_cutoff_s8: i8,
}

const AXES: [[f32; 3]; 7] = [
    // X, Y, Z
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    // XYZ, -XYZ, X-YZ, XY-Z; normalized to unit length
    [0.57735026, 0.57735026, 0.57735026],
    [-0.57735026, 0.57735026, 0.57735026],
    [0.57735026, -0.57735026, 0.57735026],
    [0.57735026, 0.57735026, -0.57735026],
];

/// Point set for [`compute_bounding_sphere`]: `points` with float stride, optional radii with float stride, optional index list.
struct Spheres<'a> {
    points: &'a [f32],
    points_stride_float: usize,
    radii: Option<(&'a [f32], usize)>,
    indices: Option<&'a [u32]>,
    count: usize,
}

impl Spheres<'_> {
    fn index(&self, i: usize) -> usize {
        self.indices.map_or(i, |ix| ix[i] as usize)
    }

    fn point(&self, v: usize) -> &[f32] {
        &self.points[v * self.points_stride_float..]
    }

    fn radius(&self, v: usize) -> f32 {
        self.radii.map_or(0.0, |(r, stride)| r[v * stride])
    }
}

fn compute_bounding_sphere(axes: usize, s: &Spheres) -> [f32; 4] {
    // find extremum points along all axes; for each axis we get a pair of points with min/max coordinates
    let mut pmin = [0usize; 7];
    let mut pmax = [0usize; 7];
    let mut tmin = [f32::MAX; 7];
    let mut tmax = [-f32::MAX; 7];

    for i in 0..s.count {
        let v = s.index(i);
        let p = s.point(v);
        let r = s.radius(v);

        for axis in 0..axes {
            let ax = &AXES[axis];

            let tp = ax[0] * p[0] + ax[1] * p[1] + ax[2] * p[2];
            let (tpmin, tpmax) = (tp - r, tp + r);

            pmin[axis] = if tpmin < tmin[axis] { v } else { pmin[axis] };
            pmax[axis] = if tpmax > tmax[axis] { v } else { pmax[axis] };
            tmin[axis] = if tpmin < tmin[axis] { tpmin } else { tmin[axis] };
            tmax[axis] = if tpmax > tmax[axis] { tpmax } else { tmax[axis] };
        }
    }

    // find the pair of points with largest distance
    let mut paxis = 0;
    let mut paxisdr = 0.0f32;

    for axis in 0..axes {
        let (p1, p2) = (s.point(pmin[axis]), s.point(pmax[axis]));
        let (r1, r2) = (s.radius(pmin[axis]), s.radius(pmax[axis]));

        let d2 = (p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]) + (p2[2] - p1[2]) * (p2[2] - p1[2]);
        let dr = d2.sqrt() + r1 + r2;

        if dr > paxisdr {
            paxisdr = dr;
            paxis = axis;
        }
    }

    // use the longest segment as the initial sphere diameter
    let (p1, p2) = (s.point(pmin[paxis]), s.point(pmax[paxis]));
    let (r1, r2) = (s.radius(pmin[paxis]), s.radius(pmax[paxis]));

    let paxisd = ((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]) + (p2[2] - p1[2]) * (p2[2] - p1[2])).sqrt();
    let paxisk = if paxisd > 0.0 { (paxisd + r2 - r1) / (2.0 * paxisd) } else { 0.0 };

    let mut center = [
        p1[0] + (p2[0] - p1[0]) * paxisk,
        p1[1] + (p2[1] - p1[1]) * paxisk,
        p1[2] + (p2[2] - p1[2]) * paxisk,
    ];
    let mut radius = paxisdr / 2.0;

    // iteratively adjust the sphere up until all points fit
    for i in 0..s.count {
        let v = s.index(i);
        let p = s.point(v);
        let r = s.radius(v);

        let d2 =
            (p[0] - center[0]) * (p[0] - center[0]) + (p[1] - center[1]) * (p[1] - center[1]) + (p[2] - center[2]) * (p[2] - center[2]);
        let d = d2.sqrt();

        if d + r > radius {
            let k = if d > 0.0 { (d + r - radius) / (2.0 * d) } else { 0.0 };

            center[0] += k * (p[0] - center[0]);
            center[1] += k * (p[1] - center[1]);
            center[2] += k * (p[2] - center[2]);
            radius = (radius + d + r) / 2.0;
        }
    }

    [center[0], center[1], center[2], radius]
}

fn cluster_bounds(indices: &[u32], corners: &[u32], vertex_positions: &[f32], vertex_positions_stride: usize) -> Bounds {
    let vertex_stride_float = vertex_positions_stride / 4;

    // compute triangle normals (.w completes plane equation)
    let mut normals = [[0.0f32; 4]; MESHLET_MAX_TRIANGLES];
    let mut triangles = 0;

    for tri in indices.as_chunks::<3>().0 {
        let p0 = &vertex_positions[vertex_stride_float * tri[0] as usize..];
        let p1 = &vertex_positions[vertex_stride_float * tri[1] as usize..];
        let p2 = &vertex_positions[vertex_stride_float * tri[2] as usize..];

        let p10 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let p20 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

        let mut normalx = p10[1] * p20[2] - p10[2] * p20[1];
        let mut normaly = p10[2] * p20[0] - p10[0] * p20[2];
        let mut normalz = p10[0] * p20[1] - p10[1] * p20[0];

        let area = (normalx * normalx + normaly * normaly + normalz * normalz).sqrt();

        // no need to include degenerate triangles - they will be invisible anyway
        if area == 0.0 {
            continue;
        }

        normalx /= area;
        normaly /= area;
        normalz /= area;

        // record triangle normals; normal and corner 0 define a plane equation
        normals[triangles] = [normalx, normaly, normalz, -(normalx * p0[0] + normaly * p0[1] + normalz * p0[2])];
        triangles += 1;
    }

    let mut bounds = Bounds::default();

    // degenerate cluster, no valid triangles => trivial reject (cone data is 0)
    if triangles == 0 {
        return bounds;
    }

    // compute cluster bounding sphere; we'll use the center to determine normal cone apex as well
    let psphere = compute_bounding_sphere(
        7,
        &Spheres {
            points: vertex_positions,
            points_stride_float: vertex_stride_float,
            radii: None,
            indices: Some(corners),
            count: corners.len(),
        },
    );

    let center = [psphere[0], psphere[1], psphere[2]];

    // treating triangle normals as points, find the bounding sphere - the sphere center determines the optimal cone axis
    let normals_flat = normals.as_flattened();
    let nsphere = compute_bounding_sphere(
        3,
        &Spheres {
            points: normals_flat,
            points_stride_float: 4,
            radii: None,
            indices: None,
            count: triangles,
        },
    );

    let mut axis = [nsphere[0], nsphere[1], nsphere[2]];
    let axislength = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let invaxislength = if axislength == 0.0 { 0.0 } else { 1.0 / axislength };

    axis[0] *= invaxislength;
    axis[1] *= invaxislength;
    axis[2] *= invaxislength;

    // compute a tight cone around all normals, mindp = cos(angle/2)
    let mut mindp = 1.0f32;

    for n in &normals[..triangles] {
        let dp = n[0] * axis[0] + n[1] * axis[1] + n[2] * axis[2];

        mindp = if dp < mindp { dp } else { mindp };
    }

    // fill bounding sphere info; note that below we can return bounds without cone information for degenerate cones
    bounds.center = center;
    bounds.radius = psphere[3];

    // degenerate cluster, normal cone is larger than a hemisphere => trivial accept
    // note that if mindp is positive but close to 0, the triangle intersection code below gets less stable
    // we arbitrarily decide that if a normal cone is ~168 degrees wide or more, the cone isn't useful
    if mindp <= 0.1 {
        bounds.cone_cutoff = 1.0;
        bounds.cone_cutoff_s8 = 127;
        return bounds;
    }

    let mut maxt = 0.0f32;

    // we need to find the point on center-t*axis ray that lies in negative half-space of all triangles
    for n in &normals[..triangles] {
        // dot(center-t*axis-corner, trinormal) = 0
        // dot(center-corner, trinormal) - t * dot(axis, trinormal) = 0
        let dc = center[0] * n[0] + center[1] * n[1] + center[2] * n[2] + n[3];
        let dn = axis[0] * n[0] + axis[1] * n[1] + axis[2] * n[2];

        // dn should be larger than mindp cutoff above
        let t = dc / dn;

        maxt = if t > maxt { t } else { maxt };
    }

    // cone apex should be in the negative half-space of all cluster triangles by construction
    bounds.cone_apex = [center[0] - axis[0] * maxt, center[1] - axis[1] * maxt, center[2] - axis[2] * maxt];

    // note: this axis is the axis of the normal cone, but our test for perspective camera effectively negates the axis
    bounds.cone_axis = axis;

    // cos(a) for normal cone is mindp; we need to add 90 degrees on both sides and invert the cone
    // which gives us -cos(a+90) = -(-sin(a)) = sin(a) = sqrt(1 - cos^2(a))
    bounds.cone_cutoff = (1.0 - mindp * mindp).sqrt();

    // quantize axis & cutoff to 8-bit SNORM format
    for k in 0..3 {
        bounds.cone_axis_s8[k] = quantize_snorm(bounds.cone_axis[k], 8) as i8;
    }

    // for the 8-bit test to be conservative, we need to adjust the cutoff by measuring the max. error
    let cone_axis_s8_e0 = (f32::from(bounds.cone_axis_s8[0]) / 127.0 - bounds.cone_axis[0]).abs();
    let cone_axis_s8_e1 = (f32::from(bounds.cone_axis_s8[1]) / 127.0 - bounds.cone_axis[1]).abs();
    let cone_axis_s8_e2 = (f32::from(bounds.cone_axis_s8[2]) / 127.0 - bounds.cone_axis[2]).abs();

    // note that we need to round this up instead of rounding to nearest, hence +1
    let cone_cutoff_s8 = (127.0 * (bounds.cone_cutoff + cone_axis_s8_e0 + cone_axis_s8_e1 + cone_axis_s8_e2) + 1.0) as i32;

    bounds.cone_cutoff_s8 = if cone_cutoff_s8 > 127 { 127 } else { cone_cutoff_s8 as i8 };

    bounds
}

/// Cluster bounds generator.
/// Creates bounding volumes that can be used for frustum, backface and occlusion culling.
/// `vertex_count` is the number of vertices in the entire mesh; `index_count / 3` must be <= 512.
pub fn compute_cluster_bounds(
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> Result<Bounds> {
    validate::triangles(index_count)?;
    ensure(
        index_count / 3 <= MESHLET_MAX_TRIANGLES,
        Error::InvalidArgument("index_count / 3 must be <= 512"),
    )?;
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    let indices = validate::indices(indices, index_count, vertex_count)?;

    let mut cache = [!0u32; 512];

    let mut corners = [0u32; MESHLET_MAX_TRIANGLES * 3 + 1]; // +1 for branchless slot
    let mut corner_count = 0;

    for &v in indices {
        let c = &mut cache[(v & 511) as usize];

        // branchless append if vertex isn't in cache
        corners[corner_count] = v;
        corner_count += usize::from(*c != v);
        *c = v;
    }

    Ok(cluster_bounds(
        indices,
        &corners[..corner_count],
        vertex_positions,
        vertex_positions_stride,
    ))
}

/// Meshlet bounds generator; see [`compute_cluster_bounds`]. `triangle_count` must be <= 512.
pub fn compute_meshlet_bounds(
    meshlet_vertices: &[u32],
    meshlet_triangles: &[u8],
    triangle_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> Result<Bounds> {
    ensure(
        triangle_count <= MESHLET_MAX_TRIANGLES,
        Error::InvalidArgument("triangle_count must be <= 512"),
    )?;
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    let meshlet_triangles = meshlet_triangles
        .get(..triangle_count * 3)
        .ok_or(Error::BufferTooSmall("meshlet_triangles"))?;

    let mut indices = [0u32; MESHLET_MAX_TRIANGLES * 3];
    let mut corner_count = 0;

    for (index, &t) in indices.iter_mut().zip(meshlet_triangles) {
        let v = *meshlet_vertices
            .get(usize::from(t))
            .ok_or(Error::BufferTooSmall("meshlet_vertices"))?;
        ensure((v as usize) < vertex_count, Error::IndexOutOfRange)?;

        *index = v;

        // meshlet_vertices[] slice should only contain vertices used by triangle indices, which is the case for any well formed meshlet
        corner_count = if usize::from(t) >= corner_count {
            usize::from(t) + 1
        } else {
            corner_count
        };
    }

    let corners = &meshlet_vertices[..corner_count];
    ensure(corners.iter().all(|&v| (v as usize) < vertex_count), Error::IndexOutOfRange)?;

    Ok(cluster_bounds(
        &indices[..triangle_count * 3],
        corners,
        vertex_positions,
        vertex_positions_stride,
    ))
}

/// Sphere bounds generator.
/// Creates bounding sphere around a set of points or a set of spheres; returns the center and radius of the sphere, with other fields of the result set to 0.
/// `radii` can be `None`; otherwise it holds a non-negative radius in the first 4 bytes of each element.
pub fn compute_sphere_bounds(
    positions: &[f32],
    count: usize,
    positions_stride: usize,
    radii: Option<&[f32]>,
    radii_stride: usize,
) -> Result<Bounds> {
    validate::floats(positions, count, positions_stride, 12)?;
    if let Some(r) = radii {
        validate::floats(r, count, radii_stride, 4)?;
    }

    let mut bounds = Bounds::default();

    if count == 0 {
        return Ok(bounds);
    }

    let psphere = compute_bounding_sphere(
        7,
        &Spheres {
            points: positions,
            points_stride_float: positions_stride / 4,
            radii: radii.map(|r| (r, radii_stride / 4)),
            indices: None,
            count,
        },
    );

    bounds.center = [psphere[0], psphere[1], psphere[2]];
    bounds.radius = psphere[3];

    Ok(bounds)
}

/// Meshlet optimizer.
/// Reorders meshlet vertices and triangles to maximize locality, with higher levels resulting in smaller compressed size at the cost of optimization time.
/// At level 0 the result is equivalent to [`optimize_meshlet`]; levels >= 1 may rotate triangle corners to improve compression.
/// `level` should be in 0..=9; `vertex_count <= 256`, `triangle_count <= 512`.
pub fn optimize_meshlet_level(
    meshlet_vertices: &mut [u32],
    vertex_count: usize,
    meshlet_triangles: &mut [u8],
    triangle_count: usize,
    level: i32,
) -> Result<()> {
    ensure(
        triangle_count <= MESHLET_MAX_TRIANGLES,
        Error::InvalidArgument("triangle_count must be <= 512"),
    )?;
    ensure(
        vertex_count <= MESHLET_MAX_VERTICES,
        Error::InvalidArgument("vertex_count must be <= 256"),
    )?;
    ensure((0..=9).contains(&level), Error::InvalidArgument("level must be in 0..=9"))?;
    let vertices = validate::output(meshlet_vertices, vertex_count)?;
    let indices = validate::output(meshlet_triangles, triangle_count * 3)?;
    ensure(indices.iter().all(|&i| usize::from(i) < vertex_count), Error::IndexOutOfRange)?;

    // cache tracks vertex timestamps (corresponding to triangle index! all 3 vertices are added at the same time and never removed)
    let mut cache = [0u8; MESHLET_MAX_VERTICES];

    // note that we start from a value that means all vertices aren't in cache
    let mut cache_last = 128u8;
    const CACHE_CUTOFF: u8 = 3; // 3 triangles = ~5..9 vertices depending on reuse

    // vertex valence is used to prioritize triangles for level>0
    // note: we use 8-bit counters for performance; for outlier vertices the valence is incorrect but that just affects the heuristic
    let mut valence = [0u8; MESHLET_MAX_VERTICES];

    for &i in indices.iter() {
        valence[usize::from(i)] = valence[usize::from(i)].wrapping_add(1);
    }

    for i in 0..triangle_count {
        let mut next = -1i32;
        let mut next_score = -1i32;
        let mut edges = 0;

        for j in i..triangle_count {
            let (a, b, c) = (
                usize::from(indices[j * 3]),
                usize::from(indices[j * 3 + 1]),
                usize::from(indices[j * 3 + 2]),
            );

            // compute cache distance using unsigned 8-bit subtraction, so cache timestamp overflow is handled gracefully
            let ad = cache_last.wrapping_sub(cache[a]);
            let bd = cache_last.wrapping_sub(cache[b]);
            let cd = cache_last.wrapping_sub(cache[c]);

            let matches = i32::from(ad < CACHE_CUTOFF) + i32::from(bd < CACHE_CUTOFF) + i32::from(cd < CACHE_CUTOFF);

            if level != 0 {
                // prefer low minimum valence
                let vmin = i32::from(valence[a].min(valence[b]).min(valence[c]));

                // prefer vertices with smaller cache distance and valence to improve traversal locality
                let mut score = matches * 1024 + (1023 - i32::from(ad) - i32::from(bd) - i32::from(cd));
                score = score * 256 + (255 - vmin);

                next = if score > next_score { j as i32 } else { next };
                next_score = if score > next_score { score } else { next_score };

                // terminate after finding enough edge matches
                if matches >= 2 {
                    edges += 1;
                    if edges >= level {
                        break;
                    }
                }
            } else {
                let score = matches;

                next = if score > next_score { j as i32 } else { next };
                next_score = if score > next_score { score } else { next_score };

                // settle for a first edge match, which makes the function ~linear in practice
                if matches >= 2 {
                    break;
                }
            }
        }

        let next = next as usize;
        let tri = [indices[next * 3], indices[next * 3 + 1], indices[next * 3 + 2]];

        // shift triangles before the next one forward so that we always keep an ordered partition
        // note: this could have swapped triangles [i] and [next] but that distorts the order and may skew the output sequence
        indices.copy_within(i * 3..next * 3, (i + 1) * 3);

        indices[i * 3..i * 3 + 3].copy_from_slice(&tri);

        // cache timestamp is the same between all vertices of each triangle to reduce overflow
        cache_last = cache_last.wrapping_add(1);

        for v in tri {
            cache[usize::from(v)] = cache_last;

            // update vertex valences for scoring heuristic
            valence[usize::from(v)] = valence[usize::from(v)].wrapping_sub(1);
        }
    }

    // rotate triangles to maximize compressibility; only done at level >= 1 for compatibility
    if level >= 1 {
        cache[..vertex_count].fill(0);

        for i in 0..triangle_count {
            let (mut a, mut b, mut c) = (indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]);
            let seen = |v: u8, cache: &[u8]| cache[usize::from(v)] != 0;

            // if only the middle vertex has been used, rotate triangle to ensure new vertices are always sequential
            if !seen(a, &cache) && seen(b, &cache) && !seen(c, &cache) {
                // abc -> bca
                (a, b, c) = (b, c, a);
            } else if !seen(a, &cache) && !seen(b, &cache) && !seen(c, &cache) {
                // out of three edges, the edge ab can not be reused by subsequent triangles in some encodings
                // if subsequent triangles don't share edges ca or bc, we can rotate the triangle to fix this
                let (mut needab, mut needbc, mut needca) = (false, false, false);

                for j in i + 1..triangle_count.min(i + usize::from(CACHE_CUTOFF) + 1) {
                    let (oa, ob, oc) = (indices[j * 3], indices[j * 3 + 1], indices[j * 3 + 2]);

                    // note: edge comparisons are reversed as reused edges are flipped
                    needab |= (oa == b && ob == a) || (ob == b && oc == a) || (oc == b && oa == a);
                    needbc |= (oa == c && ob == b) || (ob == c && oc == b) || (oc == c && oa == b);
                    needca |= (oa == a && ob == c) || (ob == a && oc == c) || (oc == a && oa == c);
                }

                if needab && !needbc {
                    // abc -> bca
                    (a, b, c) = (b, c, a);
                } else if (needab && !needca) || (!needab && needbc && !needca) {
                    // abc -> cab: put bc at most recent FIFO position
                    (a, b, c) = (c, a, b);
                }
            }

            indices[i * 3..i * 3 + 3].copy_from_slice(&[a, b, c]);

            cache[usize::from(a)] = 1;
            cache[usize::from(b)] = 1;
            cache[usize::from(c)] = 1;
        }
    }

    // reorder meshlet vertices for access locality assuming index buffer is scanned sequentially
    let mut order = [0u32; MESHLET_MAX_VERTICES];

    let mut remap = [-1i16; MESHLET_MAX_VERTICES];

    let mut vertex_offset = 0;

    for index in indices.iter_mut() {
        let r = &mut remap[usize::from(*index)];

        if *r < 0 {
            *r = vertex_offset as i16;
            order[vertex_offset] = vertices[usize::from(*index)];
            vertex_offset += 1;
        }

        *index = *r as u8;
    }

    vertices[..vertex_offset].copy_from_slice(&order[..vertex_offset]);

    Ok(())
}

/// Meshlet optimizer.
/// Reorders meshlet vertices and triangles to maximize locality which can improve rasterizer throughput or ray tracing performance.
pub fn optimize_meshlet(
    meshlet_vertices: &mut [u32],
    meshlet_triangles: &mut [u8],
    triangle_count: usize,
    vertex_count: usize,
) -> Result<()> {
    optimize_meshlet_level(meshlet_vertices, vertex_count, meshlet_triangles, triangle_count, 0)
}

/// Extract meshlet-local vertex and triangle indices from absolute cluster indices.
/// Fills `triangles` and `vertices` such that `vertices[triangles[i]] == indices[i]`, and returns the number of unique vertices.
/// `indices` should have at most 256 unique vertex indices; `index_count / 3` must be <= 512.
pub fn extract_meshlet_indices(vertices: &mut [u32], triangles: &mut [u8], indices: &[u32], index_count: usize) -> Result<usize> {
    validate::triangles(index_count)?;
    ensure(
        index_count / 3 <= MESHLET_MAX_TRIANGLES,
        Error::InvalidArgument("index_count / 3 must be <= 512"),
    )?;
    let indices = indices.get(..index_count).ok_or(Error::BufferTooSmall("indices"))?;
    let triangles = validate::output(triangles, index_count)?;

    let mut unique = 0;

    // direct mapped cache for fast lookups based on low index bits; inspired by vk_lod_clusters from NVIDIA
    let mut cache = [-1i16; 1024];

    for (t, &v) in triangles.iter_mut().zip(indices) {
        let key = (v & 1023) as usize;
        let c = cache[key];

        // fast path: vertex has been seen before
        if c >= 0 && vertices[c as usize] == v {
            *t = c as u8;
            continue;
        }

        // slow path: collision with a different vertex, so we need to look through all vertices
        // (skipped when the vertex has never been seen before)
        let found = if c < 0 {
            None
        } else {
            vertices[..unique].iter().position(|&x| x == v)
        };

        let pos = match found {
            Some(p) => p,
            None => {
                ensure(
                    unique < MESHLET_MAX_VERTICES,
                    Error::InvalidArgument("indices must have at most 256 unique vertices"),
                )?;
                *vertices.get_mut(unique).ok_or(Error::BufferTooSmall("vertices"))? = v;
                unique += 1;
                unique - 1
            }
        };

        cache[key] = pos as i16;
        *t = pos as u8;
    }

    Ok(unique)
}
