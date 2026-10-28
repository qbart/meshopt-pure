// This work is based on:
// Morten Mikkelsen. Simulation of wrinkled surfaces revisited. 2008
// Matthias Teschner, Bruno Heidelberger, Matthias Mueller, Danat Pomeranets, Markus Gross. Optimized Spatial Hashing for Collision Detection of Deformable Objects. 2003
// Cecil Hastings Jr. Approximations for digital computers. 1955

use crate::error::{Error, Result, ensure};
use crate::hash::{Hasher, hash_buckets, hash_lookup};
use crate::index::{self, IndexType};
use crate::validate;

/// Produce tangents compatible with MikkTSpace (same weighting and fallbacks) at the cost of reduced quality. Not recommended unless normal maps are baked.
pub const TANGENT_COMPATIBLE: u32 = 1 << 0;
/// For vertices only connected to degenerate triangles, output zero tangents instead of an arbitrary fallback.
pub const TANGENT_ZERO_FALLBACK: u32 = 1 << 1;

const NEXT: [usize; 4] = [1, 2, 0, 1];

#[derive(Debug, Clone, Copy, Default)]
struct Tangent {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct Normal {
    x: f32,
    y: f32,
    z: f32,
    id: i32,
}

/// Corner index -> vertex index, for indexed or unindexed input.
#[derive(Clone, Copy)]
struct Corners<'a> {
    indices: Option<&'a [u32]>,
}

impl Corners<'_> {
    fn get(&self, corner: usize) -> usize {
        self.indices.map_or(corner, |ib| ib[corner] as usize)
    }
}

/// float stream accessor (data + stride in floats)
#[derive(Clone, Copy)]
struct Stream<'a> {
    data: &'a [f32],
    stride: usize,
}

impl Stream<'_> {
    fn at(&self, i: usize) -> &[f32] {
        &self.data[i * self.stride..]
    }
}

fn compute_face_tangents(result: &mut [Tangent], corners: Corners, positions: Stream, uvs: Stream) {
    for (i, r) in result.iter_mut().enumerate() {
        let (a, b, c) = (corners.get(i * 3), corners.get(i * 3 + 1), corners.get(i * 3 + 2));

        let (pa, pb, pc) = (positions.at(a), positions.at(b), positions.at(c));
        let (ta, tb, tc) = (uvs.at(a), uvs.at(b), uvs.at(c));

        // compute face tangents using chain rule, see eq. 42
        let (dp1x, dp1y, dp1z) = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]);
        let (dp2x, dp2y, dp2z) = (pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]);
        let (dt1x, dt1y) = (tb[0] - ta[0], tb[1] - ta[1]);
        let (dt2x, dt2y) = (tc[0] - ta[0], tc[1] - ta[1]);

        let rx = dt2y * dp1x - dt1y * dp2x;
        let ry = dt2y * dp1y - dt1y * dp2y;
        let rz = dt2y * dp1z - dt1y * dp2z;

        // .w stores orientation (+1/-1) or 0 for degenerate triangles, see eq. 48
        let det = dt1x * dt2y - dt1y * dt2x;
        let mut rw = if det == 0.0 {
            0.0
        } else if det > 0.0 {
            1.0
        } else {
            -1.0
        };

        // treat zero area triangles as degenerate even if their UVs aren't for consistency
        let same = |p: &[f32], q: &[f32]| p[0] == q[0] && p[1] == q[1] && p[2] == q[2];
        if same(pa, pb) || same(pa, pc) || same(pb, pc) {
            rw = 0.0;
        }

        let rl = (rx * rx + ry * ry + rz * rz).sqrt();
        let rs = if rl != 0.0 { rw / rl } else { 0.0 };

        *r = Tangent {
            x: rx * rs,
            y: ry * rs,
            z: rz * rs,
            w: rw,
        };
    }
}

fn compute_face_normals(result: &mut [Normal], corners: Corners, positions: Stream) {
    for (i, r) in result.iter_mut().enumerate() {
        let (pa, pb, pc) = (
            positions.at(corners.get(i * 3)),
            positions.at(corners.get(i * 3 + 1)),
            positions.at(corners.get(i * 3 + 2)),
        );

        let (dp1x, dp1y, dp1z) = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]);
        let (dp2x, dp2y, dp2z) = (pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]);

        let nx = dp1y * dp2z - dp1z * dp2y;
        let ny = dp1z * dp2x - dp1x * dp2z;
        let nz = dp1x * dp2y - dp1y * dp2x;

        let nl = (nx * nx + ny * ny + nz * nz).sqrt();
        let ns = if nl != 0.0 { 1.0 / nl } else { 0.0 };

        *r = Normal {
            x: nx * ns,
            y: ny * ns,
            z: nz * ns,
            id: i32::from(nl != 0.0),
        };
    }
}

struct VertexHasherF<'a> {
    positions: Stream<'a>,
    /// normals + uvs; `None` hashes positions only
    attributes: Option<(Stream<'a>, Stream<'a>)>,
}

impl Hasher<u32> for VertexHasherF<'_> {
    fn hash(&self, index: &u32) -> usize {
        let index = *index as usize;
        let p = self.positions.at(index);

        let (mut x, mut y, mut z) = (p[0].to_bits(), p[1].to_bits(), p[2].to_bits());

        // replace negative zero with zero
        x = if x == 0x8000_0000 { 0 } else { x };
        y = if y == 0x8000_0000 { 0 } else { y };
        z = if z == 0x8000_0000 { 0 } else { z };

        // scramble bits to make sure that integer coordinates have entropy in lower bits
        x ^= x >> 17;
        y ^= y >> 17;
        z ^= z >> 17;

        let Some((normals, uvs)) = self.attributes else {
            return (x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663) ^ z.wrapping_mul(83492791)) as usize;
        };

        let n = normals.at(index);
        let t = uvs.at(index);
        let nb = |v: f32| if v.to_bits() == 0x8000_0000 { 0 } else { v.to_bits() >> 15 };

        // mix in normal bits
        x ^= nb(n[0]);
        y ^= nb(n[1]);
        z ^= nb(n[2]);

        // collect texture coordinate bits (simplified -0 handling and scrambling)
        let mut w = (t[0].to_bits() ^ t[1].to_bits()) & 0x7fff_ffff;
        w ^= w >> 13;

        // Optimized Spatial Hashing for Collision Detection of Deformable Objects
        (x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663) ^ z.wrapping_mul(83492791) ^ w.wrapping_mul(50331653)) as usize
    }

    fn equal(&mut self, lhs: &u32, rhs: &u32) -> bool {
        let (l, r) = (*lhs as usize, *rhs as usize);
        let (lp, rp) = (self.positions.at(l), self.positions.at(r));

        let positions_equal = lp[0] == rp[0] && lp[1] == rp[1] && lp[2] == rp[2];

        match self.attributes {
            None => positions_equal,
            Some((normals, uvs)) => {
                let (ln, rn) = (normals.at(l), normals.at(r));
                let (lt, rt) = (uvs.at(l), uvs.at(r));

                positions_equal && ln[0] == rn[0] && ln[1] == rn[1] && ln[2] == rn[2] && lt[0] == rt[0] && lt[1] == rt[1]
            }
        }
    }
}

fn build_vertex_remap(remap: &mut [u32], hasher: &mut VertexHasherF) {
    let mut vertex_table = vec![!0u32; hash_buckets(remap.len())];

    for (i, r) in remap.iter_mut().enumerate() {
        let index = i as u32;
        let entry = hash_lookup(&vertex_table, hasher, &index, &!0u32);

        if vertex_table[entry] == !0u32 {
            vertex_table[entry] = index;
        }

        *r = vertex_table[entry];
    }
}

struct CornerAdjacency {
    offsets: Vec<u32>,
    data: Vec<u32>,
}

fn build_corner_adjacency(corners: Corners, index_count: usize, remap: &[u32], vertex_count: usize) -> CornerAdjacency {
    let mut offsets = vec![0u32; vertex_count + 1];
    let mut data = vec![0u32; index_count];

    // fill corner counts
    for i in 0..index_count {
        offsets[remap[corners.get(i)] as usize + 1] += 1;
    }

    // fill offset table
    let mut offset = 0u32;

    for o in offsets[1..].iter_mut() {
        let count = *o;
        *o = offset;
        offset += count;
    }

    // fill corner data
    for i in 0..index_count / 3 {
        for k in 0..3 {
            let v = remap[corners.get(i * 3 + k)] as usize;

            // encode corner index in the low 2 bits
            data[offsets[v + 1] as usize] = ((i << 2) | k) as u32;
            offsets[v + 1] += 1;
        }
    }

    // finalize offsets
    offsets[0] = 0;

    CornerAdjacency { offsets, data }
}

fn follow2(parents: &mut [u32], mut index: u32) -> u32 {
    while index != parents[index as usize] {
        let parent = parents[index as usize];
        parents[index as usize] = parents[parent as usize];
        index = parent;
    }

    index
}

/// Remapped vertices of the two other corners of the triangle corner encoded in `d`.
fn corner_edges(d: u32, corners: Corners, remap: &[u32]) -> (usize, u32, u32) {
    let t = (d >> 2) as usize;
    let k = (d & 3) as usize;
    let cb = t * 3 + NEXT[k];
    let cc = t * 3 + NEXT[k + 1];

    (t, remap[corners.get(cb)], remap[corners.get(cc)])
}

fn merge_tangent_groups(groups: &mut [u32], facegroups: &mut [u32], facesign: &mut [u8], data: &[u32], corners: Corners, remap: &[u32]) {
    for (i, &di) in data.iter().enumerate() {
        let (ti, nib, nic) = corner_edges(di, corners, remap);

        for &dj in &data[i + 1..] {
            let (tj, njb, njc) = corner_edges(dj, corners, remap);

            // merge tangent groups if triangles are adjacent and orientations agree or either one is degenerate
            if (njb == nic || njc == nib) && (facesign[ti] | facesign[tj]) != 3 {
                // if either triangle is degenerate, it could have been assigned to a concrete orientation; we need to confirm that via facegroups
                if (facesign[ti] & facesign[tj]) == 0 {
                    let gti = follow2(facegroups, ti as u32) as usize;
                    let gtj = follow2(facegroups, tj as u32) as usize;

                    // skip merge if orientations diverge; otherwise, union the groups with gti as the root
                    if gti != gtj {
                        if (facesign[gti] | facesign[gtj]) == 3 {
                            continue;
                        }

                        facegroups[gtj] = gti as u32;
                        facesign[gti] |= facesign[gtj];
                    }
                }

                // union tangent groups for individual corners with gi as the root; the orientations must match due to facegroups check above
                let gi = follow2(groups, (ti * 3) as u32 + (di & 3));
                let gj = follow2(groups, (tj * 3) as u32 + (dj & 3));

                if gi != gj {
                    groups[gj as usize] = gi;
                }
            }
        }
    }
}

fn merge_normal_groups(groups: &mut [u32], data: &[u32], corners: Corners, remap: &[u32], face_normals: &[Normal], crease_cutoff: f32) {
    for (i, &di) in data.iter().enumerate() {
        let (ti, nib, nic) = corner_edges(di, corners, remap);
        let ci = (ti * 3) as u32 + (di & 3);

        let fi = face_normals[ti];

        for &dj in &data[i + 1..] {
            let (tj, njb, njc) = corner_edges(dj, corners, remap);
            let cj = (tj * 3) as u32 + (dj & 3);

            let fj = face_normals[tj];

            let adjacent = njb == nic || njc == nib;

            // merge normal groups if triangles are adjacent, non-degenerate, and normal angle is below crease threshold
            if adjacent && fi.x * fj.x + fi.y * fj.y + fi.z * fj.z > crease_cutoff && fi.id == fj.id && fi.id != 0 {
                // union normal groups for individual corners with gi as the root
                let gi = follow2(groups, ci);
                let gj = follow2(groups, cj);

                if gi != gj {
                    groups[gj as usize] = gi;
                }
            } else if adjacent && (fi.id == 0) != (fj.id == 0) {
                // corners of unattached degenerate triangles join the first adjacent non-degenerate group
                // this makes sure that two non-degenerate triangles can't be joined via a degenerate bridge
                let cd = if fi.id != 0 { cj } else { ci };
                let cn = if fi.id != 0 { ci } else { cj };

                if groups[cd as usize] == cd {
                    groups[cd as usize] = follow2(groups, cn);
                }
            }
        }
    }
}

#[allow(clippy::approx_constant)] // 3.1415926 is intentionally not f32::consts::PI (bit-exact with C)
fn optacos(x: f32) -> f32 {
    // approximation for acos(x) = sqrt(1 - abs(x)) * 2-degree polynomial, max error ~5e-4
    let mut ax = x.abs();
    ax = if ax > 1.0 { 1.0 } else { ax };
    let mut r = 1.570337 + ax * (-0.2053972 + ax * 0.05147786);
    r *= (1.0 - ax).sqrt();
    if x < 0.0 { 3.1415926 - r } else { r }
}

fn accumulate_tangents(
    result: &mut [f32],
    groups: &[u32],
    corners: Corners,
    face_tangents: &[Tangent],
    positions: Stream,
    normals: Stream,
    options: u32,
) {
    for (i, t) in face_tangents.iter().enumerate() {
        if t.w == 0.0 {
            continue;
        }

        for k in 0..3 {
            let r = groups[i * 3 + k] as usize * 4;

            let a = corners.get(i * 3 + k);
            let (pa, pb, pc) = (
                positions.at(a),
                positions.at(corners.get(i * 3 + NEXT[k])),
                positions.at(corners.get(i * 3 + NEXT[k + 1])),
            );

            let n = normals.at(a);

            // reproject tangent vector onto vertex normal
            let sd = t.x * n[0] + t.y * n[1] + t.z * n[2];
            let (sx, sy, sz) = (t.x - n[0] * sd, t.y - n[1] * sd, t.z - n[2] * sd);
            let sl = (sx * sx + sy * sy + sz * sz).sqrt();

            // compute incident angle for weighting, in tangent plane
            // note: this step is absent from the paper, reference implementation computes angle after projecting edges to tangent plane...
            let (mut dp1x, mut dp1y, mut dp1z) = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]);
            let (mut dp2x, mut dp2y, mut dp2z) = (pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]);

            let dp1d = dp1x * n[0] + dp1y * n[1] + dp1z * n[2];
            let dp2d = dp2x * n[0] + dp2y * n[1] + dp2z * n[2];

            dp1x -= n[0] * dp1d;
            dp1y -= n[1] * dp1d;
            dp1z -= n[2] * dp1d;
            dp2x -= n[0] * dp2d;
            dp2y -= n[1] * dp2d;
            dp2z -= n[2] * dp2d;

            let dp1l = dp1x * dp1x + dp1y * dp1y + dp1z * dp1z;
            let dp2l = dp2x * dp2x + dp2y * dp2y + dp2z * dp2z;
            let dpl = (dp1l * dp2l).sqrt();

            let cosa = (dp1x * dp2x + dp1y * dp2y + dp1z * dp2z) * (if dpl == 0.0 { 0.0 } else { 1.0 / dpl });
            let angle = optacos(cosa); // optacos handles clamping to [-1..1]

            // accumulate renormalized tangent weighted by angle
            let mut w = angle * (if sl == 0.0 { 0.0 } else { 1.0 / sl });

            // weight larger adjacent triangles more to reduce interpolation artifacts from slivers; this deviates from reference implementation for higher quality
            if (options & TANGENT_COMPATIBLE) == 0 {
                w *= dpl;
            }

            result[r] += sx * w;
            result[r + 1] += sy * w;
            result[r + 2] += sz * w;
        }
    }
}

fn accumulate_normals(result: &mut [f32], groups: &[u32], corners: Corners, face_normals: &[Normal], positions: Stream) {
    for (i, n) in face_normals.iter().enumerate() {
        for k in 0..3 {
            let r = groups[i * 3 + k] as usize * 3;

            let (pa, pb, pc) = (
                positions.at(corners.get(i * 3 + k)),
                positions.at(corners.get(i * 3 + NEXT[k])),
                positions.at(corners.get(i * 3 + NEXT[k + 1])),
            );

            // compute incident angle for weighting
            let (dp1x, dp1y, dp1z) = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]);
            let (dp2x, dp2y, dp2z) = (pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]);

            let dp1l = dp1x * dp1x + dp1y * dp1y + dp1z * dp1z;
            let dp2l = dp2x * dp2x + dp2y * dp2y + dp2z * dp2z;
            let dpl = (dp1l * dp2l).sqrt();

            let cosa = (dp1x * dp2x + dp1y * dp2y + dp1z * dp2z) * (if dpl == 0.0 { 0.0 } else { 1.0 / dpl });
            let angle = optacos(cosa); // optacos handles clamping to [-1..1]

            // accumulate normal weighted by angle and edge length product; this weighting scheme matches the default scheme used for tangents
            let w = angle * dpl;

            result[r] += n.x * w;
            result[r + 1] += n.y * w;
            result[r + 2] += n.z * w;
        }
    }
}

fn smooth_normals(result: &mut [f32], scratch: &mut [f32], groups: &[u32], face_normals: &[Normal], alpha: f32) {
    scratch.fill(0.0);

    // for each triangle, accumulate normal deltas alongside each edge in both directions
    for (i, fnormal) in face_normals.iter().enumerate() {
        // degenerate triangles may connect groups across hard edges
        if fnormal.id == 0 {
            continue;
        }

        for k in 0..3 {
            let ga = groups[i * 3 + k] as usize;
            let gb = groups[i * 3 + NEXT[k]] as usize;

            let na = [result[ga * 3], result[ga * 3 + 1], result[ga * 3 + 2]];
            let nb = [result[gb * 3], result[gb * 3 + 1], result[gb * 3 + 2]];

            // normal alignment is symmetric; we compute it once per edge and use dp^2 for stronger alignment
            let dp = na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2];
            let w = if dp > 0.0 { dp * dp } else { 0.0 };

            let (nx, ny, nz) = ((nb[0] - na[0]) * w, (nb[1] - na[1]) * w, (nb[2] - na[2]) * w);

            // accumulate deltas; the last component is used to compute delta averages below
            scratch[ga * 4] += nx;
            scratch[ga * 4 + 1] += ny;
            scratch[ga * 4 + 2] += nz;
            scratch[ga * 4 + 3] += 1.0;

            scratch[gb * 4] -= nx;
            scratch[gb * 4 + 1] -= ny;
            scratch[gb * 4 + 2] -= nz;
            scratch[gb * 4 + 3] += 1.0;
        }
    }

    // apply average delta to group roots and renormalize the results
    for (i, &g) in groups.iter().enumerate() {
        if g as usize == i && scratch[i * 4 + 3] > 0.0 {
            let r = &mut result[i * 3..i * 3 + 3];
            let s = &scratch[i * 4..i * 4 + 4];

            let sw = alpha / s[3];
            let (rx, ry, rz) = (r[0] + s[0] * sw, r[1] + s[1] * sw, r[2] + s[2] * sw);

            let rl = (rx * rx + ry * ry + rz * rz).sqrt();
            let rs = if rl == 0.0 { 0.0 } else { 1.0 / rl };

            r[0] = rx * rs;
            r[1] = ry * rs;
            r[2] = rz * rs;
        }
    }
}

fn validate_corners(indices: Option<&[u32]>, index_count: usize, vertex_count: usize) -> Result<Corners<'_>> {
    validate::triangles(index_count)?;
    ensure(index_count < (3usize << 30), Error::InvalidArgument("index_count is too large"))?;

    match indices {
        Some(ib) => Ok(Corners {
            indices: Some(validate::indices(ib, index_count, vertex_count)?),
        }),
        None => {
            ensure(
                index_count == vertex_count,
                Error::InvalidArgument("index_count must equal vertex_count without indices"),
            )?;
            Ok(Corners { indices: None })
        }
    }
}

/// Tangent space generator.
/// Computes per-corner tangent vectors; for each corner, computes normalized tangent vector (xyz) and orientation (w, +/-1).
/// Bitangent can be reconstructed via cross(normal, tangent.xyz) * tangent.w.
///
/// `result` must hold `index_count * 4` floats; `indices` can be `None` if the input is unindexed.
/// `options` is a combination of [`TANGENT_COMPATIBLE`] and [`TANGENT_ZERO_FALLBACK`].
pub fn generate_tangents<T: IndexType>(
    result: &mut [f32],
    indices: Option<&[T]>,
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    vertex_normals: &[f32],
    vertex_normals_stride: usize,
    vertex_uvs: &[f32],
    vertex_uvs_stride: usize,
    options: u32,
) -> Result<()> {
    let indices = index::input_opt(indices);
    generate_tangents_u32(
        result,
        indices.as_deref(),
        index_count,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        vertex_normals,
        vertex_normals_stride,
        vertex_uvs,
        vertex_uvs_stride,
        options,
    )
}

fn generate_tangents_u32(
    result: &mut [f32],
    indices: Option<&[u32]>,
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    vertex_normals: &[f32],
    vertex_normals_stride: usize,
    vertex_uvs: &[f32],
    vertex_uvs_stride: usize,
    options: u32,
) -> Result<()> {
    let corners = validate_corners(indices, index_count, vertex_count)?;
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    validate::floats(vertex_normals, vertex_count, vertex_normals_stride, 12)?;
    validate::floats(vertex_uvs, vertex_count, vertex_uvs_stride, 8)?;
    let result = validate::output(result, index_count * 4)?;

    let positions = Stream {
        data: vertex_positions,
        stride: vertex_positions_stride / 4,
    };
    let normals = Stream {
        data: vertex_normals,
        stride: vertex_normals_stride / 4,
    };
    let uvs = Stream {
        data: vertex_uvs,
        stride: vertex_uvs_stride / 4,
    };

    let face_count = index_count / 3;

    // compute vertex remap to unique vertex index
    let mut remap = vec![0u32; vertex_count];
    build_vertex_remap(
        &mut remap,
        &mut VertexHasherF {
            positions,
            attributes: Some((normals, uvs)),
        },
    );

    // build adjacency information
    let adjacency = build_corner_adjacency(corners, index_count, &remap, vertex_count);

    // compute per-triangle tangents
    let mut face_tangents = vec![Tangent::default(); face_count];
    compute_face_tangents(&mut face_tangents, corners, positions, uvs);

    // compute per-corner tangent groups: triangles adjacent to the same vertex with the same orientation are merged in strips
    let mut groups: Vec<u32> = (0..index_count as u32).collect();

    // compute per-face orientation groups: degenerate triangles are merged with adjacent non-degenerate ones and inherit their orientation, stored as a bitmask
    let mut facegroups: Vec<u32> = (0..face_count as u32).collect();
    let mut facesign: Vec<u8> = face_tangents
        .iter()
        .map(|t| u8::from(t.w > 0.0) | (u8::from(t.w < 0.0) << 1))
        .collect();

    for i in 0..vertex_count {
        let range = adjacency.offsets[i] as usize..adjacency.offsets[i + 1] as usize;
        if !range.is_empty() {
            merge_tangent_groups(&mut groups, &mut facegroups, &mut facesign, &adjacency.data[range], corners, &remap);
        }
    }

    for i in 0..index_count {
        groups[i] = follow2(&mut groups, i as u32);
    }

    // accumulate tangents into their own respective groups
    result.fill(0.0);
    accumulate_tangents(result, &groups, corners, &face_tangents, positions, normals, options);

    // finalize tangent signs for each face using facegroups
    for i in 0..face_count {
        let fg = follow2(&mut facegroups, i as u32) as usize;
        let w = if facesign[fg] & 1 != 0 { 1.0 } else { -1.0 }; // note: if face is degenerate, we canonicalize to -1 for consistency

        for k in 0..3 {
            result[(i * 3 + k) * 4 + 3] = w;
        }
    }

    // finalize tangent vectors by normalizing roots and propagating the rest
    for (i, &g) in groups.iter().enumerate() {
        if g as usize == i {
            let r = &mut result[i * 4..i * 4 + 3];
            let l = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            let s = if l == 0.0 { 0.0 } else { 1.0 / l };

            r[0] *= s;
            r[1] *= s;
            r[2] *= s;

            // for isolated degenerate triangles, use (1, 0, 0) tangent for consistency
            r[0] = if l == 0.0 && (options & TANGENT_ZERO_FALLBACK) == 0 {
                1.0
            } else {
                r[0]
            };
        }
    }

    for (i, &g) in groups.iter().enumerate() {
        if g as usize != i {
            result.copy_within(g as usize * 4..g as usize * 4 + 3, i * 4); // .w was set per face earlier
        }
    }

    Ok(())
}

/// Experimental: Normal generator.
/// Computes per-corner normal vectors; normals are averaged across soft edges, using crease angle to determine which edges are hard.
///
/// `result` must hold `index_count * 3` floats; `indices` can be `None` if the input is unindexed.
/// `crease_angle` is the angle in radians below which normals are averaged across edges (0..=pi);
/// `smoothing` is the amount of smoothing to apply as a post-process (recommended range 0..=5).
pub fn generate_normals<T: IndexType>(
    result: &mut [f32],
    indices: Option<&[T]>,
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    crease_angle: f32,
    smoothing: f32,
) -> Result<()> {
    let indices = index::input_opt(indices);
    generate_normals_u32(
        result,
        indices.as_deref(),
        index_count,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        crease_angle,
        smoothing,
    )
}

fn generate_normals_u32(
    result: &mut [f32],
    indices: Option<&[u32]>,
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    crease_angle: f32,
    smoothing: f32,
) -> Result<()> {
    let corners = validate_corners(indices, index_count, vertex_count)?;
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    ensure(
        (0.0..=core::f32::consts::PI).contains(&crease_angle),
        Error::InvalidArgument("crease_angle must be in 0..=pi"),
    )?;
    ensure(smoothing >= 0.0, Error::InvalidArgument("smoothing must be >= 0"))?;
    let result = validate::output(result, index_count * 3)?;

    let positions = Stream {
        data: vertex_positions,
        stride: vertex_positions_stride / 4,
    };

    let face_count = index_count / 3;
    let crease_cutoff = crease_angle.cos();

    // compute vertex remap to unique vertex index
    let mut remap = vec![0u32; vertex_count];
    build_vertex_remap(
        &mut remap,
        &mut VertexHasherF {
            positions,
            attributes: None,
        },
    );

    // build adjacency information
    let adjacency = build_corner_adjacency(corners, index_count, &remap, vertex_count);

    // compute per-triangle normals
    let mut face_normals = vec![Normal::default(); face_count];
    compute_face_normals(&mut face_normals, corners, positions);

    // compute per-corner normal groups: triangles adjacent to the same vertex with similar normals are merged in strips
    let mut groups: Vec<u32> = (0..index_count as u32).collect();

    for i in 0..vertex_count {
        let range = adjacency.offsets[i] as usize..adjacency.offsets[i + 1] as usize;
        if !range.is_empty() {
            merge_normal_groups(&mut groups, &adjacency.data[range], corners, &remap, &face_normals, crease_cutoff);
        }
    }

    for i in 0..index_count {
        groups[i] = follow2(&mut groups, i as u32);
    }

    // accumulate normals into their own respective groups
    result.fill(0.0);
    accumulate_normals(result, &groups, corners, &face_normals, positions);

    // normalize accumulated normals; we need to do this before smoothing as smoothing expects normalized inputs
    for (i, &g) in groups.iter().enumerate() {
        if g as usize == i {
            let r = &mut result[i * 3..i * 3 + 3];
            let l = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            let s = if l == 0.0 { 0.0 } else { 1.0 / l };

            r[0] *= s;
            r[1] *= s;
            r[2] *= s;
        }
    }

    if smoothing > 0.0 {
        let passes = (smoothing.ceil() as i32).min(10); // clamp to avoid pathological cases

        let mut scratch = vec![0.0f32; index_count * 4];

        for pass in 0..passes {
            let rem = smoothing - pass as f32;
            let alpha = 0.5 * (if rem < 1.0 { rem } else { 1.0 });

            smooth_normals(result, &mut scratch, &groups, &face_normals, alpha);
        }
    }

    // finalize normal vectors by propagating the rest
    for (i, &g) in groups.iter().enumerate() {
        if g as usize != i {
            result.copy_within(g as usize * 3..g as usize * 3 + 3, i * 3);
        }
    }

    Ok(())
}
