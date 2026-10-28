// This work is based on:
// Michael Garland and Paul S. Heckbert. Surface simplification using quadric error metrics. 1997
// Michael Garland. Quadric-based polygonal surface simplification. 1999
// Peter Lindstrom. Out-of-Core Simplification of Large Polygonal Models. 2000
// Matthias Teschner, Bruno Heidelberger, Matthias Mueller, Danat Pomeranets, Markus Gross. Optimized Spatial Hashing for Collision Detection of Deformable Objects. 2003
// Peter Van Sandt, Yannis Chronis, Jignesh M. Patel. Efficiently Searching In-Memory Sorted Arrays: Revenge of the Interpolation Search? 2019
// Hugues Hoppe. New Quadric Metric for Simplifying Meshes with Appearance Attributes. 1999
// Hugues Hoppe, Steve Marschner. Efficient Minimization of New Quadric Metric for Simplifying Meshes with Appearance Attributes. 2000

use crate::error::{Error, Result, ensure};
use crate::hash::{Hasher, hash_buckets, hash_lookup};
use crate::index::{self, IndexType};
use crate::validate;

/// Do not move vertices that are located on the topological border (vertices on triangle edges that don't have a paired triangle). Useful for simplifying portions of the larger mesh.
pub const SIMPLIFY_LOCK_BORDER: u32 = 1 << 0;
/// Improve simplification performance assuming input indices are a sparse subset of the mesh. Note that error becomes relative to subset extents.
pub const SIMPLIFY_SPARSE: u32 = 1 << 1;
/// Treat error limit and resulting error as absolute instead of relative to mesh extents.
pub const SIMPLIFY_ERROR_ABSOLUTE: u32 = 1 << 2;
/// Remove disconnected parts of the mesh during simplification incrementally, regardless of the topological restrictions inside components.
pub const SIMPLIFY_PRUNE: u32 = 1 << 3;
/// Produce more regular triangle sizes and shapes during simplification, at some cost to geometric and attribute quality.
pub const SIMPLIFY_REGULARIZE: u32 = 1 << 4;
/// Allow collapses across attribute discontinuities, except for vertices that are tagged with [`SIMPLIFY_VERTEX_PROTECT`] in vertex_lock.
pub const SIMPLIFY_PERMISSIVE: u32 = 1 << 5;
/// Produce more regular triangle sizes and shapes during simplification, at a small cost to geometric and attribute quality.
pub const SIMPLIFY_REGULARIZE_LIGHT: u32 = 1 << 6;
/// Experimental: Try to preserve fold lines between opposite-facing triangles, at a small performance cost.
pub const SIMPLIFY_PRESERVE_FOLDS: u32 = 1 << 7;
/// Experimental: Clamp attribute error to match position error scale and avoid extreme error values in areas with high attribute variance.
pub const SIMPLIFY_ERROR_CLAMPED: u32 = 1 << 8;

/// Do not move this vertex.
pub const SIMPLIFY_VERTEX_LOCK: u8 = 1 << 0;
/// Protect attribute discontinuity at this vertex; must be used together with [`SIMPLIFY_PERMISSIVE`] option.
pub const SIMPLIFY_VERTEX_PROTECT: u8 = 1 << 1;
/// Increase priority for this vertex, making it more likely that it's preserved during simplification.
pub const SIMPLIFY_VERTEX_PRIORITY: u8 = 1 << 2;

#[derive(Debug, Clone, Copy, Default)]
struct Edge {
    next: u32,
    prev: u32,
}

struct EdgeAdjacency {
    offsets: Vec<u32>,
    data: Vec<Edge>,
}

impl EdgeAdjacency {
    fn new(index_count: usize, vertex_count: usize) -> Self {
        EdgeAdjacency {
            offsets: vec![0; vertex_count + 1],
            data: vec![Edge::default(); index_count],
        }
    }

    fn edges(&self, v: u32) -> &[Edge] {
        &self.data[self.offsets[v as usize] as usize..self.offsets[v as usize + 1] as usize]
    }

    fn update(&mut self, indices: &[u32], vertex_count: usize, remap: Option<&[u32]>) {
        let map = |v: u32| remap.map_or(v, |r| r[v as usize]);

        // fill edge counts
        self.offsets.fill(0);

        for &i in indices {
            self.offsets[map(i) as usize + 1] += 1;
        }

        // fill offset table
        let mut offset = 0u32;

        for o in self.offsets[1..=vertex_count].iter_mut() {
            let count = *o;
            *o = offset;
            offset += count;
        }

        // fill edge data
        for tri in indices.as_chunks::<3>().0 {
            let (a, b, c) = (map(tri[0]), map(tri[1]), map(tri[2]));

            for (v, next, prev) in [(a, b, c), (b, c, a), (c, a, b)] {
                let slot = &mut self.offsets[v as usize + 1];
                self.data[*slot as usize] = Edge { next, prev };
                *slot += 1;
            }
        }

        // finalize offsets
        self.offsets[0] = 0;
    }
}

/// float3 position stream with optional sparse indirection
#[derive(Clone, Copy)]
struct Positions<'a> {
    data: &'a [f32],
    stride_float: usize,
    sparse_remap: Option<&'a [u32]>,
}

impl Positions<'_> {
    fn get(&self, index: u32) -> &[f32] {
        let ri = self.sparse_remap.map_or(index, |r| r[index as usize]) as usize;
        &self.data[ri * self.stride_float..ri * self.stride_float + 3]
    }
}

struct PositionHasher<'a> {
    positions: Positions<'a>,
}

impl Hasher<u32> for PositionHasher<'_> {
    fn hash(&self, index: &u32) -> usize {
        let key = self.positions.get(*index);
        let (mut x, mut y, mut z) = (key[0].to_bits(), key[1].to_bits(), key[2].to_bits());

        // replace negative zero with zero
        x = if x == 0x8000_0000 { 0 } else { x };
        y = if y == 0x8000_0000 { 0 } else { y };
        z = if z == 0x8000_0000 { 0 } else { z };

        // scramble bits to make sure that integer coordinates have entropy in lower bits
        x ^= x >> 17;
        y ^= y >> 17;
        z ^= z >> 17;

        // Optimized Spatial Hashing for Collision Detection of Deformable Objects
        (x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663) ^ z.wrapping_mul(83492791)) as usize
    }

    fn equal(&mut self, lhs: &u32, rhs: &u32) -> bool {
        let (lv, rv) = (self.positions.get(*lhs), self.positions.get(*rhs));

        lv[0] == rv[0] && lv[1] == rv[1] && lv[2] == rv[2]
    }
}

struct RemapHasher<'a> {
    remap: &'a [u32],
}

impl Hasher<u32> for RemapHasher<'_> {
    fn hash(&self, id: &u32) -> usize {
        id.wrapping_mul(0x5bd1e995) as usize
    }

    fn equal(&mut self, lhs: &u32, rhs: &u32) -> bool {
        self.remap[*lhs as usize] == *rhs
    }
}

fn build_position_remap(remap: &mut [u32], wedge: Option<&mut [u32]>, positions: Positions) {
    let vertex_count = remap.len();
    let mut hasher = PositionHasher { positions };

    let mut table = vec![!0u32; hash_buckets(vertex_count)];

    // build forward remap: for each vertex, which other (canonical) vertex does it map to?
    // we use position equivalence for this, and remap vertices to other existing vertices
    for (i, r) in remap.iter_mut().enumerate() {
        let index = i as u32;
        let entry = hash_lookup(&table, &mut hasher, &index, &!0u32);

        if table[entry] == !0u32 {
            table[entry] = index;
        }

        *r = table[entry];
    }

    let Some(wedge) = wedge else {
        return;
    };

    // build wedge table: for each vertex, which other vertex is the next wedge that also maps to the same vertex?
    // entries in table form a (cyclic) wedge loop per vertex; for manifold vertices, wedge[i] == remap[i] == i
    for (i, w) in wedge.iter_mut().enumerate() {
        *w = i as u32;
    }

    for i in 0..vertex_count {
        if remap[i] != i as u32 {
            let r = remap[i] as usize;

            wedge[i] = wedge[r];
            wedge[r] = i as u32;
        }
    }
}

/// Rewrites `indices` to dense vertex ids and returns the dense => sparse vertex mapping.
fn build_sparse_remap(indices: &mut [u32], vertex_count: usize) -> Vec<u32> {
    // use a bit set to compute the precise number of unique vertices
    let mut filter = vec![0u8; vertex_count.div_ceil(8)];

    let mut unique = 0;
    for &index in indices.iter() {
        let (byte, bit) = (index as usize / 8, 1u8 << (index % 8));
        unique += usize::from(filter[byte] & bit == 0);
        filter[byte] |= bit;
    }

    let mut remap = vec![0u32; unique];
    let mut offset = 0;

    // temporary map dense => sparse
    let mut revremap = vec![!0u32; hash_buckets(unique)];

    // fill remap, using revremap as a helper, and rewrite indices in the same pass
    for index in indices.iter_mut() {
        let entry = hash_lookup(&revremap, &mut RemapHasher { remap: &remap }, index, &!0u32);

        if revremap[entry] == !0u32 {
            remap[offset] = *index;
            revremap[entry] = offset as u32;
            offset += 1;
        }

        *index = revremap[entry];
    }

    remap
}

const KIND_MANIFOLD: u8 = 0; // not on an attribute seam, not on any boundary
const KIND_BORDER: u8 = 1; // not on an attribute seam, has exactly two open edges
const KIND_SEAM: u8 = 2; // on an attribute seam with exactly two attribute seam edges
const KIND_COMPLEX: u8 = 3; // none of the above; these vertices can move as long as all wedges move to the target vertex
const KIND_FRINGE: u8 = 4; // complex vertex on a geometric border; can only move to another fringe vertex
const KIND_LOCKED: u8 = 5; // none of the above; these vertices can't move

const KIND_COUNT: usize = 6;

// manifold vertices can collapse onto anything
// border/seam vertices can collapse onto border/seam respectively, or locked
// complex vertices can collapse onto complex/locked
// fringe vertices can only collapse onto fringe vertices; border, manifold and complex vertices can collapse onto them
// a rule of thumb is that collapsing kind A into kind B preserves the kind B in the target vertex
// for example, while we could collapse Complex into Manifold, this would mean the target vertex isn't Manifold anymore
const CAN_COLLAPSE: [[bool; KIND_COUNT]; KIND_COUNT] = [
    [true, true, true, true, true, true],
    [false, true, false, false, true, true],
    [false, false, true, false, false, true],
    [false, false, false, true, true, true],
    [false, false, false, false, true, false],
    [false, false, false, false, false, false],
];

// if a vertex is manifold or seam, adjoining edges are guaranteed to have an opposite edge
// note that for seam edges, the opposite edge isn't present in the attribute-based topology
// but is present if you consider a position-only mesh variant
// while many complex collapses have the opposite edge, since complex vertices collapse to the
// same wedge, keeping opposite edges separate improves the quality by considering both targets
const HAS_OPPOSITE: [[bool; KIND_COUNT]; KIND_COUNT] = [
    [true, true, true, true, true, true],
    [true, false, true, false, false, false],
    [true, true, true, false, false, true],
    [true, false, false, false, false, false],
    [true, false, false, false, false, false],
    [true, false, true, false, false, false],
];

fn can_collapse(k0: u8, k1: u8) -> bool {
    CAN_COLLAPSE[usize::from(k0)][usize::from(k1)]
}

fn has_opposite(k0: u8, k1: u8) -> bool {
    HAS_OPPOSITE[usize::from(k0)][usize::from(k1)]
}

fn has_edge(adjacency: &EdgeAdjacency, a: u32, b: u32) -> bool {
    adjacency.edges(a).iter().any(|e| e.next == b)
}

fn has_edge_wedge(adjacency: &EdgeAdjacency, a: u32, b: u32, remap: &[u32], wedge: &[u32]) -> bool {
    let mut v = a;

    loop {
        if adjacency.edges(v).iter().any(|e| remap[e.next as usize] == remap[b as usize]) {
            return true;
        }

        v = wedge[v as usize];
        if v == a {
            return false;
        }
    }
}

fn classify_vertices(
    result: &mut [u8],
    loop_: &mut [u32],
    loopback: &mut [u32],
    adjacency: &EdgeAdjacency,
    remap: &[u32],
    wedge: &[u32],
    vertex_lock: Option<&[u8]>,
    sparse_remap: Option<&[u32]>,
    options: u32,
) {
    let vertex_count = result.len();

    loop_.fill(!0u32);
    loopback.fill(!0u32);

    // incoming & outgoing open edges: ~0u if no open edges, i if there are more than 1
    // note that this is the same data as required in loop[] arrays; loop[] data is only used for border/seam by default
    // in permissive mode we also use it to guide complex-complex collapses, so we fill it for all vertices
    let openinc = loopback;
    let openout = loop_;

    for vertex in 0..vertex_count as u32 {
        for edge in adjacency.edges(vertex) {
            let target = edge.next;

            if target == vertex {
                // degenerate triangles have two distinct edges instead of three, and the self edge
                // is bi-directional by definition; this can break border/seam classification by "closing"
                // the open edge from another triangle and falsely marking the vertex as manifold
                // instead we mark the vertex as having >1 open edges which turns it into locked/complex
                openinc[vertex as usize] = vertex;
                openout[vertex as usize] = vertex;
            } else if !has_edge(adjacency, target, vertex) {
                let t = target as usize;
                openinc[t] = if openinc[t] == !0u32 { vertex } else { target };
                openout[vertex as usize] = if openout[vertex as usize] == !0u32 { target } else { vertex };
            }
        }
    }

    for i in 0..vertex_count {
        let iu = i as u32;

        if remap[i] == iu {
            if wedge[i] == iu {
                // no attribute seam, need to check if it's manifold
                let (openi, openo) = (openinc[i], openout[i]);

                // note: we classify any vertices with no open edges as manifold
                // this is technically incorrect - if 4 triangles share an edge, we'll classify vertices as manifold
                // it's unclear if this is a problem in practice
                result[i] = if openi == !0u32 && openo == !0u32 {
                    KIND_MANIFOLD
                } else if openi != !0u32 && openo != !0u32 && remap[openi as usize] == remap[openo as usize] && openi != iu {
                    // classify half-seams as seams (the branch below would mis-classify them as borders)
                    // half-seam is a single vertex that connects to both vertices of a potential seam
                    // treating these as seams allows collapsing the "full" seam vertex onto them
                    KIND_SEAM
                } else if openi != iu && openo != iu {
                    KIND_BORDER
                } else {
                    KIND_LOCKED
                };
            } else if wedge[wedge[i] as usize] == iu {
                // attribute seam; need to distinguish between Seam and Locked
                let w = wedge[i];
                let (openiv, openov) = (openinc[i], openout[i]);
                let (openiw, openow) = (openinc[w as usize], openout[w as usize]);

                // seam should have one open half-edge for each vertex, and the edges need to "connect" - point to the same vertex post-remap
                result[i] = if openiv != !0u32
                    && openiv != iu
                    && openov != !0u32
                    && openov != iu
                    && openiw != !0u32
                    && openiw != w
                    && openow != !0u32
                    && openow != w
                    && remap[openiv as usize] == remap[openow as usize]
                    && remap[openov as usize] == remap[openiw as usize]
                    && remap[openiv as usize] != remap[openov as usize]
                {
                    KIND_SEAM
                } else {
                    // mismatched seam endpoints or disconnected seam; we don't have classification available
                    KIND_LOCKED
                };
            } else {
                // more than one vertex maps to this one; we don't have classification available
                result[i] = KIND_LOCKED;
            }
        } else {
            result[i] = result[remap[i] as usize];
        }
    }

    let sparse = |v: u32| sparse_remap.map_or(v, |r| r[v as usize]) as usize;

    if options & SIMPLIFY_PERMISSIVE != 0 {
        for i in 0..vertex_count {
            if result[i] == KIND_SEAM || result[i] == KIND_LOCKED {
                if remap[i] != i as u32 {
                    // only process primary vertices; wedges will be updated to match the primary vertex
                    result[i] = result[remap[i] as usize];
                    continue;
                }

                let iu = i as u32;
                let mut v = iu;

                // vertex_lock may protect any wedge, not just the primary vertex, so we switch to complex only if no wedges are protected
                let mut protect = false;
                loop {
                    protect |= vertex_lock.is_some_and(|l| l[sparse(v)] & SIMPLIFY_VERTEX_PROTECT != 0);
                    v = wedge[v as usize];
                    if v == iu {
                        break;
                    }
                }

                // check if any adjoining edge doesn't have an opposite edge (indicating vertex is on the border)
                let mut border = false;
                loop {
                    for e in adjacency.edges(v) {
                        border |= !has_edge_wedge(adjacency, e.next, v, remap, wedge);
                    }
                    v = wedge[v as usize];
                    if v == iu {
                        break;
                    }
                }

                // border vertices with attribute discontinuities can only collapse along the border, so they need a separate tagging
                result[i] = if protect {
                    result[i]
                } else if border {
                    KIND_FRINGE
                } else {
                    KIND_COMPLEX
                };
            }
        }
    }

    if let Some(lock) = vertex_lock {
        // vertex_lock may lock any wedge, not just the primary vertex, so we need to lock the primary vertex and relock any wedges
        for i in 0..vertex_count {
            if lock[sparse(i as u32)] & SIMPLIFY_VERTEX_LOCK != 0 {
                result[remap[i] as usize] = KIND_LOCKED;
            }
        }

        for i in 0..vertex_count {
            if result[remap[i] as usize] == KIND_LOCKED {
                result[i] = KIND_LOCKED;
            }
        }
    }

    if options & SIMPLIFY_LOCK_BORDER != 0 {
        for r in result.iter_mut() {
            if *r == KIND_BORDER || *r == KIND_FRINGE {
                *r = KIND_LOCKED;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Vector3 {
    x: f32,
    y: f32,
    z: f32,
}

/// Returns `(extent, offset)`; fills `result` with positions normalized to the unit cube when provided.
fn rescale_positions(result: Option<&mut [Vector3]>, positions: Positions, vertex_count: usize) -> (f32, [f32; 3]) {
    let mut minv = [f32::MAX; 3];
    let mut maxv = [-f32::MAX; 3];

    for i in 0..vertex_count {
        let v = positions.get(i as u32);

        for j in 0..3 {
            let vj = v[j];

            minv[j] = if minv[j] > vj { vj } else { minv[j] };
            maxv[j] = if maxv[j] < vj { vj } else { maxv[j] };
        }
    }

    let mut extent = 0.0f32;

    extent = if (maxv[0] - minv[0]) < extent { extent } else { maxv[0] - minv[0] };
    extent = if (maxv[1] - minv[1]) < extent { extent } else { maxv[1] - minv[1] };
    extent = if (maxv[2] - minv[2]) < extent { extent } else { maxv[2] - minv[2] };

    if let Some(result) = result {
        let scale = if extent == 0.0 { 0.0 } else { 1.0 / extent };

        for (i, r) in result.iter_mut().enumerate().take(vertex_count) {
            let v = positions.get(i as u32);

            r.x = (v[0] - minv[0]) * scale;
            r.y = (v[1] - minv[1]) * scale;
            r.z = (v[2] - minv[2]) * scale;
        }
    }

    (extent, minv)
}

/// float attribute stream with sparse indirection
#[derive(Clone, Copy)]
struct Attributes<'a> {
    data: &'a [f32],
    stride_float: usize,
    weights: &'a [f32],
}

fn rescale_attributes(
    result: &mut [f32],
    attributes: Attributes,
    vertex_count: usize,
    attribute_count: usize,
    attribute_remap: &[u32],
    sparse_remap: Option<&[u32]>,
) {
    for i in 0..vertex_count {
        let ri = sparse_remap.map_or(i as u32, |r| r[i]) as usize;

        for k in 0..attribute_count {
            let rk = attribute_remap[k] as usize;
            let a = attributes.data[ri * attributes.stride_float + rk];

            result[i * attribute_count + k] = a * attributes.weights[rk];
        }
    }
}

const MAX_ATTRIBUTES: usize = 32;

#[derive(Debug, Clone, Copy, Default)]
struct Quadric {
    // a00*x^2 + a11*y^2 + a22*z^2 + 2*a10*xy + 2*a20*xz + 2*a21*yz + 2*b0*x + 2*b1*y + 2*b2*z + c
    a00: f32,
    a11: f32,
    a22: f32,
    a10: f32,
    a20: f32,
    a21: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    c: f32,
    w: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct QuadricGrad {
    // gx*x + gy*y + gz*z + gw
    gx: f32,
    gy: f32,
    gz: f32,
    gw: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct Reservoir {
    x: f32,
    y: f32,
    z: f32,
    r: f32,
    g: f32,
    b: f32,
    w: f32,
}

/// `value` holds `bidi` (u32) during picking and `error` (f32 bits) after ranking (C union).
#[derive(Debug, Clone, Copy, Default)]
struct Collapse {
    v0: u32,
    v1: u32,
    value: u32,
}

impl Collapse {
    fn error(&self) -> f32 {
        f32::from_bits(self.value)
    }
}

fn normalize(v: &mut Vector3) -> f32 {
    let length = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();

    if length > 0.0 {
        v.x /= length;
        v.y /= length;
        v.z /= length;
    }

    length
}

fn quadric_add(q: &mut Quadric, r: &Quadric) {
    q.a00 += r.a00;
    q.a11 += r.a11;
    q.a22 += r.a22;
    q.a10 += r.a10;
    q.a20 += r.a20;
    q.a21 += r.a21;
    q.b0 += r.b0;
    q.b1 += r.b1;
    q.b2 += r.b2;
    q.c += r.c;
    q.w += r.w;
}

fn quadric_add_grad(g: &mut QuadricGrad, r: &QuadricGrad) {
    g.gx += r.gx;
    g.gy += r.gy;
    g.gz += r.gz;
    g.gw += r.gw;
}

fn quadric_add_grads(g: &mut [QuadricGrad], r: &[QuadricGrad]) {
    for (gk, rk) in g.iter_mut().zip(r) {
        quadric_add_grad(gk, rk);
    }
}

fn quadric_eval(q: &Quadric, v: &Vector3) -> f32 {
    let mut rx = (q.b0 + q.a10 * v.y) * 2.0;
    let mut ry = (q.b1 + q.a21 * v.z) * 2.0;
    let mut rz = (q.b2 + q.a20 * v.x) * 2.0;

    rx += q.a00 * v.x;
    ry += q.a11 * v.y;
    rz += q.a22 * v.z;

    q.c + rx * v.x + ry * v.y + rz * v.z
}

fn quadric_error(q: &Quadric, v: &Vector3) -> f32 {
    let r = quadric_eval(q, v);
    let s = if q.w == 0.0 { 0.0 } else { 1.0 / q.w };

    r.abs() * s
}

fn quadric_error_attr(q: &Quadric, g: &[QuadricGrad], v: &Vector3, va: &[f32], clamped: bool) -> f32 {
    let mut r = quadric_eval(q, v);

    // see quadricFromAttributes for general derivation; here we need to add the parts of (eval(pos) - attr)^2 that depend on attr
    for (gk, &a) in g.iter().zip(va) {
        let gv = v.x * gk.gx + v.y * gk.gy + v.z * gk.gz + gk.gw;

        r += a * (a * q.w - 2.0 * gv);
    }

    // note: unlike position error, we do not normalize by Q.w to retain edge scaling as described in quadricFromAttributes
    // we do support clamping, with the error limited to the cumulative weight (= area), which makes perceptual error map to position error more closely
    let e = r.abs();
    if clamped { if e < q.w { e } else { q.w } } else { e }
}

fn quadric_from_plane(a: f32, b: f32, c: f32, d: f32, w: f32) -> Quadric {
    let aw = a * w;
    let bw = b * w;
    let cw = c * w;
    let dw = d * w;

    Quadric {
        a00: a * aw,
        a11: b * bw,
        a22: c * cw,
        a10: a * bw,
        a20: a * cw,
        a21: b * cw,
        b0: a * dw,
        b1: b * dw,
        b2: c * dw,
        c: d * dw,
        w,
    }
}

fn quadric_from_point(x: f32, y: f32, z: f32, w: f32) -> Quadric {
    Quadric {
        a00: w,
        a11: w,
        a22: w,
        a10: 0.0,
        a20: 0.0,
        a21: 0.0,
        b0: -x * w,
        b1: -y * w,
        b2: -z * w,
        c: (x * x + y * y + z * z) * w,
        w,
    }
}

fn sub(a: &Vector3, b: &Vector3) -> Vector3 {
    Vector3 {
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z,
    }
}

fn cross(a: &Vector3, b: &Vector3) -> Vector3 {
    Vector3 {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
    }
}

fn quadric_from_triangle(p0: &Vector3, p1: &Vector3, p2: &Vector3, weight: f32) -> Quadric {
    let p10 = sub(p1, p0);
    let p20 = sub(p2, p0);

    // normal = cross(p1 - p0, p2 - p0)
    let mut normal = cross(&p10, &p20);
    let area = normalize(&mut normal);

    let distance = normal.x * p0.x + normal.y * p0.y + normal.z * p0.z;

    // we use sqrtf(area) so that the error is scaled linearly; this tends to improve silhouettes
    quadric_from_plane(normal.x, normal.y, normal.z, -distance, area.sqrt() * weight)
}

fn quadric_from_triangle_edge(p0: &Vector3, p1: &Vector3, p2: &Vector3, weight: f32) -> Quadric {
    let p10 = sub(p1, p0);

    // edge length; keep squared length around for projection correction
    let lengthsq = p10.x * p10.x + p10.y * p10.y + p10.z * p10.z;
    let length = lengthsq.sqrt();

    // p20p = length of projection of p2-p0 onto p1-p0; note that p10 is unnormalized so we need to correct it later
    let p20 = sub(p2, p0);
    let p20p = p20.x * p10.x + p20.y * p10.y + p20.z * p10.z;

    // perp = perpendicular vector from p2 to line segment p1-p0
    // note: since p10 is unnormalized we need to correct the projection; we scale p20 instead to take advantage of normalize below
    let mut perp = Vector3 {
        x: p20.x * lengthsq - p10.x * p20p,
        y: p20.y * lengthsq - p10.y * p20p,
        z: p20.z * lengthsq - p10.z * p20p,
    };
    normalize(&mut perp);

    let distance = perp.x * p0.x + perp.y * p0.y + perp.z * p0.z;

    // note: the weight is scaled linearly with edge length; this has to match the triangle weight
    quadric_from_plane(perp.x, perp.y, perp.z, -distance, length * weight)
}

fn quadric_from_attributes(
    g: &mut [QuadricGrad],
    p0: &Vector3,
    p1: &Vector3,
    p2: &Vector3,
    va0: &[f32],
    va1: &[f32],
    va2: &[f32],
) -> Quadric {
    // for each attribute we want to encode the following function into the quadric:
    // (eval(pos) - attr)^2
    // where eval(pos) interpolates attribute across the triangle like so:
    // eval(pos) = pos.x * gx + pos.y * gy + pos.z * gz + gw
    // where gx/gy/gz/gw are gradients
    let p10 = sub(p1, p0);
    let p20 = sub(p2, p0);

    // normal = cross(p1 - p0, p2 - p0)
    let normal = cross(&p10, &p20);
    let area = (normal.x * normal.x + normal.y * normal.y + normal.z * normal.z).sqrt() * 0.5;

    // quadric is weighted with the square of edge length (= area)
    // this equalizes the units with the positional error (which, after normalization, is a square of distance)
    // as a result, a change in weighted attribute of 1 along distance d is approximately equivalent to a change in position of d
    let w = area;

    // we compute gradients using barycentric coordinates; barycentric coordinates can be computed as follows:
    // v = (d11 * d20 - d01 * d21) / denom
    // w = (d00 * d21 - d01 * d20) / denom
    // u = 1 - v - w
    // here v0, v1 are triangle edge vectors, v2 is a vector from point to triangle corner, and dij = dot(vi, vj)
    // note: v2 and d20/d21 can not be evaluated here as v2 is effectively an unknown variable; we need these only as variables for derivation of gradients
    let (v0, v1) = (&p10, &p20);
    let d00 = v0.x * v0.x + v0.y * v0.y + v0.z * v0.z;
    let d01 = v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
    let d11 = v1.x * v1.x + v1.y * v1.y + v1.z * v1.z;
    let denom = d00 * d11 - d01 * d01;
    let denomr = if denom == 0.0 { 0.0 } else { 1.0 / denom };

    // precompute gradient factors
    // these are derived by directly computing derivative of eval(pos) = a0 * u + a1 * v + a2 * w and factoring out expressions that are shared between attributes
    let gx1 = (d11 * v0.x - d01 * v1.x) * denomr;
    let gx2 = (d00 * v1.x - d01 * v0.x) * denomr;
    let gy1 = (d11 * v0.y - d01 * v1.y) * denomr;
    let gy2 = (d00 * v1.y - d01 * v0.y) * denomr;
    let gz1 = (d11 * v0.z - d01 * v1.z) * denomr;
    let gz2 = (d00 * v1.z - d01 * v0.z) * denomr;

    let mut q = Quadric { w, ..Quadric::default() };

    for (k, gk) in g.iter_mut().enumerate() {
        let (a0, a1, a2) = (va0[k], va1[k], va2[k]);

        // compute gradient of eval(pos) for x/y/z/w
        // the formulas below are obtained by directly computing derivative of eval(pos) = a0 * u + a1 * v + a2 * w
        let gx = gx1 * (a1 - a0) + gx2 * (a2 - a0);
        let gy = gy1 * (a1 - a0) + gy2 * (a2 - a0);
        let gz = gz1 * (a1 - a0) + gz2 * (a2 - a0);
        let gw = a0 - p0.x * gx - p0.y * gy - p0.z * gz;

        // quadric encodes (eval(pos)-attr)^2; this means that the resulting expansion needs to compute, for example, pos.x * pos.y * K
        // since quadrics already encode factors for pos.x * pos.y, we can accumulate almost everything in basic quadric fields
        // note: for simplicity we scale all factors by weight here instead of outside the loop
        q.a00 += w * (gx * gx);
        q.a11 += w * (gy * gy);
        q.a22 += w * (gz * gz);

        q.a10 += w * (gy * gx);
        q.a20 += w * (gz * gx);
        q.a21 += w * (gz * gy);

        q.b0 += w * (gx * gw);
        q.b1 += w * (gy * gw);
        q.b2 += w * (gz * gw);

        q.c += w * (gw * gw);

        // the only remaining sum components are ones that depend on attr; these will be added during error evaluation, see quadricError
        *gk = QuadricGrad {
            gx: w * gx,
            gy: w * gy,
            gz: w * gz,
            gw: w * gw,
        };
    }

    q
}

fn quadric_volume_gradient(p0: &Vector3, p1: &Vector3, p2: &Vector3) -> QuadricGrad {
    let p10 = sub(p1, p0);
    let p20 = sub(p2, p0);

    // normal = cross(p1 - p0, p2 - p0)
    let mut normal = cross(&p10, &p20);
    let area = normalize(&mut normal) * 0.5;

    QuadricGrad {
        gx: normal.x * area,
        gy: normal.y * area,
        gz: normal.z * area,
        gw: (-p0.x * normal.x - p0.y * normal.y - p0.z * normal.z) * area,
    }
}

fn quadric_solve(p: &mut Vector3, q: &Quadric, gv: &QuadricGrad) -> bool {
    // solve A*p = -b where A is the quadric matrix and b is the linear term
    let (a00, a11, a22) = (q.a00, q.a11, q.a22);
    let (a10, a20, a21) = (q.a10, q.a20, q.a21);
    let (x0, x1, x2) = (-q.b0, -q.b1, -q.b2);

    let eps = 1e-6 * q.w;

    // LDL decomposition: A = LDL^T
    let d0 = a00;
    let l10 = a10 / d0;
    let l20 = a20 / d0;

    let d1 = a11 - a10 * l10;
    let dl21 = a21 - a20 * l10;
    let l21 = dl21 / d1;

    let d2 = a22 - a20 * l20 - dl21 * l21;

    // solve L*y = x
    let y0 = x0;
    let y1 = x1 - l10 * y0;
    let y2 = x2 - l20 * y0 - l21 * y1;

    // solve D*z = y
    let z0 = y0 / d0;
    let z1 = y1 / d1;
    let z2 = y2 / d2;

    // augment system with linear constraint GV using Lagrange multiplier
    let (a30, a31, a32) = (gv.gx, gv.gy, gv.gz);
    let x3 = -gv.gw;

    let l30 = a30 / d0;
    let dl31 = a31 - a30 * l10;
    let l31 = dl31 / d1;
    let dl32 = a32 - a30 * l20 - dl31 * l21;
    let l32 = dl32 / d2;
    let d3 = 0.0 - a30 * l30 - dl31 * l31 - dl32 * l32;

    let y3 = x3 - l30 * y0 - l31 * y1 - l32 * y2;
    let z3 = if d3.abs() > eps { y3 / d3 } else { 0.0 }; // if d3 is zero, we can ignore the constraint

    // substitute L^T*p = z
    let lambda = z3;
    let pz = z2 - l32 * lambda;
    let py = z1 - l21 * pz - l31 * lambda;
    let px = z0 - l10 * py - l20 * pz - l30 * lambda;

    *p = Vector3 { x: px, y: py, z: pz };

    d0.abs() > eps && d1.abs() > eps && d2.abs() > eps
}

fn quadric_reduce_attributes(q: &mut Quadric, a: &Quadric, g: &[QuadricGrad]) {
    // update vertex quadric with attribute quadric; multiply by vertex weight to minimize normalized error
    q.a00 += a.a00 * q.w;
    q.a11 += a.a11 * q.w;
    q.a22 += a.a22 * q.w;
    q.a10 += a.a10 * q.w;
    q.a20 += a.a20 * q.w;
    q.a21 += a.a21 * q.w;
    q.b0 += a.b0 * q.w;
    q.b1 += a.b1 * q.w;
    q.b2 += a.b2 * q.w;

    let iaw = if a.w == 0.0 { 0.0 } else { q.w / a.w };

    // update linear system based on attribute gradients (BB^T/a)
    for gk in g {
        q.a00 -= (gk.gx * gk.gx) * iaw;
        q.a11 -= (gk.gy * gk.gy) * iaw;
        q.a22 -= (gk.gz * gk.gz) * iaw;
        q.a10 -= (gk.gx * gk.gy) * iaw;
        q.a20 -= (gk.gx * gk.gz) * iaw;
        q.a21 -= (gk.gy * gk.gz) * iaw;

        q.b0 -= (gk.gx * gk.gw) * iaw;
        q.b1 -= (gk.gy * gk.gw) * iaw;
        q.b2 -= (gk.gz * gk.gw) * iaw;
    }
}

fn fill_face_quadrics(
    vertex_quadrics: &mut [Quadric],
    mut volume_gradients: Option<&mut [QuadricGrad]>,
    indices: &[u32],
    vertex_positions: &[Vector3],
    remap: &[u32],
) {
    for tri in indices.as_chunks::<3>().0 {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);

        let q = quadric_from_triangle(&vertex_positions[i0], &vertex_positions[i1], &vertex_positions[i2], 1.0);

        for i in [i0, i1, i2] {
            quadric_add(&mut vertex_quadrics[remap[i] as usize], &q);
        }

        if let Some(gradients) = volume_gradients.as_deref_mut() {
            let gv = quadric_volume_gradient(&vertex_positions[i0], &vertex_positions[i1], &vertex_positions[i2]);

            for i in [i0, i1, i2] {
                quadric_add_grad(&mut gradients[remap[i] as usize], &gv);
            }
        }
    }
}

fn fill_vertex_quadrics(
    vertex_quadrics: &mut [Quadric],
    vertex_positions: &[Vector3],
    remap: &[u32],
    vertex_lock: Option<&[u8]>,
    sparse_remap: Option<&[u32]>,
    options: u32,
) {
    // by default, we use a very small weight to improve triangulation and numerical stability without affecting the shape or error
    let factor = if options & SIMPLIFY_REGULARIZE_LIGHT != 0 {
        1e-2
    } else if options & SIMPLIFY_REGULARIZE != 0 {
        1e-1
    } else {
        1e-7
    };

    for (i, q) in vertex_quadrics.iter_mut().enumerate() {
        if remap[i] != i as u32 {
            continue;
        }

        // increase regularization weight for vertices marked as priority; for now we only examine the primary vertex
        let ri = sparse_remap.map_or(i as u32, |r| r[i]) as usize;
        let priority = vertex_lock.is_some_and(|l| l[ri] & SIMPLIFY_VERTEX_PRIORITY != 0);

        let p = &vertex_positions[i];
        let w = q.w * if priority { 1.0 } else { factor };

        quadric_add(q, &quadric_from_point(p.x, p.y, p.z, w));
    }
}

fn fill_edge_quadrics(
    vertex_quadrics: &mut [Quadric],
    indices: &[u32],
    vertex_positions: &[Vector3],
    remap: &[u32],
    vertex_kind: &[u8],
    loop_: &[u32],
    loopback: &[u32],
) {
    const NEXT: [usize; 4] = [1, 2, 0, 1];

    for tri in indices.as_chunks::<3>().0 {
        for e in 0..3 {
            let i0 = tri[e];
            let i1 = tri[NEXT[e]];

            let k0 = vertex_kind[i0 as usize];
            let k1 = vertex_kind[i1 as usize];

            // early out: manifold vertices never lie on a border
            if k0 == KIND_MANIFOLD || k1 == KIND_MANIFOLD {
                continue;
            }

            // one of the vertices must be border/seam/fringe; additional checks help validate the border
            if k0 != KIND_BORDER && k0 != KIND_SEAM && k0 != KIND_FRINGE && k1 != KIND_BORDER && k1 != KIND_SEAM && k1 != KIND_FRINGE {
                continue;
            }

            // if i0/i1 are border/seam we check if they are on the same edge loop
            // note that we need to add the error even for edges that connect e.g. border & locked
            // if we don't do that, the adjacent border->border edge won't have correct errors for corners
            if (k0 == KIND_BORDER || k0 == KIND_SEAM) && loop_[i0 as usize] != i1 {
                continue;
            }

            if (k1 == KIND_BORDER || k1 == KIND_SEAM) && loopback[i1 as usize] != i0 {
                continue;
            }

            // fringe vertices don't have proper edge loops, so we need to add edge quadrics to fringe-fringe or fringe-locked edges
            // note that border-fringe edges are still processed, guarded by the loop check above
            if (k0 == KIND_FRINGE || k1 == KIND_FRINGE) && (k0 == KIND_COMPLEX || k1 == KIND_COMPLEX) {
                continue;
            }

            let i2 = tri[NEXT[e + 1]];

            // we try hard to maintain border edge geometry; seam edges can move more freely
            // due to topological restrictions on collapses, seam quadrics slightly improve collapse structure but aren't critical
            const EDGE_WEIGHT_SEAM: f32 = 0.5; // applied twice due to opposite edges
            const EDGE_WEIGHT_BORDER: f32 = 10.0;

            let edge_weight = if k0 == KIND_SEAM || k1 == KIND_SEAM {
                EDGE_WEIGHT_SEAM
            } else {
                EDGE_WEIGHT_BORDER
            };

            let (p0, p1, p2) = (
                &vertex_positions[i0 as usize],
                &vertex_positions[i1 as usize],
                &vertex_positions[i2 as usize],
            );

            let mut q = quadric_from_triangle_edge(p0, p1, p2, edge_weight);

            let mut qt = quadric_from_triangle(p0, p1, p2, edge_weight);

            // mix edge quadric with triangle quadric to stabilize collapses in both directions; both quadrics inherit edge weight so that their error is added
            qt.w = 0.0;
            quadric_add(&mut q, &qt);

            quadric_add(&mut vertex_quadrics[remap[i0 as usize] as usize], &q);
            quadric_add(&mut vertex_quadrics[remap[i1 as usize] as usize], &q);
        }
    }
}

fn opposite_vertex(adjacency: &EdgeAdjacency, a: u32, b: u32, remap: &[u32], wedge: &[u32]) -> u32 {
    let mut v = a;
    let mut r = !0u32;

    loop {
        for e in adjacency.edges(v) {
            if remap[e.next as usize] == remap[b as usize] {
                r = if r == !0u32 { e.prev } else { a };
            }
        }

        v = wedge[v as usize];
        if v == a {
            break;
        }
    }

    // we only return the opposite vertex if it's unique
    if r == a { !0u32 } else { r }
}

fn fill_fold_quadrics(
    vertex_quadrics: &mut [Quadric],
    adjacency: &EdgeAdjacency,
    vertex_positions: &[Vector3],
    remap: &[u32],
    wedge: &[u32],
) {
    for i0 in 0..vertex_positions.len() {
        for e in adjacency.edges(i0 as u32) {
            let (i1, i2) = (e.next, e.prev);

            // since we only process paired edges each edge should occur twice; skip redundant edges
            if remap[i1 as usize] > remap[i0] {
                continue;
            }

            let i3 = opposite_vertex(adjacency, i1, i0 as u32, remap, wedge);

            if i3 == !0u32 {
                continue;
            }

            let (p0, p1, p2, p3) = (
                &vertex_positions[i0],
                &vertex_positions[i1 as usize],
                &vertex_positions[i2 as usize],
                &vertex_positions[i3 as usize],
            );

            let p10 = sub(p1, p0);
            let p20 = sub(p2, p0);
            let p30 = sub(p3, p0);

            let normal = cross(&p10, &p20);
            let opposite = cross(&p30, &p10);
            let nl = (normal.x * normal.x + normal.y * normal.y + normal.z * normal.z).sqrt();
            let ol = (opposite.x * opposite.x + opposite.y * opposite.y + opposite.z * opposite.z).sqrt();

            // identify folds by checking dihedral angle against a somewhat arbitrary ~30 degree cutoff
            if normal.x * opposite.x + normal.y * opposite.y + normal.z * opposite.z >= -0.85 * nl * ol {
                continue;
            }

            // use both triangles for symmetry since we won't visit this edge again
            let mut q = quadric_from_triangle_edge(p0, p1, p2, 1.0);
            let qo = quadric_from_triangle_edge(p0, p1, p3, 1.0);
            quadric_add(&mut q, &qo);

            quadric_add(&mut vertex_quadrics[remap[i0] as usize], &q);
            quadric_add(&mut vertex_quadrics[remap[i1 as usize] as usize], &q);
        }
    }
}

fn fill_attribute_quadrics(
    attribute_quadrics: &mut [Quadric],
    attribute_gradients: &mut [QuadricGrad],
    indices: &[u32],
    vertex_positions: &[Vector3],
    vertex_attributes: &[f32],
    attribute_count: usize,
) {
    let mut g = [QuadricGrad::default(); MAX_ATTRIBUTES];
    let g = &mut g[..attribute_count];

    for tri in indices.as_chunks::<3>().0 {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);

        let attr = |i: usize| &vertex_attributes[i * attribute_count..(i + 1) * attribute_count];

        let qa = quadric_from_attributes(
            g,
            &vertex_positions[i0],
            &vertex_positions[i1],
            &vertex_positions[i2],
            attr(i0),
            attr(i1),
            attr(i2),
        );

        for i in [i0, i1, i2] {
            quadric_add(&mut attribute_quadrics[i], &qa);
            quadric_add_grads(&mut attribute_gradients[i * attribute_count..(i + 1) * attribute_count], g);
        }
    }
}

// does triangle ABC flip when C is replaced with D?
fn has_triangle_flip(a: &Vector3, b: &Vector3, c: &Vector3, d: &Vector3) -> bool {
    let eb = sub(b, a);
    let ec = sub(c, a);
    let ed = sub(d, a);

    let nbc = cross(&eb, &ec);
    let nbd = cross(&eb, &ed);

    let ndp = nbc.x * nbd.x + nbc.y * nbd.y + nbc.z * nbd.z;
    let abc = nbc.x * nbc.x + nbc.y * nbc.y + nbc.z * nbc.z;
    let abd = nbd.x * nbd.x + nbd.y * nbd.y + nbd.z * nbd.z;

    // scale is cos(angle); somewhat arbitrarily set to ~75 degrees
    // note that the "pure" check is ndp <= 0 (90 degree cutoff) but that allows flipping through a series of close-to-90 collapses
    ndp <= 0.25 * (abc * abd).sqrt()
}

fn has_triangle_flips_collapse(adjacency: &EdgeAdjacency, vertex_positions: &[Vector3], collapse_remap: &[u32], i0: u32, i1: u32) -> bool {
    let v0 = &vertex_positions[i0 as usize];
    let v1 = &vertex_positions[i1 as usize];

    for e in adjacency.edges(i0) {
        let a = collapse_remap[e.next as usize];
        let b = collapse_remap[e.prev as usize];

        // skip triangles that will get collapsed by i0->i1 collapse or already got collapsed previously
        if a == i1 || b == i1 || a == b {
            continue;
        }

        // early-out when at least one triangle flips due to a collapse
        if has_triangle_flip(&vertex_positions[a as usize], &vertex_positions[b as usize], v0, v1) {
            return true;
        }
    }

    false
}

fn has_triangle_flips_position(adjacency: &EdgeAdjacency, vertex_positions: &[Vector3], i0: u32, v1: &Vector3) -> bool {
    let v0 = &vertex_positions[i0 as usize];

    adjacency
        .edges(i0)
        .iter()
        .any(|e| has_triangle_flip(&vertex_positions[e.next as usize], &vertex_positions[e.prev as usize], v0, v1))
}

fn get_neighborhood_radius(adjacency: &EdgeAdjacency, vertex_positions: &[Vector3], i0: u32) -> f32 {
    let v0 = &vertex_positions[i0 as usize];

    let mut result = 0.0f32;

    for e in adjacency.edges(i0) {
        let va = &vertex_positions[e.next as usize];
        let vb = &vertex_positions[e.prev as usize];

        let da = (va.x - v0.x) * (va.x - v0.x) + (va.y - v0.y) * (va.y - v0.y) + (va.z - v0.z) * (va.z - v0.z);
        let db = (vb.x - v0.x) * (vb.x - v0.x) + (vb.y - v0.y) * (vb.y - v0.y) + (vb.z - v0.z) * (vb.z - v0.z);

        result = if result < da { da } else { result };
        result = if result < db { db } else { result };
    }

    result.sqrt()
}

fn get_complex_target(v: u32, target: u32, remap: &[u32], loop_: &[u32], loopback: &[u32]) -> u32 {
    let r = remap[target as usize];
    let (lv, lbv) = (loop_[v as usize], loopback[v as usize]);

    // use loop metadata to guide complex collapses towards the correct wedge
    // this works for edges on attribute discontinuities because loop/loopback track the single half-edge without a pair, similar to seams
    if lv != !0u32 && remap[lv as usize] == r {
        lv
    } else if lbv != !0u32 && remap[lbv as usize] == r {
        lbv
    } else {
        target
    }
}

fn bound_edge_collapses(adjacency: &EdgeAdjacency, index_count: usize, vertex_kind: &[u8]) -> usize {
    let mut dual_count = 0;

    for (i, &k) in vertex_kind.iter().enumerate() {
        let e = (adjacency.offsets[i + 1] - adjacency.offsets[i]) as usize;

        dual_count += if k == KIND_MANIFOLD || k == KIND_SEAM { e } else { 0 };
    }

    // pad capacity by 3 so that we can check for overflow once per triangle instead of once per edge
    (index_count - dual_count / 2) + 3
}

fn pick_edge_collapses(
    collapses: &mut [Collapse],
    indices: &[u32],
    remap: &[u32],
    vertex_kind: &[u8],
    loop_: &[u32],
    loopback: &[u32],
) -> usize {
    const NEXT: [usize; 3] = [1, 2, 0];

    let collapse_capacity = collapses.len();
    let mut collapse_count = 0;

    for tri in indices.as_chunks::<3>().0 {
        // this should never happen as boundEdgeCollapses should give an upper bound for the collapse count, but in an unlikely event it does we can just drop extra collapses
        if collapse_count + 3 > collapse_capacity {
            break;
        }

        for e in 0..3 {
            let i0 = tri[e];
            let i1 = tri[NEXT[e]];

            // this can happen either when input has a zero-length edge, or when we perform collapses for complex
            // topology w/seams and collapse a manifold vertex that connects to both wedges onto one of them
            // we leave edges like this alone since they may be important for preserving mesh integrity
            if remap[i0 as usize] == remap[i1 as usize] {
                continue;
            }

            let k0 = vertex_kind[i0 as usize];
            let k1 = vertex_kind[i1 as usize];

            // the edge has to be collapsible in at least one direction
            if !(can_collapse(k0, k1) | can_collapse(k1, k0)) {
                continue;
            }

            // manifold and seam edges should occur twice (i0->i1 and i1->i0) - skip redundant edges
            if has_opposite(k0, k1) && remap[i1 as usize] > remap[i0 as usize] {
                continue;
            }

            // two vertices are on a border or a seam, but there's no direct edge between them
            // this indicates that they belong to two different edge loops and we should not collapse this edge
            // loop[] and loopback[] track half edges so we only need to check one of them
            if (k0 == KIND_BORDER || k0 == KIND_SEAM) && k1 != KIND_MANIFOLD && loop_[i0 as usize] != i1 {
                continue;
            }
            if (k1 == KIND_BORDER || k1 == KIND_SEAM) && k0 != KIND_MANIFOLD && loopback[i1 as usize] != i0 {
                continue;
            }

            // edge can be collapsed in either direction - we will pick the one with minimum error
            // note: we evaluate error later during collapse ranking, here we just tag the edge as bidirectional
            collapses[collapse_count] = if can_collapse(k0, k1) & can_collapse(k1, k0) {
                Collapse { v0: i0, v1: i1, value: 1 }
            } else {
                // edge can only be collapsed in one direction
                let e0 = if can_collapse(k0, k1) { i0 } else { i1 };
                let e1 = if can_collapse(k0, k1) { i1 } else { i0 };

                Collapse { v0: e0, v1: e1, value: 0 }
            };
            collapse_count += 1;
        }
    }

    collapse_count
}

/// Shared attribute data used during collapse evaluation.
struct AttributeData<'a> {
    vertex_attributes: &'a [f32],
    attribute_quadrics: &'a [Quadric],
    attribute_gradients: &'a [QuadricGrad],
    attribute_count: usize,
}

impl AttributeData<'_> {
    fn error(&self, v: u32, vertex_positions: &[Vector3], t: u32, clamped: bool) -> f32 {
        let (v, t, n) = (v as usize, t as usize, self.attribute_count);
        quadric_error_attr(
            &self.attribute_quadrics[v],
            &self.attribute_gradients[v * n..(v + 1) * n],
            &vertex_positions[t],
            &self.vertex_attributes[t * n..(t + 1) * n],
            clamped,
        )
    }
}

/// Topology data shared by collapse ranking/execution.
struct Topology<'a> {
    remap: &'a [u32],
    wedge: &'a [u32],
    vertex_kind: &'a [u8],
    loop_: &'a [u32],
    loopback: &'a [u32],
}

impl Topology<'_> {
    /// Seam pair of a seam collapse i0 -> i1: returns `(s0, s1)`.
    fn seam_pair(&self, i0: u32, i1: u32) -> (u32, u32) {
        let s0 = self.wedge[i0 as usize];
        let s1 = if self.loop_[i0 as usize] == i1 {
            self.loopback[s0 as usize]
        } else {
            self.loop_[s0 as usize]
        };

        // note: this should never happen as s1 must be valid for consistent seams, but if we ever hit this case we'll get a memory safety issue; for now play it safe
        (s0, if s1 != !0u32 { s1 } else { self.wedge[i1 as usize] })
    }

    fn complex_target(&self, v: u32, target: u32) -> u32 {
        get_complex_target(v, target, self.remap, self.loop_, self.loopback)
    }
}

fn rank_edge_collapses(
    collapses: &mut [Collapse],
    vertex_positions: &[Vector3],
    vertex_quadrics: &[Quadric],
    attributes: &AttributeData,
    topo: &Topology,
    clamped: bool,
) {
    let remap = topo.remap;

    for c in collapses.iter_mut() {
        let (i0, i1) = (c.v0, c.v1);
        let bidi = c.value != 0;

        let mut ei = quadric_error(&vertex_quadrics[remap[i0 as usize] as usize], &vertex_positions[i1 as usize]);
        let mut ej = if bidi {
            quadric_error(&vertex_quadrics[remap[i1 as usize] as usize], &vertex_positions[i0 as usize])
        } else {
            f32::MAX
        };

        if attributes.attribute_count != 0 {
            ei += attributes.error(i0, vertex_positions, i1, clamped);
            ej += if bidi {
                attributes.error(i1, vertex_positions, i0, clamped)
            } else {
                0.0
            };

            // seam edges need to aggregate attribute errors between primary and secondary edges, as attribute quadrics are separate
            if topo.vertex_kind[i0 as usize] == KIND_SEAM {
                // for seam collapses we need to find the seam pair; this is a bit tricky since we need to rely on edge loops as target vertex may be locked (and thus have more than two wedges)
                let (s0, s1) = topo.seam_pair(i0, i1);

                ei += attributes.error(s0, vertex_positions, s1, clamped);
                ej += if bidi {
                    attributes.error(s1, vertex_positions, s0, clamped)
                } else {
                    0.0
                };
            } else {
                // complex edges can have multiple wedges, so we need to aggregate errors for all wedges based on the selected target
                let k0 = topo.vertex_kind[i0 as usize];
                if k0 == KIND_COMPLEX || k0 == KIND_FRINGE {
                    let mut v = topo.wedge[i0 as usize];
                    while v != i0 {
                        let t = topo.complex_target(v, i1);

                        ei += attributes.error(v, vertex_positions, t, clamped);
                        v = topo.wedge[v as usize];
                    }
                }

                let k1 = topo.vertex_kind[i1 as usize];
                if (k1 == KIND_COMPLEX || k1 == KIND_FRINGE) && bidi {
                    let mut v = topo.wedge[i1 as usize];
                    while v != i1 {
                        let t = topo.complex_target(v, i0);

                        ej += attributes.error(v, vertex_positions, t, clamped);
                        v = topo.wedge[v as usize];
                    }
                }
            }
        }

        // pick edge direction with minimal error (branchless)
        let rev = bidi & (ej < ei);

        c.v0 = if rev { i1 } else { i0 };
        c.v1 = if rev { i0 } else { i1 };
        c.value = (if ej < ei { ej } else { ei }).to_bits();
    }
}

fn sort_edge_collapses(sort_order: &mut [u32], collapses: &[Collapse]) {
    // we use counting sort to order collapses by error; since the exact sort order is not as critical,
    // only top 12 bits of exponent+mantissa (8 bits of exponent and 4 bits of mantissa) are used.
    // to avoid excessive stack usage, we clamp the exponent range as collapses with errors much higher than 1 are not useful.
    const SORT_BITS: u32 = 12;
    const SORT_BINS: usize = 2048 + 512; // exponent range [-127, 32)

    // skip sign bit since error is non-negative
    let key = |c: &Collapse| (((c.value << 1) >> (32 - SORT_BITS)) as usize).min(SORT_BINS - 1);

    // fill histogram for counting sort
    let mut histogram = [0u32; SORT_BINS];

    for c in collapses {
        histogram[key(c)] += 1;
    }

    // compute offsets based on histogram data
    let mut histogram_sum = 0u32;

    for h in histogram.iter_mut() {
        let count = *h;
        *h = histogram_sum;
        histogram_sum += count;
    }

    // compute sort order based on offsets
    for (i, c) in collapses.iter().enumerate() {
        let k = key(c);
        sort_order[histogram[k] as usize] = i as u32;
        histogram[k] += 1;
    }
}

fn perform_edge_collapses(
    collapse_remap: &mut [u32],
    collapse_locked: &mut [bool],
    collapses: &[Collapse],
    collapse_order: &[u32],
    topo: &Topology,
    vertex_positions: &[Vector3],
    adjacency: &EdgeAdjacency,
    triangle_collapse_goal: usize,
    error_limit: f32,
    result_error: &mut f32,
) -> usize {
    let collapse_count = collapses.len();
    let remap = topo.remap;

    let mut edge_collapses = 0;
    let mut triangle_collapses = 0;

    // most collapses remove 2 triangles; use this to establish a bound on the pass in terms of error limit
    // note that edge_collapse_goal is an estimate; triangle_collapse_goal will be used to actually limit collapses
    let mut edge_collapse_goal = triangle_collapse_goal / 2;

    for &order in collapse_order {
        let c = &collapses[order as usize];

        if c.error() > error_limit {
            break;
        }

        if triangle_collapses >= triangle_collapse_goal {
            break;
        }

        // we limit the error in each pass based on the error of optimal last collapse; since many collapses will be locked
        // as they will share vertices with other successful collapses, we need to increase the acceptable error by some factor
        let error_goal = if edge_collapse_goal < collapse_count {
            1.5 * collapses[collapse_order[edge_collapse_goal] as usize].error()
        } else {
            f32::MAX
        };

        // on average, each collapse is expected to lock 6 other collapses; to avoid degenerate passes on meshes with odd
        // topology, we only abort if we got over 1/6 collapses accordingly.
        if c.error() > error_goal && c.error() > *result_error && triangle_collapses > triangle_collapse_goal / 6 {
            break;
        }

        let (i0, i1) = (c.v0, c.v1);

        let r0 = remap[i0 as usize];
        let r1 = remap[i1 as usize];

        let kind = topo.vertex_kind[i0 as usize];

        // we don't collapse vertices that had source or target vertex involved in a collapse
        // it's important to not move the vertices twice since it complicates the tracking/remapping logic
        // it's important to not move other vertices towards a moved vertex to preserve error since we don't re-rank collapses mid-pass
        if collapse_locked[r0 as usize] | collapse_locked[r1 as usize] {
            continue;
        }

        if has_triangle_flips_collapse(adjacency, vertex_positions, collapse_remap, r0, r1) {
            // adjust collapse goal since this collapse is invalid and shouldn't factor into error goal
            edge_collapse_goal += 1;

            continue;
        }

        if kind == KIND_COMPLEX || kind == KIND_FRINGE {
            collapse_remap[i0 as usize] = i1;

            // remap all vertices in the complex to the target vertex, using the same ranking that we used to evaluate the collapses
            let mut v = topo.wedge[i0 as usize];
            while v != i0 {
                collapse_remap[v as usize] = topo.complex_target(v, i1);
                v = topo.wedge[v as usize];
            }
        } else if kind == KIND_SEAM {
            // for seam collapses we need to move the seam pair together; this is a bit tricky since we need to rely on edge loops as target vertex may be locked (and thus have more than two wedges)
            let (s0, s1) = topo.seam_pair(i0, i1);

            collapse_remap[i0 as usize] = i1;
            collapse_remap[s0 as usize] = s1;
        } else {
            collapse_remap[i0 as usize] = i1;
        }

        // note: we technically don't need to lock r1 if it's a locked vertex, as it can't move and its quadric won't be used
        // however, this results in slightly worse error on some meshes because the locked collapses get an unfair advantage wrt scheduling
        collapse_locked[r0 as usize] = true;
        collapse_locked[r1 as usize] = true;

        // border edges collapse 1 triangle, other edges collapse 2 or more
        // fringe edges often collapse 1 but may also collapse 2 because they don't respect loops, so we conservatively assume 2
        triangle_collapses += if kind == KIND_BORDER { 1 } else { 2 };
        edge_collapses += 1;

        *result_error = if *result_error < c.error() { c.error() } else { *result_error };
    }

    edge_collapses
}

/// Mutable quadric state of the simplifier.
struct QuadricState {
    vertex_quadrics: Vec<Quadric>,
    volume_gradients: Option<Vec<QuadricGrad>>,
    attribute_quadrics: Vec<Quadric>,
    attribute_gradients: Vec<QuadricGrad>,
    attribute_count: usize,
}

fn update_quadrics(collapse_remap: &[u32], state: &mut QuadricState, vertex_positions: &[Vector3], remap: &[u32], vertex_error: &mut f32) {
    let n = state.attribute_count;

    for (i, &target) in collapse_remap.iter().enumerate() {
        if target == i as u32 {
            continue;
        }

        let (i0, i1) = (i, target as usize);

        let r0 = remap[i0] as usize;
        let r1 = remap[i1] as usize;

        // ensure we only update vertex_quadrics once: primary vertex must be moved if any wedge is moved
        if i0 == r0 {
            let q = state.vertex_quadrics[r0];
            quadric_add(&mut state.vertex_quadrics[r1], &q);

            if let Some(gradients) = state.volume_gradients.as_mut() {
                let g = gradients[r0];
                quadric_add_grad(&mut gradients[r1], &g);
            }
        }

        if n != 0 {
            let qa = state.attribute_quadrics[i0];
            quadric_add(&mut state.attribute_quadrics[i1], &qa);

            let mut g = [QuadricGrad::default(); MAX_ATTRIBUTES];
            g[..n].copy_from_slice(&state.attribute_gradients[i0 * n..(i0 + 1) * n]);
            quadric_add_grads(&mut state.attribute_gradients[i1 * n..(i1 + 1) * n], &g[..n]);

            if i0 == r0 {
                // when attributes are used, distance error needs to be recomputed as collapses don't track it; it is safe to do this after the quadric adjustment
                let derr = quadric_error(&state.vertex_quadrics[r0], &vertex_positions[r1]);
                *vertex_error = if *vertex_error < derr { derr } else { *vertex_error };
            }
        }
    }
}

fn solve_positions(
    vertex_positions: &mut [Vector3],
    state: &QuadricState,
    topo: &Topology,
    adjacency: &EdgeAdjacency,
    vertex_update: &[bool],
) {
    let n = state.attribute_count;
    let (remap, wedge, vertex_kind) = (topo.remap, topo.wedge, topo.vertex_kind);

    for i in 0..vertex_positions.len() {
        if !vertex_update[i] {
            continue;
        }

        // moving vertices on an attribute discontinuity may result in extrapolating UV outside of the chart bounds
        // moving vertices on a border requires a stronger edge quadric to preserve the border geometry
        if vertex_kind[i] != KIND_MANIFOLD && vertex_kind[i] != KIND_COMPLEX {
            continue;
        }

        if remap[i] != i as u32 {
            vertex_positions[i] = vertex_positions[remap[i] as usize];
            continue;
        }

        let vp = vertex_positions[i];

        let mut q = state.vertex_quadrics[i];
        let mut gv = QuadricGrad::default();

        // add a point quadric for regularization to stabilize the solution
        let r = quadric_from_point(vp.x, vp.y, vp.z, q.w * 1e-4);
        quadric_add(&mut q, &r);

        if n != 0 {
            // optimal point simultaneously minimizes attribute quadrics for all wedges
            let mut v = i;
            loop {
                quadric_reduce_attributes(&mut q, &state.attribute_quadrics[v], &state.attribute_gradients[v * n..(v + 1) * n]);
                v = wedge[v] as usize;
                if v == i {
                    break;
                }
            }

            // minimizing attribute quadrics results in volume loss so we incorporate volume gradient as a constraint
            if let Some(gradients) = &state.volume_gradients {
                gv = gradients[i];
            }
        }

        let mut p = Vector3::default();
        if !quadric_solve(&mut p, &q, &gv) {
            continue;
        }

        // reject updates that move the vertex too far from its neighborhood
        // this detects and fixes most cases when the quadric is not well-defined
        let nr = get_neighborhood_radius(adjacency, vertex_positions, i as u32);
        let dp = (p.x - vp.x) * (p.x - vp.x) + (p.y - vp.y) * (p.y - vp.y) + (p.z - vp.z) * (p.z - vp.z);

        if dp > nr * nr {
            continue;
        }

        // reject updates that would flip a neighboring triangle, as we do for edge collapse
        if has_triangle_flips_position(adjacency, vertex_positions, i as u32, &p) {
            continue;
        }

        // reject updates that increase positional error too much; allow some tolerance to improve attribute quality
        if quadric_error(&state.vertex_quadrics[i], &p) > quadric_error(&state.vertex_quadrics[i], &vp) * 1.5 + 1e-6 {
            continue;
        }

        vertex_positions[i] = p;
    }
}

fn solve_attributes(
    vertex_positions: &[Vector3],
    vertex_attributes: &mut [f32],
    state: &QuadricState,
    topo: &Topology,
    vertex_update: &[bool],
) {
    let n = state.attribute_count;
    let (remap, wedge, vertex_kind) = (topo.remap, topo.wedge, topo.vertex_kind);

    for i in 0..vertex_positions.len() {
        if !vertex_update[i] {
            continue;
        }

        if remap[i] != i as u32 {
            continue;
        }

        for k in 0..n {
            let mut shared = !0u32;

            // for complex vertices, preserve attribute continuity and use highest weight wedge if values were shared
            if vertex_kind[i] == KIND_COMPLEX || vertex_kind[i] == KIND_FRINGE {
                shared = i as u32;

                let mut v = wedge[i] as usize;
                while v != i {
                    if vertex_attributes[v * n + k] != vertex_attributes[i * n + k] {
                        shared = !0u32;
                    } else if shared != !0u32 && state.attribute_quadrics[v].w > state.attribute_quadrics[shared as usize].w {
                        shared = v as u32;
                    }
                    v = wedge[v] as usize;
                }
            }

            // update attributes for all wedges
            let mut v = i;
            loop {
                let r = if shared == !0u32 { v } else { shared as usize };

                let p = &vertex_positions[i]; // same for all wedges
                let a = &state.attribute_quadrics[r];
                let g = &state.attribute_gradients[r * n + k];

                let iw = if a.w == 0.0 { 0.0 } else { 1.0 / a.w };
                let av = (g.gx * p.x + g.gy * p.y + g.gz * p.z + g.gw) * iw;

                vertex_attributes[v * n + k] = av;
                v = wedge[v] as usize;
                if v == i {
                    break;
                }
            }
        }
    }
}

/// Applies collapse remap to the index buffer, dropping degenerate triangles; returns the new index count.
fn remap_index_buffer_collapse(indices: &mut [u32], collapse_remap: &[u32], remap: &[u32]) -> usize {
    let mut write = 0;

    for i in (0..indices.len()).step_by(3) {
        let v0 = collapse_remap[indices[i] as usize];
        let v1 = collapse_remap[indices[i + 1] as usize];
        let v2 = collapse_remap[indices[i + 2] as usize];

        // collapse zero area triangles even if they are not topologically degenerate
        // this is required to cleanup manifold->seam collapses when a vertex is collapsed onto a seam pair
        // as well as complex collapses and some other cases where cross wedge collapses are performed
        let (r0, r1, r2) = (remap[v0 as usize], remap[v1 as usize], remap[v2 as usize]);

        if r0 != r1 && r0 != r2 && r1 != r2 {
            indices[write..write + 3].copy_from_slice(&[v0, v1, v2]);
            write += 3;
        }
    }

    write
}

fn remap_edge_loops(loop_: &mut [u32], collapse_remap: &[u32]) {
    for i in 0..loop_.len() {
        // note: this is a no-op for vertices that were remapped
        // ideally we would clear the loop entries for those for consistency, even though they aren't going to be used
        // however, the remapping process needs loop information for remapped vertices, so this would require a separate pass
        if loop_[i] != !0u32 {
            let l = loop_[i];
            let r = collapse_remap[l as usize];

            // i == r is a special case when the seam edge is collapsed in a direction opposite to where loop goes
            loop_[i] = if i as u32 == r {
                if loop_[l as usize] != !0u32 {
                    collapse_remap[loop_[l as usize] as usize]
                } else {
                    !0u32
                }
            } else {
                r
            };
        }
    }
}

fn follow(parents: &mut [u32], mut index: u32) -> u32 {
    while index != parents[index as usize] {
        let parent = parents[index as usize];
        parents[index as usize] = parents[parent as usize];
        index = parent;
    }

    index
}

fn build_components(components: &mut [u32], indices: &[u32], remap: &[u32]) -> usize {
    const NEXT: [usize; 4] = [1, 2, 0, 1];

    let vertex_count = components.len();

    for (i, c) in components.iter_mut().enumerate() {
        *c = i as u32;
    }

    // compute a unique (but not sequential!) index for each component via union-find
    for tri in indices.as_chunks::<3>().0 {
        for e in 0..3 {
            let r0 = follow(components, remap[tri[e] as usize]);
            let r1 = follow(components, remap[tri[NEXT[e]] as usize]);

            // merge components with larger indices into components with smaller indices
            // this guarantees that the root of the component is always the one with the smallest index
            if r0 != r1 {
                components[r0.max(r1) as usize] = r0.min(r1);
            }
        }
    }

    // make sure each element points to the component root *before* we renumber the components
    for i in 0..vertex_count {
        if remap[i] == i as u32 {
            components[i] = follow(components, i as u32);
        }
    }

    let mut next_component = 0u32;

    // renumber components using sequential indices
    // a sequential pass is sufficient because component root always has the smallest index
    // note: it is unsafe to use follow() in this pass because we're replacing component links with sequential indices inplace
    for i in 0..vertex_count {
        if remap[i] == i as u32 {
            let root = components[i];
            components[i] = if root == i as u32 {
                next_component += 1;
                next_component - 1
            } else {
                components[root as usize]
            };
        } else {
            components[i] = components[remap[i] as usize];
        }
    }

    next_component as usize
}

fn measure_components(component_errors: &mut [f32], component_count: usize, components: &[u32], vertex_positions: &[Vector3]) {
    component_errors[..component_count * 4].fill(0.0);

    // compute approximate sphere center for each component as an average
    for (&c, v) in components.iter().zip(vertex_positions) {
        let c = c as usize;

        component_errors[c * 4] += v.x;
        component_errors[c * 4 + 1] += v.y;
        component_errors[c * 4 + 2] += v.z;
        component_errors[c * 4 + 3] += 1.0; // weight
    }

    // complete the center computation, and reinitialize [3] as a radius
    for ce in component_errors[..component_count * 4].as_chunks_mut::<4>().0 {
        let w = ce[3];
        let iw = if w == 0.0 { 0.0 } else { 1.0 / w };

        ce[0] *= iw;
        ce[1] *= iw;
        ce[2] *= iw;
        ce[3] = 0.0; // radius
    }

    // compute squared radius for each component
    for (&c, v) in components.iter().zip(vertex_positions) {
        let c = c as usize;

        let dx = v.x - component_errors[c * 4];
        let dy = v.y - component_errors[c * 4 + 1];
        let dz = v.z - component_errors[c * 4 + 2];
        let r = dx * dx + dy * dy + dz * dz;

        component_errors[c * 4 + 3] = if component_errors[c * 4 + 3] < r {
            r
        } else {
            component_errors[c * 4 + 3]
        };
    }

    // we've used the output buffer as scratch space, so we need to move the results to proper indices
    // note: we keep the squared error to make it match quadric error metric
    for i in 0..component_count {
        component_errors[i] = component_errors[i * 4 + 3];
    }
}

fn prune_components(indices: &mut [u32], components: &[u32], component_errors: &[f32], error_cutoff: f32, nexterror: &mut f32) -> usize {
    let mut write = 0;
    let mut min_error = f32::MAX;

    for i in (0..indices.len()).step_by(3) {
        let (v0, v1, v2) = (indices[i], indices[i + 1], indices[i + 2]);
        let c = components[v0 as usize] as usize;

        if component_errors[c] > error_cutoff {
            min_error = if min_error > component_errors[c] {
                component_errors[c]
            } else {
                min_error
            };

            indices[write..write + 3].copy_from_slice(&[v0, v1, v2]);
            write += 3;
        }
    }

    // update next error with the smallest error of the remaining components
    *nexterror = min_error;
    write
}

struct CellHasher<'a> {
    vertex_ids: Option<&'a [u32]>,
}

impl Hasher<u32> for CellHasher<'_> {
    fn hash(&self, i: &u32) -> usize {
        let mut h = self.vertex_ids.map_or(*i, |ids| ids[*i as usize]);

        // MurmurHash2 finalizer
        h ^= h >> 13;
        h = h.wrapping_mul(0x5bd1e995);
        h ^= h >> 15;
        h as usize
    }

    fn equal(&mut self, lhs: &u32, rhs: &u32) -> bool {
        match self.vertex_ids {
            Some(ids) => ids[*lhs as usize] == ids[*rhs as usize],
            None => lhs == rhs,
        }
    }
}

struct TriangleHasher<'a> {
    indices: &'a [u32],
}

impl Hasher<u32> for TriangleHasher<'_> {
    fn hash(&self, i: &u32) -> usize {
        let tri = &self.indices[*i as usize * 3..];

        // Optimized Spatial Hashing for Collision Detection of Deformable Objects
        (tri[0].wrapping_mul(73856093) ^ tri[1].wrapping_mul(19349663) ^ tri[2].wrapping_mul(83492791)) as usize
    }

    fn equal(&mut self, lhs: &u32, rhs: &u32) -> bool {
        let (l, r) = (*lhs as usize * 3, *rhs as usize * 3);
        self.indices[l..l + 3] == self.indices[r..r + 3]
    }
}

fn compute_vertex_ids(vertex_ids: &mut [u32], vertex_positions: &[Vector3], vertex_lock: Option<&[u8]>, grid_size: i32) {
    let cell_scale = (grid_size - 1) as f32;

    for (i, (id, v)) in vertex_ids.iter_mut().zip(vertex_positions).enumerate() {
        let xi = (v.x * cell_scale + 0.5) as i32;
        let yi = (v.y * cell_scale + 0.5) as i32;
        let zi = (v.z * cell_scale + 0.5) as i32;

        *id = if vertex_lock.is_some_and(|l| l[i] & SIMPLIFY_VERTEX_LOCK != 0) {
            (1 << 30) | i as u32
        } else {
            ((xi << 20) | (yi << 10) | zi) as u32
        };
    }
}

fn count_triangles(vertex_ids: &[u32], indices: &[u32]) -> usize {
    indices
        .as_chunks::<3>()
        .0
        .iter()
        .filter(|tri| {
            let (id0, id1, id2) = (
                vertex_ids[tri[0] as usize],
                vertex_ids[tri[1] as usize],
                vertex_ids[tri[2] as usize],
            );
            id0 != id1 && id0 != id2 && id1 != id2
        })
        .count()
}

fn fill_vertex_cells(table: &mut [u32], vertex_cells: &mut [u32], vertex_ids: &[u32]) -> usize {
    let mut hasher = CellHasher {
        vertex_ids: Some(vertex_ids),
    };

    table.fill(!0u32);

    let mut result = 0;

    for i in 0..vertex_ids.len() {
        let entry = hash_lookup(table, &mut hasher, &(i as u32), &!0u32);

        if table[entry] == !0u32 {
            table[entry] = i as u32;
            vertex_cells[i] = result as u32;
            result += 1;
        } else {
            vertex_cells[i] = vertex_cells[table[entry] as usize];
        }
    }

    result
}

fn count_vertex_cells(table: &mut [u32], vertex_ids: &[u32]) -> usize {
    let mut hasher = CellHasher { vertex_ids: None };

    table.fill(!0u32);

    let mut result = 0;

    for &id in vertex_ids {
        let entry = hash_lookup(table, &mut hasher, &id, &!0u32);

        result += usize::from(table[entry] == !0u32);
        table[entry] = id;
    }

    result
}

fn fill_cell_quadrics(cell_quadrics: &mut [Quadric], indices: &[u32], vertex_positions: &[Vector3], vertex_cells: &[u32]) {
    for tri in indices.as_chunks::<3>().0 {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);

        let (c0, c1, c2) = (vertex_cells[i0] as usize, vertex_cells[i1] as usize, vertex_cells[i2] as usize);

        let single_cell = c0 == c1 && c0 == c2;

        let q = quadric_from_triangle(
            &vertex_positions[i0],
            &vertex_positions[i1],
            &vertex_positions[i2],
            if single_cell { 3.0 } else { 1.0 },
        );

        if single_cell {
            quadric_add(&mut cell_quadrics[c0], &q);
        } else {
            quadric_add(&mut cell_quadrics[c0], &q);
            quadric_add(&mut cell_quadrics[c1], &q);
            quadric_add(&mut cell_quadrics[c2], &q);
        }
    }
}

/// Optional float3 color stream for point simplification.
#[derive(Clone, Copy)]
struct Colors<'a> {
    data: Option<&'a [f32]>,
    stride_float: usize,
}

impl Colors<'_> {
    fn get(&self, i: usize) -> [f32; 3] {
        match self.data {
            Some(c) => [c[i * self.stride_float], c[i * self.stride_float + 1], c[i * self.stride_float + 2]],
            None => [0.0; 3],
        }
    }
}

fn fill_cell_reservoirs(cell_reservoirs: &mut [Reservoir], vertex_positions: &[Vector3], colors: Colors, vertex_cells: &[u32]) {
    for (i, v) in vertex_positions.iter().enumerate() {
        let r = &mut cell_reservoirs[vertex_cells[i] as usize];

        let color = colors.get(i);

        r.x += v.x;
        r.y += v.y;
        r.z += v.z;
        r.r += color[0];
        r.g += color[1];
        r.b += color[2];
        r.w += 1.0;
    }

    for r in cell_reservoirs.iter_mut() {
        let iw = if r.w == 0.0 { 0.0 } else { 1.0 / r.w };

        r.x *= iw;
        r.y *= iw;
        r.z *= iw;
        r.r *= iw;
        r.g *= iw;
        r.b *= iw;
    }
}

fn fill_cell_remap_quadrics(
    cell_remap: &mut [u32],
    cell_errors: &mut [f32],
    vertex_cells: &[u32],
    cell_quadrics: &[Quadric],
    vertex_positions: &[Vector3],
) {
    cell_remap.fill(!0u32);

    for (i, v) in vertex_positions.iter().enumerate() {
        let cell = vertex_cells[i] as usize;
        let error = quadric_error(&cell_quadrics[cell], v);

        if cell_remap[cell] == !0u32 || cell_errors[cell] > error {
            cell_remap[cell] = i as u32;
            cell_errors[cell] = error;
        }
    }
}

fn fill_cell_remap_reservoirs(
    cell_remap: &mut [u32],
    cell_errors: &mut [f32],
    vertex_cells: &[u32],
    cell_reservoirs: &[Reservoir],
    vertex_positions: &[Vector3],
    colors: Colors,
    color_weight: f32,
) {
    cell_remap.fill(!0u32);

    for (i, v) in vertex_positions.iter().enumerate() {
        let cell = vertex_cells[i] as usize;
        let r = &cell_reservoirs[cell];

        let color = colors.get(i);

        let pos_error = (v.x - r.x) * (v.x - r.x) + (v.y - r.y) * (v.y - r.y) + (v.z - r.z) * (v.z - r.z);
        let col_error = (color[0] - r.r) * (color[0] - r.r) + (color[1] - r.g) * (color[1] - r.g) + (color[2] - r.b) * (color[2] - r.b);
        let error = pos_error + color_weight * col_error;

        if cell_remap[cell] == !0u32 || cell_errors[cell] > error {
            cell_remap[cell] = i as u32;
            cell_errors[cell] = error;
        }
    }
}

fn filter_triangles(destination: &mut [u32], tritable: &mut [u32], indices: &[u32], vertex_cells: &[u32], cell_remap: &[u32]) -> usize {
    tritable.fill(!0u32);

    let mut result = 0;

    for tri in indices.as_chunks::<3>().0 {
        let (c0, c1, c2) = (
            vertex_cells[tri[0] as usize],
            vertex_cells[tri[1] as usize],
            vertex_cells[tri[2] as usize],
        );

        if c0 != c1 && c0 != c2 && c1 != c2 {
            let (mut a, mut b, mut c) = (cell_remap[c0 as usize], cell_remap[c1 as usize], cell_remap[c2 as usize]);

            if b < a && b < c {
                (a, b, c) = (b, c, a);
            } else if c < a && c < b {
                (a, b, c) = (c, a, b);
            }

            destination[result * 3..result * 3 + 3].copy_from_slice(&[a, b, c]);

            let entry = hash_lookup(tritable, &mut TriangleHasher { indices: destination }, &(result as u32), &!0u32);

            if tritable[entry] == !0u32 {
                tritable[entry] = result as u32;
                result += 1;
            }
        }
    }

    result * 3
}

fn interpolate(y: f32, x0: f32, y0: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    // three point interpolation from "revenge of interpolation search" paper
    let num = (y1 - y) * (x1 - x2) * (x1 - x0) * (y2 - y0);
    let den = (y2 - y) * (x1 - x2) * (y0 - y1) + (y0 - y) * (x1 - x0) * (y1 - y2);
    x1 + if den == 0.0 { 0.0 } else { num / den }
}

/// Development-only option: fill debug visualization data (vertex kind/loop bits) instead of plain index data. Do *not* use.
#[doc(hidden)]
pub const SIMPLIFY_INTERNAL_DEBUG: u32 = 1 << 30;
const SIMPLIFY_INTERNAL_SOLVE: u32 = 1 << 29;

/// Internal data needed to write back solved positions/attributes (C: `finalizeVertices` arguments).
struct SolveOutput {
    vertex_positions: Vec<Vector3>,
    vertex_attributes: Vec<f32>,
    sparse_remap: Option<Vec<u32>>,
    attribute_remap: [u32; MAX_ATTRIBUTES],
    attribute_count: usize,
    vertex_scale: f32,
    vertex_offset: [f32; 3],
    vertex_kind: Vec<u8>,
    vertex_update: Vec<bool>,
}

struct EdgeInput<'a> {
    positions: &'a [f32],
    positions_stride: usize,
    attributes: &'a [f32],
    attributes_stride: usize,
    weights: &'a [f32],
    attribute_count: usize,
    vertex_lock: Option<&'a [u8]>,
}

fn validate_edge_input(
    input: &EdgeInput,
    indices: &[u32],
    index_count: usize,
    vertex_count: usize,
    target_index_count: usize,
    target_error: f32,
) -> Result<()> {
    validate::triangles(index_count)?;
    validate::floats(input.positions, vertex_count, input.positions_stride, 12)?;
    ensure(
        target_index_count <= index_count,
        Error::InvalidArgument("target_index_count must be <= index_count"),
    )?;
    ensure(target_error >= 0.0, Error::InvalidArgument("target_error must be >= 0"))?;
    ensure(
        input.attributes_stride >= input.attribute_count * 4 && input.attributes_stride <= 256 && input.attributes_stride.is_multiple_of(4),
        Error::InvalidArgument("vertex_attributes_stride out of range"),
    )?;
    ensure(
        input.attribute_count <= MAX_ATTRIBUTES,
        Error::InvalidArgument("attribute_count must be <= 32"),
    )?;
    if input.attribute_count != 0 {
        let weights = input
            .weights
            .get(..input.attribute_count)
            .ok_or(Error::BufferTooSmall("attribute_weights"))?;
        ensure(
            weights.iter().all(|&w| w >= 0.0),
            Error::InvalidArgument("attribute_weights must be >= 0"),
        )?;
        let required = if vertex_count == 0 {
            0
        } else {
            ((vertex_count - 1) * input.attributes_stride) / 4 + input.attribute_count
        };
        ensure(input.attributes.len() >= required, Error::BufferTooSmall("vertex_attributes"))?;
    }
    if let Some(lock) = input.vertex_lock {
        ensure(lock.len() >= vertex_count, Error::BufferTooSmall("vertex_lock"))?;
    }
    validate::indices(indices, index_count, vertex_count)?;
    Ok(())
}

/// Core edge collapse simplifier (C: `meshopt_simplifyEdge`); `result` holds the input indices and receives the output.
fn simplify_edge(
    result: &mut [u32],
    input: &EdgeInput,
    vertex_count: usize,
    target_index_count: usize,
    target_error: f32,
    options: u32,
) -> (usize, f32, Option<SolveOutput>) {
    let index_count = result.len();
    let mut vertex_count = vertex_count;
    let vertex_lock = input.vertex_lock;

    // build an index remap and update indices/vertex_count to minimize the subsequent work
    // note: as a consequence, errors will be computed relative to the subset extent
    let sparse_remap = if options & SIMPLIFY_SPARSE != 0 {
        let sr = build_sparse_remap(result, vertex_count);
        vertex_count = sr.len();
        Some(sr)
    } else {
        None
    };
    let sparse = sparse_remap.as_deref();

    let positions = Positions {
        data: input.positions,
        stride_float: input.positions_stride / 4,
        sparse_remap: sparse,
    };

    // build adjacency information
    let mut adjacency = EdgeAdjacency::new(index_count, vertex_count);
    adjacency.update(result, vertex_count, None);

    // build position remap that maps each vertex to the one with identical position
    // wedge table stores next vertex with identical position for each vertex
    let mut remap = vec![0u32; vertex_count];
    let mut wedge = vec![0u32; vertex_count];
    build_position_remap(&mut remap, Some(&mut wedge), positions);

    // classify vertices; vertex kind determines collapse rules, see kCanCollapse
    let mut vertex_kind = vec![0u8; vertex_count];
    let mut loop_ = vec![0u32; vertex_count];
    let mut loopback = vec![0u32; vertex_count];
    classify_vertices(
        &mut vertex_kind,
        &mut loop_,
        &mut loopback,
        &adjacency,
        &remap,
        &wedge,
        vertex_lock,
        sparse,
        options,
    );

    let mut vertex_positions = vec![Vector3::default(); vertex_count];
    let (vertex_scale, vertex_offset) = rescale_positions(Some(&mut vertex_positions), positions, vertex_count);

    let mut attribute_remap = [0u32; MAX_ATTRIBUTES];
    let mut attribute_count = input.attribute_count;
    let mut vertex_attributes = Vec::new();

    if attribute_count != 0 {
        // remap attributes to only include ones with weight > 0 to minimize memory/compute overhead for quadrics
        let mut attributes_used = 0;
        for i in 0..attribute_count {
            if input.weights[i] > 0.0 {
                attribute_remap[attributes_used] = i as u32;
                attributes_used += 1;
            }
        }

        attribute_count = attributes_used;
        vertex_attributes = vec![0.0f32; vertex_count * attribute_count];
        let attributes = Attributes {
            data: input.attributes,
            stride_float: input.attributes_stride / 4,
            weights: input.weights,
        };
        rescale_attributes(
            &mut vertex_attributes,
            attributes,
            vertex_count,
            attribute_count,
            &attribute_remap,
            sparse,
        );
    }

    let mut state = QuadricState {
        vertex_quadrics: vec![Quadric::default(); vertex_count],
        volume_gradients: None,
        attribute_quadrics: Vec::new(),
        attribute_gradients: Vec::new(),
        attribute_count,
    };

    if attribute_count != 0 {
        state.attribute_quadrics = vec![Quadric::default(); vertex_count];
        state.attribute_gradients = vec![QuadricGrad::default(); vertex_count * attribute_count];

        if options & SIMPLIFY_INTERNAL_SOLVE != 0 {
            state.volume_gradients = Some(vec![QuadricGrad::default(); vertex_count]);
        }
    }

    fill_face_quadrics(
        &mut state.vertex_quadrics,
        state.volume_gradients.as_deref_mut(),
        result,
        &vertex_positions,
        &remap,
    );
    fill_vertex_quadrics(&mut state.vertex_quadrics, &vertex_positions, &remap, vertex_lock, sparse, options);
    fill_edge_quadrics(
        &mut state.vertex_quadrics,
        result,
        &vertex_positions,
        &remap,
        &vertex_kind,
        &loop_,
        &loopback,
    );

    if options & SIMPLIFY_PRESERVE_FOLDS != 0 {
        fill_fold_quadrics(&mut state.vertex_quadrics, &adjacency, &vertex_positions, &remap, &wedge);
    }

    if attribute_count != 0 {
        fill_attribute_quadrics(
            &mut state.attribute_quadrics,
            &mut state.attribute_gradients,
            result,
            &vertex_positions,
            &vertex_attributes,
            attribute_count,
        );
    }

    let mut components = Vec::new();
    let mut component_errors = Vec::new();
    let mut component_count = 0;
    let mut component_nexterror = 0.0f32;

    if options & SIMPLIFY_PRUNE != 0 {
        components = vec![0u32; vertex_count];
        component_count = build_components(&mut components, result, &remap);

        component_errors = vec![0.0f32; component_count * 4]; // overallocate for temporary use inside measureComponents
        measure_components(&mut component_errors, component_count, &components, &vertex_positions);

        component_nexterror = f32::MAX;
        for &e in &component_errors[..component_count] {
            component_nexterror = if component_nexterror > e { e } else { component_nexterror };
        }
    }

    let collapse_capacity = bound_edge_collapses(&adjacency, index_count, &vertex_kind);

    let mut edge_collapses = vec![Collapse::default(); collapse_capacity];
    let mut collapse_order = vec![0u32; collapse_capacity];
    let mut collapse_remap = vec![0u32; vertex_count];
    let mut collapse_locked = vec![false; vertex_count];

    let mut result_count = index_count;
    let mut result_error = 0.0f32;
    let mut vertex_error = 0.0f32;

    // target_error input is linear; we need to adjust it to match quadricError units
    let error_scale = if options & SIMPLIFY_ERROR_ABSOLUTE != 0 {
        vertex_scale
    } else {
        1.0
    };
    let error_limit = (target_error * target_error) / (error_scale * error_scale);

    while result_count > target_index_count {
        // note: throughout the simplification process adjacency structure reflects welded topology for result-in-progress
        adjacency.update(&result[..result_count], vertex_count, Some(&remap));

        let edge_collapse_count = pick_edge_collapses(
            &mut edge_collapses,
            &result[..result_count],
            &remap,
            &vertex_kind,
            &loop_,
            &loopback,
        );

        // no edges can be collapsed any more due to topology restrictions
        if edge_collapse_count == 0 {
            break;
        }

        let topo = Topology {
            remap: &remap,
            wedge: &wedge,
            vertex_kind: &vertex_kind,
            loop_: &loop_,
            loopback: &loopback,
        };
        let attributes = AttributeData {
            vertex_attributes: &vertex_attributes,
            attribute_quadrics: &state.attribute_quadrics,
            attribute_gradients: &state.attribute_gradients,
            attribute_count,
        };

        rank_edge_collapses(
            &mut edge_collapses[..edge_collapse_count],
            &vertex_positions,
            &state.vertex_quadrics,
            &attributes,
            &topo,
            options & SIMPLIFY_ERROR_CLAMPED != 0,
        );

        sort_edge_collapses(&mut collapse_order[..edge_collapse_count], &edge_collapses[..edge_collapse_count]);

        let triangle_collapse_goal = (result_count - target_index_count) / 3;

        for (i, r) in collapse_remap.iter_mut().enumerate() {
            *r = i as u32;
        }

        collapse_locked.fill(false);

        let collapses = perform_edge_collapses(
            &mut collapse_remap,
            &mut collapse_locked,
            &edge_collapses[..edge_collapse_count],
            &collapse_order[..edge_collapse_count],
            &topo,
            &vertex_positions,
            &adjacency,
            triangle_collapse_goal,
            error_limit,
            &mut result_error,
        );

        // no edges can be collapsed any more due to hitting the error limit or triangle collapse limit
        if collapses == 0 {
            break;
        }

        update_quadrics(&collapse_remap, &mut state, &vertex_positions, &remap, &mut vertex_error);

        // updateQuadrics will update vertex error if we use attributes, but if we don't then result_error and vertex_error are equivalent
        vertex_error = if attribute_count == 0 { result_error } else { vertex_error };

        // note: we update loops following edge collapses, but after this we might still have stale loop data
        // this can happen when a triangle with a loop edge gets collapsed along a non-loop edge
        // that works since a loop that points to a vertex that is no longer connected is not affecting collapse logic
        remap_edge_loops(&mut loop_, &collapse_remap);
        remap_edge_loops(&mut loopback, &collapse_remap);

        result_count = remap_index_buffer_collapse(&mut result[..result_count], &collapse_remap, &remap);

        if options & SIMPLIFY_PRUNE != 0 && result_count > target_index_count && component_nexterror <= vertex_error {
            result_count = prune_components(
                &mut result[..result_count],
                &components,
                &component_errors,
                vertex_error,
                &mut component_nexterror,
            );
        }
    }

    // at this point, component_nexterror might be stale: component it references may have been removed through a series of edge collapses
    let mut component_nextstale = true;

    // we're done with the regular simplification but we're still short of the target; try pruning more aggressively towards error_limit
    while options & SIMPLIFY_PRUNE != 0 && result_count > target_index_count && component_nexterror <= error_limit {
        let component_cutoff = if component_nexterror * 1.5 < error_limit {
            component_nexterror * 1.5
        } else {
            error_limit
        };

        // track maximum error in eligible components as we are increasing resulting error
        let mut component_maxerror = 0.0f32;
        for &e in &component_errors[..component_count] {
            if e > component_maxerror && e <= component_cutoff {
                component_maxerror = e;
            }
        }

        let new_count = prune_components(
            &mut result[..result_count],
            &components,
            &component_errors,
            component_cutoff,
            &mut component_nexterror,
        );
        if new_count == result_count && !component_nextstale {
            break;
        }

        component_nextstale = false; // pruneComponents guarantees next error is up to date
        result_count = new_count;
        result_error = if result_error < component_maxerror {
            component_maxerror
        } else {
            result_error
        };
        vertex_error = if vertex_error < component_maxerror {
            component_maxerror
        } else {
            vertex_error
        };
    }

    let mut solve_output = None;

    // if solve is requested, compute updated vertex data; it is written to input buffers by the caller (C: finalizeVertices)
    if options & SIMPLIFY_INTERNAL_SOLVE != 0 {
        let mut vertex_update = collapse_locked; // reuse as scratch space
        vertex_update.fill(false);

        // limit quadric solve to vertices that are still used in the result
        for &v in &result[..result_count] {
            // mark the vertex for finalizeVertices and root vertex for solve*
            vertex_update[remap[v as usize] as usize] = true;
            vertex_update[v as usize] = true;
        }

        // edge adjacency may be stale as we haven't updated it after last series of edge collapses
        adjacency.update(&result[..result_count], vertex_count, Some(&remap));

        let topo = Topology {
            remap: &remap,
            wedge: &wedge,
            vertex_kind: &vertex_kind,
            loop_: &loop_,
            loopback: &loopback,
        };

        solve_positions(&mut vertex_positions, &state, &topo, &adjacency, &vertex_update);

        if attribute_count != 0 {
            solve_attributes(&vertex_positions, &mut vertex_attributes, &state, &topo, &vertex_update);
        }

        solve_output = Some(SolveOutput {
            vertex_positions,
            vertex_attributes,
            sparse_remap: sparse_remap.clone(),
            attribute_remap,
            attribute_count,
            vertex_scale,
            vertex_offset,
            vertex_kind: vertex_kind.clone(),
            vertex_update,
        });
    }

    // if debug visualization data is requested, fill it instead of index data; for simplicity, this doesn't work with sparsity
    if options & SIMPLIFY_INTERNAL_DEBUG != 0 && sparse_remap.is_none() {
        for tri in result[..result_count].as_chunks_mut::<3>().0 {
            let [a, b, c] = *tri;
            let bits = |x: u32, y: u32| {
                (u32::from(vertex_kind[x as usize]) << 28) | (u32::from(loop_[x as usize] == y || loopback[y as usize] == x) << 31)
            };

            tri[0] |= bits(a, b);
            tri[1] |= bits(b, c);
            tri[2] |= bits(c, a);
        }
    }

    // convert resulting indices back into the dense space of the larger mesh
    if let Some(sr) = &sparse_remap {
        for r in result[..result_count].iter_mut() {
            *r = sr[*r as usize];
        }
    }

    // result_error is quadratic; we need to remap it back to linear
    (result_count, result_error.sqrt() * error_scale, solve_output)
}

/// Where solved attributes are written back: a separate buffer, or the position buffer itself at a float offset (C pointer aliasing).
enum AttributeTarget<'a> {
    Separate(&'a mut [f32]),
    Interleaved(usize),
}

fn finalize_vertices(
    vertex_positions_data: &mut [f32],
    vertex_positions_stride: usize,
    mut attributes: AttributeTarget,
    vertex_attributes_stride: usize,
    attribute_weights: &[f32],
    out: &SolveOutput,
    vertex_lock: Option<&[u8]>,
) {
    let positions_stride_float = vertex_positions_stride / 4;
    let attributes_stride_float = vertex_attributes_stride / 4;
    let n = out.attribute_count;

    for i in 0..out.vertex_positions.len() {
        if !out.vertex_update[i] {
            continue;
        }

        let ri = out.sparse_remap.as_ref().map_or(i, |r| r[i] as usize);

        // updating externally locked vertices is not allowed
        if vertex_lock.is_some_and(|l| l[ri] & SIMPLIFY_VERTEX_LOCK != 0) {
            continue;
        }

        // moving locked vertices may result in floating point drift
        if out.vertex_kind[i] != KIND_LOCKED {
            let p = &out.vertex_positions[i];
            let v = &mut vertex_positions_data[ri * positions_stride_float..ri * positions_stride_float + 3];

            v[0] = p.x * out.vertex_scale + out.vertex_offset[0];
            v[1] = p.y * out.vertex_scale + out.vertex_offset[1];
            v[2] = p.z * out.vertex_scale + out.vertex_offset[2];
        }

        for k in 0..n {
            let rk = out.attribute_remap[k] as usize;
            let value = out.vertex_attributes[i * n + k] / attribute_weights[rk];

            match &mut attributes {
                AttributeTarget::Separate(data) => data[ri * attributes_stride_float + rk] = value,
                AttributeTarget::Interleaved(offset) => vertex_positions_data[*offset + ri * attributes_stride_float + rk] = value,
            }
        }
    }
}

/// Mesh simplifier.
/// Reduces the number of triangles in the mesh, attempting to preserve mesh appearance as much as possible.
/// Returns the number of indices after simplification, with `destination` containing new index data
/// (`destination` must hold `index_count` elements, *not* `target_index_count`).
///
/// `target_error` is the error relative to mesh extents that can be tolerated, e.g. 0.01 = 1% deformation (unless [`SIMPLIFY_ERROR_ABSOLUTE`] is used).
/// `options` is a bitmask of `SIMPLIFY_*` options; 0 is a safe default. `result_error` receives the resulting error when provided.
pub fn simplify<T: IndexType>(
    destination: &mut [T],
    indices: &[T],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    target_index_count: usize,
    target_error: f32,
    options: u32,
    result_error: Option<&mut f32>,
) -> Result<usize> {
    let indices = index::input(indices);
    index::output(destination, |destination| {
        simplify_u32(
            destination,
            &indices,
            index_count,
            vertex_positions,
            vertex_count,
            vertex_positions_stride,
            target_index_count,
            target_error,
            options,
            result_error,
        )
    })
}

fn simplify_u32(
    destination: &mut [u32],
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    target_index_count: usize,
    target_error: f32,
    options: u32,
    result_error: Option<&mut f32>,
) -> Result<usize> {
    simplify_with_attributes(
        destination,
        indices,
        index_count,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        &[],
        0,
        &[],
        0,
        None,
        target_index_count,
        target_error,
        options,
        result_error,
    )
}

/// Mesh simplifier with attribute metric.
/// Similar to [`simplify`], but incorporates attribute values into the error metric used to prioritize simplification order.
///
/// `vertex_attributes` holds `attribute_count` floats per vertex (spaced `vertex_attributes_stride` bytes apart), `attribute_weights` holds
/// `attribute_count` weights; `attribute_count` must be <= 32. `vertex_lock` can be `None`; otherwise it holds `SIMPLIFY_VERTEX_*` flags per vertex.
pub fn simplify_with_attributes<T: IndexType>(
    destination: &mut [T],
    indices: &[T],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    vertex_attributes: &[f32],
    vertex_attributes_stride: usize,
    attribute_weights: &[f32],
    attribute_count: usize,
    vertex_lock: Option<&[u8]>,
    target_index_count: usize,
    target_error: f32,
    options: u32,
    result_error: Option<&mut f32>,
) -> Result<usize> {
    let indices = index::input(indices);
    index::output(destination, |destination| {
        simplify_with_attributes_u32(
            destination,
            &indices,
            index_count,
            vertex_positions,
            vertex_count,
            vertex_positions_stride,
            vertex_attributes,
            vertex_attributes_stride,
            attribute_weights,
            attribute_count,
            vertex_lock,
            target_index_count,
            target_error,
            options,
            result_error,
        )
    })
}

fn simplify_with_attributes_u32(
    destination: &mut [u32],
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    vertex_attributes: &[f32],
    vertex_attributes_stride: usize,
    attribute_weights: &[f32],
    attribute_count: usize,
    vertex_lock: Option<&[u8]>,
    target_index_count: usize,
    target_error: f32,
    options: u32,
    result_error: Option<&mut f32>,
) -> Result<usize> {
    ensure(
        options & SIMPLIFY_INTERNAL_SOLVE == 0,
        Error::InvalidArgument("use simplify_with_update for solving"),
    )?;

    let input = EdgeInput {
        positions: vertex_positions,
        positions_stride: vertex_positions_stride,
        attributes: vertex_attributes,
        attributes_stride: vertex_attributes_stride,
        weights: attribute_weights,
        attribute_count,
        vertex_lock,
    };
    validate_edge_input(&input, indices, index_count, vertex_count, target_index_count, target_error)?;
    let result = validate::output(destination, index_count)?;

    result.copy_from_slice(&indices[..index_count]);

    let (count, error, _) = simplify_edge(result, &input, vertex_count, target_index_count, target_error, options);

    if let Some(e) = result_error {
        *e = error;
    }

    Ok(count)
}

/// Mesh simplifier with position/attribute update.
/// Similar to [`simplify_with_attributes`], but destructively updates positions and attribute values for optimal appearance.
/// Returns the number of indices after simplification; `indices` are destructively updated with new index data.
/// Attributes with zero weights are not updated.
pub fn simplify_with_update<T: IndexType>(
    indices: &mut [T],
    index_count: usize,
    vertex_positions: &mut [f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    vertex_attributes: &mut [f32],
    vertex_attributes_stride: usize,
    attribute_weights: &[f32],
    attribute_count: usize,
    vertex_lock: Option<&[u8]>,
    target_index_count: usize,
    target_error: f32,
    options: u32,
    result_error: Option<&mut f32>,
) -> Result<usize> {
    index::output(indices, |indices| {
        simplify_with_update_u32(
            indices,
            index_count,
            vertex_positions,
            vertex_count,
            vertex_positions_stride,
            vertex_attributes,
            vertex_attributes_stride,
            attribute_weights,
            attribute_count,
            vertex_lock,
            target_index_count,
            target_error,
            options,
            result_error,
        )
    })
}

fn simplify_with_update_u32(
    indices: &mut [u32],
    index_count: usize,
    vertex_positions: &mut [f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    vertex_attributes: &mut [f32],
    vertex_attributes_stride: usize,
    attribute_weights: &[f32],
    attribute_count: usize,
    vertex_lock: Option<&[u8]>,
    target_index_count: usize,
    target_error: f32,
    options: u32,
    result_error: Option<&mut f32>,
) -> Result<usize> {
    let (count, error, solve) = {
        let input = EdgeInput {
            positions: vertex_positions,
            positions_stride: vertex_positions_stride,
            attributes: vertex_attributes,
            attributes_stride: vertex_attributes_stride,
            weights: attribute_weights,
            attribute_count,
            vertex_lock,
        };
        validate_edge_input(&input, indices, index_count, vertex_count, target_index_count, target_error)?;

        simplify_edge(
            &mut indices[..index_count],
            &input,
            vertex_count,
            target_index_count,
            target_error,
            options | SIMPLIFY_INTERNAL_SOLVE,
        )
    };

    if let Some(out) = &solve {
        finalize_vertices(
            vertex_positions,
            vertex_positions_stride,
            AttributeTarget::Separate(vertex_attributes),
            vertex_attributes_stride,
            attribute_weights,
            out,
            vertex_lock,
        );
    }

    if let Some(e) = result_error {
        *e = error;
    }

    Ok(count)
}

/// Variant of [`simplify_with_update`] for interleaved vertex data, where attributes live in the same buffer as positions
/// starting `vertex_attributes_offset` floats into `vertex_data` (C: `vertex_attributes` aliasing `vertex_positions`).
pub fn simplify_with_update_interleaved<T: IndexType>(
    indices: &mut [T],
    index_count: usize,
    vertex_data: &mut [f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    vertex_attributes_offset: usize,
    vertex_attributes_stride: usize,
    attribute_weights: &[f32],
    attribute_count: usize,
    vertex_lock: Option<&[u8]>,
    target_index_count: usize,
    target_error: f32,
    options: u32,
    result_error: Option<&mut f32>,
) -> Result<usize> {
    index::output(indices, |indices| {
        simplify_with_update_interleaved_u32(
            indices,
            index_count,
            vertex_data,
            vertex_count,
            vertex_positions_stride,
            vertex_attributes_offset,
            vertex_attributes_stride,
            attribute_weights,
            attribute_count,
            vertex_lock,
            target_index_count,
            target_error,
            options,
            result_error,
        )
    })
}

fn simplify_with_update_interleaved_u32(
    indices: &mut [u32],
    index_count: usize,
    vertex_data: &mut [f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    vertex_attributes_offset: usize,
    vertex_attributes_stride: usize,
    attribute_weights: &[f32],
    attribute_count: usize,
    vertex_lock: Option<&[u8]>,
    target_index_count: usize,
    target_error: f32,
    options: u32,
    result_error: Option<&mut f32>,
) -> Result<usize> {
    let (count, error, solve) = {
        let attributes = vertex_data
            .get(vertex_attributes_offset..)
            .ok_or(Error::BufferTooSmall("vertex_data"))?;
        let input = EdgeInput {
            positions: vertex_data,
            positions_stride: vertex_positions_stride,
            attributes,
            attributes_stride: vertex_attributes_stride,
            weights: attribute_weights,
            attribute_count,
            vertex_lock,
        };
        validate_edge_input(&input, indices, index_count, vertex_count, target_index_count, target_error)?;

        simplify_edge(
            &mut indices[..index_count],
            &input,
            vertex_count,
            target_index_count,
            target_error,
            options | SIMPLIFY_INTERNAL_SOLVE,
        )
    };

    if let Some(out) = &solve {
        let target = AttributeTarget::Interleaved(vertex_attributes_offset);
        finalize_vertices(
            vertex_data,
            vertex_positions_stride,
            target,
            vertex_attributes_stride,
            attribute_weights,
            out,
            vertex_lock,
        );
    }

    if let Some(e) = result_error {
        *e = error;
    }

    Ok(count)
}

/// Mesh simplifier (sloppy).
/// Reduces the number of triangles in the mesh, sacrificing mesh appearance for simplification performance.
/// The algorithm doesn't preserve mesh topology but can stop short of the target goal based on target error.
/// `vertex_lock` can be `None`; otherwise vertices that can't be moved should set 1 consistently for all indices with the same position.
/// `target_error` is relative to mesh extents, value range [0..1].
pub fn simplify_sloppy<T: IndexType>(
    destination: &mut [T],
    indices: &[T],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    vertex_lock: Option<&[u8]>,
    target_index_count: usize,
    target_error: f32,
    result_error: Option<&mut f32>,
) -> Result<usize> {
    let indices = index::input(indices);
    index::output(destination, |destination| {
        simplify_sloppy_u32(
            destination,
            &indices,
            index_count,
            vertex_positions,
            vertex_count,
            vertex_positions_stride,
            vertex_lock,
            target_index_count,
            target_error,
            result_error,
        )
    })
}

fn simplify_sloppy_u32(
    destination: &mut [u32],
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    vertex_lock: Option<&[u8]>,
    target_index_count: usize,
    target_error: f32,
    result_error: Option<&mut f32>,
) -> Result<usize> {
    validate::triangles(index_count)?;
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    ensure(
        target_index_count <= index_count,
        Error::InvalidArgument("target_index_count must be <= index_count"),
    )?;
    ensure(target_error >= 0.0, Error::InvalidArgument("target_error must be >= 0"))?;
    if let Some(lock) = vertex_lock {
        ensure(lock.len() >= vertex_count, Error::BufferTooSmall("vertex_lock"))?;
    }
    let indices = validate::indices(indices, index_count, vertex_count)?;

    // we expect to get ~2 triangles/vertex in the output
    let target_cell_count = target_index_count / 6;

    let positions = Positions {
        data: vertex_positions,
        stride_float: vertex_positions_stride / 4,
        sparse_remap: None,
    };

    let mut vertex_positions = vec![Vector3::default(); vertex_count];
    rescale_positions(Some(&mut vertex_positions), positions, vertex_count);

    // find the optimal grid size using guided binary search
    let mut vertex_ids = vec![0u32; vertex_count];

    const INTERPOLATION_PASSES: i32 = 5;

    // invariant: # of triangles in min_grid <= target_count
    let mut min_grid = (1.0
        / if target_error < 1e-3 {
            1e-3
        } else if target_error < 1.0 {
            target_error
        } else {
            1.0
        }) as i32;
    let mut max_grid = 1025;
    let mut min_triangles = 0;
    let mut max_triangles = index_count / 3;

    // when we're error-limited, we compute the triangle count for the min. size; this accelerates convergence and provides the correct answer when we can't use a larger grid
    if min_grid > 1 || vertex_lock.is_some() {
        compute_vertex_ids(&mut vertex_ids, &vertex_positions, vertex_lock, min_grid);
        min_triangles = count_triangles(&vertex_ids, indices);
    }

    // instead of starting in the middle, let's guess as to what the answer might be! triangle count usually grows as a square of grid size...
    let mut next_grid_size = ((target_cell_count as f32).sqrt() + 0.5) as i32;

    for pass in 0..10 + INTERPOLATION_PASSES {
        if min_triangles >= target_index_count / 3 || max_grid - min_grid <= 1 {
            break;
        }

        // we clamp the prediction of the grid size to make sure that the search converges
        let mut grid_size = next_grid_size;
        grid_size = if grid_size <= min_grid {
            min_grid + 1
        } else if grid_size >= max_grid {
            max_grid - 1
        } else {
            grid_size
        };

        compute_vertex_ids(&mut vertex_ids, &vertex_positions, vertex_lock, grid_size);
        let triangles = count_triangles(&vertex_ids, indices);

        let tip = interpolate(
            (target_index_count / 3) as f32,
            min_grid as f32,
            min_triangles as f32,
            grid_size as f32,
            triangles as f32,
            max_grid as f32,
            max_triangles as f32,
        );

        if triangles <= target_index_count / 3 {
            min_grid = grid_size;
            min_triangles = triangles;
        } else {
            max_grid = grid_size;
            max_triangles = triangles;
        }

        // we start by using interpolation search - it usually converges faster
        // however, interpolation search has a worst case of O(N) so we switch to binary search after a few iterations which converges in O(logN)
        next_grid_size = if pass < INTERPOLATION_PASSES {
            (tip + 0.5) as i32
        } else {
            (min_grid + max_grid) / 2
        };
    }

    if min_triangles == 0 {
        if let Some(e) = result_error {
            *e = 1.0;
        }

        return Ok(0);
    }

    // build vertex->cell association by mapping all vertices with the same quantized position to the same cell
    let mut table = vec![0u32; hash_buckets(vertex_count)];

    let mut vertex_cells = vec![0u32; vertex_count];

    compute_vertex_ids(&mut vertex_ids, &vertex_positions, vertex_lock, min_grid);
    let cell_count = fill_vertex_cells(&mut table, &mut vertex_cells, &vertex_ids);

    // build a quadric for each target cell
    let mut cell_quadrics = vec![Quadric::default(); cell_count];

    fill_cell_quadrics(&mut cell_quadrics, indices, &vertex_positions, &vertex_cells);

    // for each target cell, find the vertex with the minimal error
    let mut cell_remap = vec![0u32; cell_count];
    let mut cell_errors = vec![0.0f32; cell_count];

    fill_cell_remap_quadrics(&mut cell_remap, &mut cell_errors, &vertex_cells, &cell_quadrics, &vertex_positions);

    // compute error
    let mut error = 0.0f32;

    for &e in &cell_errors {
        error = if error < e { e } else { error };
    }

    // vertex collapses often result in duplicate triangles; we need a table to filter them out
    let mut tritable = vec![0u32; hash_buckets(min_triangles)];

    // output is bounded by min_triangles; C callers may pass smaller (or no) buffers when the result is known to be empty
    let destination = validate::output(destination, min_triangles * 3)?;

    let write = filter_triangles(destination, &mut tritable, indices, &vertex_cells, &cell_remap);

    if let Some(e) = result_error {
        *e = error.sqrt();
    }

    Ok(write)
}

/// Mesh simplifier (pruner).
/// Reduces the number of triangles in the mesh by removing small isolated parts of the mesh.
/// `destination` must hold `index_count` elements; `target_error` is relative to mesh extents, value range [0..1].
pub fn simplify_prune<T: IndexType>(
    destination: &mut [T],
    indices: &[T],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    target_error: f32,
) -> Result<usize> {
    let indices = index::input(indices);
    index::output(destination, |destination| {
        simplify_prune_u32(
            destination,
            &indices,
            index_count,
            vertex_positions,
            vertex_count,
            vertex_positions_stride,
            target_error,
        )
    })
}

fn simplify_prune_u32(
    destination: &mut [u32],
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    target_error: f32,
) -> Result<usize> {
    validate::triangles(index_count)?;
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    ensure(target_error >= 0.0, Error::InvalidArgument("target_error must be >= 0"))?;
    let indices = validate::indices(indices, index_count, vertex_count)?;
    let result = validate::output(destination, index_count)?;

    result.copy_from_slice(indices);

    let positions = Positions {
        data: vertex_positions,
        stride_float: vertex_positions_stride / 4,
        sparse_remap: None,
    };

    // build position remap that maps each vertex to the one with identical position
    let mut remap = vec![0u32; vertex_count];
    build_position_remap(&mut remap, None, positions);

    let mut vertex_positions = vec![Vector3::default(); vertex_count];
    rescale_positions(Some(&mut vertex_positions), positions, vertex_count);

    let mut components = vec![0u32; vertex_count];
    let component_count = build_components(&mut components, indices, &remap);

    let mut component_errors = vec![0.0f32; component_count * 4]; // overallocate for temporary use inside measureComponents
    measure_components(&mut component_errors, component_count, &components, &vertex_positions);

    let mut component_nexterror = 0.0f32;
    Ok(prune_components(
        result,
        &components,
        &component_errors,
        target_error * target_error,
        &mut component_nexterror,
    ))
}

/// Point cloud simplifier.
/// Reduces the number of points in the cloud to reach the given target; returns the number of points after simplification.
/// `destination` must hold `target_vertex_count` elements. `vertex_colors` can be `None`; otherwise it holds float3 colors.
/// `color_weight` determines relative priority of color wrt position; 1.0 is a safe default.
pub fn simplify_points(
    destination: &mut [u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    vertex_colors: Option<&[f32]>,
    vertex_colors_stride: usize,
    color_weight: f32,
    target_vertex_count: usize,
) -> Result<usize> {
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    match vertex_colors {
        Some(colors) => validate::floats(colors, vertex_count, vertex_colors_stride, 12)?,
        None => ensure(
            vertex_colors_stride == 0 || ((12..=256).contains(&vertex_colors_stride) && vertex_colors_stride.is_multiple_of(4)),
            Error::InvalidArgument("vertex_colors_stride out of range"),
        )?,
    }
    ensure(
        target_vertex_count <= vertex_count,
        Error::InvalidArgument("target_vertex_count must be <= vertex_count"),
    )?;
    let destination = validate::output(destination, target_vertex_count)?;

    let target_cell_count = target_vertex_count;

    if target_cell_count == 0 {
        return Ok(0);
    }

    let positions = Positions {
        data: vertex_positions,
        stride_float: vertex_positions_stride / 4,
        sparse_remap: None,
    };

    let mut vertex_positions = vec![Vector3::default(); vertex_count];
    rescale_positions(Some(&mut vertex_positions), positions, vertex_count);

    // find the optimal grid size using guided binary search
    let mut vertex_ids = vec![0u32; vertex_count];

    let mut table = vec![0u32; hash_buckets(vertex_count)];

    const INTERPOLATION_PASSES: i32 = 5;

    // invariant: # of vertices in min_grid <= target_count
    let mut min_grid = 0;
    let mut max_grid = 1025;
    let mut min_vertices = 0;
    let mut max_vertices = vertex_count;

    // instead of starting in the middle, let's guess as to what the answer might be! surface point count usually grows as a square of grid size...
    let mut next_grid_size = ((target_cell_count as f32).sqrt() + 0.5) as i32;

    for pass in 0..10 + INTERPOLATION_PASSES {
        // we clamp the prediction of the grid size to make sure that the search converges
        let mut grid_size = next_grid_size;
        grid_size = if grid_size <= min_grid {
            min_grid + 1
        } else if grid_size >= max_grid {
            max_grid - 1
        } else {
            grid_size
        };

        compute_vertex_ids(&mut vertex_ids, &vertex_positions, None, grid_size);
        let vertices = count_vertex_cells(&mut table, &vertex_ids);

        let tip = interpolate(
            target_vertex_count as f32,
            min_grid as f32,
            min_vertices as f32,
            grid_size as f32,
            vertices as f32,
            max_grid as f32,
            max_vertices as f32,
        );

        if vertices <= target_vertex_count {
            min_grid = grid_size;
            min_vertices = vertices;
        } else {
            max_grid = grid_size;
            max_vertices = vertices;
        }

        if vertices == target_vertex_count || max_grid - min_grid <= 1 {
            break;
        }

        // we start by using interpolation search - it usually converges faster
        // however, interpolation search has a worst case of O(N) so we switch to binary search after a few iterations which converges in O(logN)
        next_grid_size = if pass < INTERPOLATION_PASSES {
            (tip + 0.5) as i32
        } else {
            (min_grid + max_grid) / 2
        };
    }

    if min_vertices == 0 {
        return Ok(0);
    }

    // build vertex->cell association by mapping all vertices with the same quantized position to the same cell
    let mut vertex_cells = vec![0u32; vertex_count];

    compute_vertex_ids(&mut vertex_ids, &vertex_positions, None, min_grid);
    let cell_count = fill_vertex_cells(&mut table, &mut vertex_cells, &vertex_ids);

    // accumulate points into a reservoir for each target cell
    let mut cell_reservoirs = vec![Reservoir::default(); cell_count];

    let colors = Colors {
        data: vertex_colors,
        stride_float: vertex_colors_stride / 4,
    };
    fill_cell_reservoirs(&mut cell_reservoirs, &vertex_positions, colors, &vertex_cells);

    // for each target cell, find the vertex with the minimal error
    let mut cell_remap = vec![0u32; cell_count];
    let mut cell_errors = vec![0.0f32; cell_count];

    // we scale the color weight to bring it to the same scale as position so that error addition makes sense
    let color_weight_scaled = color_weight * if min_grid == 1 { 1.0 } else { 1.0 / (min_grid - 1) as f32 };

    fill_cell_remap_reservoirs(
        &mut cell_remap,
        &mut cell_errors,
        &vertex_cells,
        &cell_reservoirs,
        &vertex_positions,
        colors,
        color_weight_scaled * color_weight_scaled,
    );

    // copy results to the output
    destination[..cell_count].copy_from_slice(&cell_remap);

    Ok(cell_count)
}

/// Returns the error scaling factor used by the simplifier to convert between absolute and relative extents.
/// Absolute error must be *divided* by the scaling factor before passing it to [`simplify`] as `target_error`;
/// relative error returned via `result_error` must be *multiplied* by the scaling factor to get absolute error.
pub fn simplify_scale(vertex_positions: &[f32], vertex_count: usize, vertex_positions_stride: usize) -> Result<f32> {
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;

    let positions = Positions {
        data: vertex_positions,
        stride_float: vertex_positions_stride / 4,
        sparse_remap: None,
    };

    Ok(rescale_positions(None, positions, vertex_count).0)
}
