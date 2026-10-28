// This work is based on:
// Graham Wihlidal. Optimizing the Graphics Pipeline with Compute. 2016
// Ingo Wald, Vlastimil Havran. On building fast kd-Trees for Ray Tracing, and on doing that in O(N log N). 2006

use crate::error::{Error, Result, ensure};
use crate::index::{self, IndexType};
use crate::validate;

// We keep a limited number of seed triangles and add a few triangles per finished meshlet
const MESHLET_MAX_SEEDS: usize = 256;
const MESHLET_ADD_SEEDS: usize = 4;

// To avoid excessive recursion for malformed inputs, we limit the maximum depth of the tree
const MESHLET_MAX_TREE_DEPTH: i32 = 50;

/// Meshlet is a small mesh cluster (subset) that consists of:
/// - triangles, an 8-bit micro triangle (index) buffer, that for each triangle specifies three local vertices to use;
/// - vertices, a 32-bit vertex indirection buffer, that for each local vertex specifies which mesh vertex to fetch vertex attributes from.
///
/// For efficiency, meshlet triangles and vertices are packed into two large arrays; this structure contains offsets and counts to access the data.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Meshlet {
    /// offsets within meshlet_vertices and meshlet_triangles arrays with meshlet data
    pub vertex_offset: u32,
    pub triangle_offset: u32,

    /// number of vertices and triangles used in the meshlet; data is stored in consecutive range [offset..offset+count) for vertices and [offset..offset+count*3) for triangles
    pub vertex_count: u32,
    pub triangle_count: u32,
}

struct TriangleAdjacency2 {
    counts: Vec<u32>,
    offsets: Vec<u32>,
    data: Vec<u32>,
}

fn build_triangle_adjacency(indices: &[u32], vertex_count: usize) -> TriangleAdjacency2 {
    let mut counts = vec![0u32; vertex_count];
    let mut offsets = vec![0u32; vertex_count];
    let mut data = vec![0u32; indices.len()];

    // fill triangle counts
    for &i in indices {
        counts[i as usize] += 1;
    }

    // fill offset table
    let mut offset = 0u32;

    for (o, &c) in offsets.iter_mut().zip(&counts) {
        *o = offset;
        offset += c;
    }

    // fill triangle data
    for (i, tri) in indices.as_chunks::<3>().0.iter().enumerate() {
        for &v in tri {
            data[offsets[v as usize] as usize] = i as u32;
            offsets[v as usize] += 1;
        }
    }

    // fix offsets that have been disturbed by the previous pass
    for (o, &c) in offsets.iter_mut().zip(&counts) {
        *o -= c;
    }

    TriangleAdjacency2 { counts, offsets, data }
}

fn build_triangle_adjacency_sparse(indices: &[u32], vertex_count: usize) -> TriangleAdjacency2 {
    // sparse mode can build adjacency more quickly by ignoring unused vertices, using a bit to mark visited vertices
    const SPARSE_SEEN: u32 = 1 << 31;

    let mut counts = vec![0u32; vertex_count];
    let mut offsets = vec![0u32; vertex_count];
    let mut data = vec![0u32; indices.len()];

    // fill triangle counts
    for &i in indices {
        counts[i as usize] += 1;
    }

    // fill offset table; uses sparse_seen bit to tag visited vertices
    let mut offset = 0u32;

    for &v in indices {
        let v = v as usize;

        if counts[v] & SPARSE_SEEN == 0 {
            offsets[v] = offset;
            offset += counts[v];
            counts[v] |= SPARSE_SEEN;
        }
    }

    // fill triangle data
    for (i, tri) in indices.as_chunks::<3>().0.iter().enumerate() {
        for &v in tri {
            data[offsets[v as usize] as usize] = i as u32;
            offsets[v as usize] += 1;
        }
    }

    // fix offsets that have been disturbed by the previous pass
    // also fix counts (that were marked with sparse_seen by the first pass)
    for &v in indices {
        let v = v as usize;

        if counts[v] & SPARSE_SEEN != 0 {
            counts[v] &= !SPARSE_SEEN;
            offsets[v] -= counts[v];
        }
    }

    TriangleAdjacency2 { counts, offsets, data }
}

#[derive(Debug, Clone, Copy, Default)]
struct Cone {
    px: f32,
    py: f32,
    pz: f32,
    nx: f32,
    ny: f32,
    nz: f32,
}

fn get_meshlet_score(distance: f32, spread: f32, cone_weight: f32, expected_radius: f32) -> f32 {
    let cone = 1.0 - spread * cone_weight;
    let cone_clamped = if cone < 1e-3 { 1e-3 } else { cone };

    (1.0 + distance / expected_radius * (1.0 - cone_weight)) * cone_clamped
}

fn get_meshlet_cone(acc: &Cone, triangle_count: u32) -> Cone {
    let mut result = *acc;

    let center_scale = if triangle_count == 0 { 0.0 } else { 1.0 / triangle_count as f32 };

    result.px *= center_scale;
    result.py *= center_scale;
    result.pz *= center_scale;

    let axis_length = result.nx * result.nx + result.ny * result.ny + result.nz * result.nz;
    let axis_scale = if axis_length == 0.0 { 0.0 } else { 1.0 / axis_length.sqrt() };

    result.nx *= axis_scale;
    result.ny *= axis_scale;
    result.nz *= axis_scale;

    result
}

fn compute_triangle_cones(triangles: &mut [Cone], indices: &[u32], vertex_positions: &[f32], vertex_positions_stride: usize) -> f32 {
    let vertex_stride_float = vertex_positions_stride / 4;

    let mut mesh_area = 0.0f32;

    for (t, tri) in triangles.iter_mut().zip(indices.as_chunks::<3>().0) {
        let p0 = &vertex_positions[vertex_stride_float * tri[0] as usize..];
        let p1 = &vertex_positions[vertex_stride_float * tri[1] as usize..];
        let p2 = &vertex_positions[vertex_stride_float * tri[2] as usize..];

        let p10 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let p20 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

        let normalx = p10[1] * p20[2] - p10[2] * p20[1];
        let normaly = p10[2] * p20[0] - p10[0] * p20[2];
        let normalz = p10[0] * p20[1] - p10[1] * p20[0];

        let area = (normalx * normalx + normaly * normaly + normalz * normalz).sqrt();
        let invarea = if area == 0.0 { 0.0 } else { 1.0 / area };

        t.px = (p0[0] + p1[0] + p2[0]) / 3.0;
        t.py = (p0[1] + p1[1] + p2[1]) / 3.0;
        t.pz = (p0[2] + p1[2] + p2[2]) / 3.0;

        t.nx = normalx * invarea;
        t.ny = normaly * invarea;
        t.nz = normalz * invarea;

        mesh_area += area;
    }

    mesh_area
}

/// Output buffers shared by all meshlet builders.
struct MeshletOutput<'a> {
    meshlets: &'a mut [Meshlet],
    meshlet_vertices: &'a mut [u32],
    meshlet_triangles: &'a mut [u8],
}

/// Appends a triangle; output writes are bounds checked since callers may size buffers below the documented worst case.
fn append_meshlet(
    meshlet: &mut Meshlet,
    a: u32,
    b: u32,
    c: u32,
    used: &mut [i16],
    out: &mut MeshletOutput,
    meshlet_offset: usize,
    max_vertices: usize,
    max_triangles: usize,
    split: bool,
) -> Result<bool> {
    let (a, b, c) = (a as usize, b as usize, c as usize);

    let mut result = false;

    let used_extra = usize::from(used[a] < 0) + usize::from(used[b] < 0) + usize::from(used[c] < 0);

    if meshlet.vertex_count as usize + used_extra > max_vertices || meshlet.triangle_count as usize >= max_triangles || split {
        *out.meshlets.get_mut(meshlet_offset).ok_or(Error::BufferTooSmall("meshlets"))? = *meshlet;

        let vo = meshlet.vertex_offset as usize;
        for &v in &out.meshlet_vertices[vo..vo + meshlet.vertex_count as usize] {
            used[v as usize] = -1;
        }

        meshlet.vertex_offset += meshlet.vertex_count;
        meshlet.triangle_offset += meshlet.triangle_count * 3;
        meshlet.vertex_count = 0;
        meshlet.triangle_count = 0;

        result = true;
    }

    for v in [a, b, c] {
        if used[v] < 0 {
            used[v] = meshlet.vertex_count as i16;
            let slot = (meshlet.vertex_offset + meshlet.vertex_count) as usize;
            *out.meshlet_vertices
                .get_mut(slot)
                .ok_or(Error::BufferTooSmall("meshlet_vertices"))? = v as u32;
            meshlet.vertex_count += 1;
        }
    }

    let to = (meshlet.triangle_offset + meshlet.triangle_count * 3) as usize;
    let tri = out
        .meshlet_triangles
        .get_mut(to..to + 3)
        .ok_or(Error::BufferTooSmall("meshlet_triangles"))?;
    tri.copy_from_slice(&[used[a] as u8, used[b] as u8, used[c] as u8]);
    meshlet.triangle_count += 1;

    Ok(result)
}

fn flush_meshlet(meshlet: &Meshlet, out: &mut MeshletOutput, meshlet_offset: &mut usize) -> Result<()> {
    if meshlet.triangle_count != 0 {
        *out.meshlets.get_mut(*meshlet_offset).ok_or(Error::BufferTooSmall("meshlets"))? = *meshlet;
        *meshlet_offset += 1;
    }

    Ok(())
}

fn triangle(indices: &[u32], t: u32) -> [usize; 3] {
    let t = t as usize * 3;
    [indices[t] as usize, indices[t + 1] as usize, indices[t + 2] as usize]
}

fn neighbors(adjacency: &TriangleAdjacency2, index: usize) -> &[u32] {
    let offset = adjacency.offsets[index] as usize;
    &adjacency.data[offset..offset + adjacency.counts[index] as usize]
}

fn get_neighbor_triangle(
    meshlet: &Meshlet,
    meshlet_cone: &Cone,
    meshlet_vertices: &[u32],
    indices: &[u32],
    adjacency: &TriangleAdjacency2,
    triangles: &[Cone],
    used: &[i16],
    meshlet_expected_radius: f32,
    cone_weight: f32,
) -> u32 {
    let live_triangles = &adjacency.counts;

    let mut best_triangle = !0u32;
    let mut best_priority = 5;
    let mut best_score = f32::MAX;

    let vo = meshlet.vertex_offset as usize;

    for &index in &meshlet_vertices[vo..vo + meshlet.vertex_count as usize] {
        for &triangle_index in neighbors(adjacency, index as usize) {
            let [a, b, c] = triangle(indices, triangle_index);

            let extra = i32::from(used[a] < 0) + i32::from(used[b] < 0) + i32::from(used[c] < 0);

            // triangles that don't add new vertices to meshlets are max. priority
            let priority = if extra == 0 {
                0
            }
            // artificially increase the priority of dangling triangles as they're expensive to add to new meshlets
            else if live_triangles[a] == 1 || live_triangles[b] == 1 || live_triangles[c] == 1 {
                1
            }
            // if two vertices have live count of 2, removing this triangle will make another triangle dangling which is good for overall flow
            else if i32::from(live_triangles[a] == 2) + i32::from(live_triangles[b] == 2) + i32::from(live_triangles[c] == 2) >= 2 {
                1 + extra
            }
            // otherwise adjust priority to be after the above cases, 3 or 4 based on used[] count
            else {
                2 + extra
            };

            // since topology-based priority is always more important than the score, we can skip scoring in some cases
            if priority > best_priority {
                continue;
            }

            let tri_cone = &triangles[triangle_index as usize];

            let (dx, dy, dz) = (
                tri_cone.px - meshlet_cone.px,
                tri_cone.py - meshlet_cone.py,
                tri_cone.pz - meshlet_cone.pz,
            );
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            let spread = tri_cone.nx * meshlet_cone.nx + tri_cone.ny * meshlet_cone.ny + tri_cone.nz * meshlet_cone.nz;

            let score = get_meshlet_score(distance, spread, cone_weight, meshlet_expected_radius);

            // note that topology-based priority is always more important than the score
            // this helps maintain reasonable effectiveness of meshlet data and reduces scoring cost
            if priority < best_priority || score < best_score {
                best_triangle = triangle_index;
                best_priority = priority;
                best_score = score;
            }
        }
    }

    best_triangle
}

fn append_seed_triangles(
    seeds: &mut [u32],
    meshlet: &Meshlet,
    meshlet_vertices: &[u32],
    indices: &[u32],
    adjacency: &TriangleAdjacency2,
    triangles: &[Cone],
    corner: [f32; 3],
) -> usize {
    let live_triangles = &adjacency.counts;

    let mut best_seeds = [!0u32; MESHLET_ADD_SEEDS];
    let mut best_live = [!0u32; MESHLET_ADD_SEEDS];
    let mut best_score = [f32::MAX; MESHLET_ADD_SEEDS];

    let vo = meshlet.vertex_offset as usize;

    for &index in &meshlet_vertices[vo..vo + meshlet.vertex_count as usize] {
        let mut best_neighbor = !0u32;
        let mut best_neighbor_live = !0u32;

        // find the neighbor with the smallest live metric
        for &triangle_index in neighbors(adjacency, index as usize) {
            let [a, b, c] = triangle(indices, triangle_index);

            let live = live_triangles[a] + live_triangles[b] + live_triangles[c];

            if live < best_neighbor_live {
                best_neighbor = triangle_index;
                best_neighbor_live = live;
            }
        }

        // add the neighbor to the list of seeds; the list is unsorted and the replacement criteria is approximate
        if best_neighbor == !0u32 {
            continue;
        }

        let t = &triangles[best_neighbor as usize];
        let (dx, dy, dz) = (t.px - corner[0], t.py - corner[1], t.pz - corner[2]);
        let best_neighbor_score = (dx * dx + dy * dy + dz * dz).sqrt();

        for j in 0..MESHLET_ADD_SEEDS {
            // non-strict comparison reduces the number of duplicate seeds (triangles adjacent to multiple vertices)
            if best_neighbor_live < best_live[j] || (best_neighbor_live == best_live[j] && best_neighbor_score <= best_score[j]) {
                best_seeds[j] = best_neighbor;
                best_live[j] = best_neighbor_live;
                best_score[j] = best_neighbor_score;
                break;
            }
        }
    }

    // add surviving seeds to the meshlet
    let mut seed_count = 0;

    for &s in &best_seeds {
        if s != !0u32 {
            seeds[seed_count] = s;
            seed_count += 1;
        }
    }

    seed_count
}

fn prune_seed_triangles(seeds: &mut [u32], seed_count: usize, emitted_flags: &[bool]) -> usize {
    let mut result = 0;

    for i in 0..seed_count {
        let index = seeds[i];

        seeds[result] = index;
        result += usize::from(!emitted_flags[index as usize]);
    }

    result
}

fn select_seed_triangle(seeds: &[u32], indices: &[u32], triangles: &[Cone], live_triangles: &[u32], corner: [f32; 3]) -> u32 {
    let mut best_seed = !0u32;
    let mut best_live = !0u32;
    let mut best_score = f32::MAX;

    for &index in seeds {
        let [a, b, c] = triangle(indices, index);

        let live = live_triangles[a] + live_triangles[b] + live_triangles[c];
        let t = &triangles[index as usize];
        let (dx, dy, dz) = (t.px - corner[0], t.py - corner[1], t.pz - corner[2]);
        let score = (dx * dx + dy * dy + dz * dz).sqrt();

        if live < best_live || (live == best_live && score < best_score) {
            best_seed = index;
            best_live = live;
            best_score = score;
        }
    }

    best_seed
}

/// kd-tree node; `value` holds the split plane (f32 bits) for branches or the point index for leaves.
#[derive(Debug, Clone, Copy, Default)]
struct KdNode {
    value: u32,
    // leaves: axis = 3, children = number of points including this one
    // branches: axis != 3, left subtree = skip 1, right subtree = skip 1+children
    axis: u32,
    children: u32,
}

fn point(triangles: &[Cone], index: u32) -> [f32; 3] {
    let t = &triangles[index as usize];
    [t.px, t.py, t.pz]
}

fn kdtree_partition(indices: &mut [u32], points: &[Cone], axis: usize, pivot: f32) -> usize {
    let mut m = 0;

    // invariant: elements in range [0, m) are < pivot, elements in range [m, i) are >= pivot
    for i in 0..indices.len() {
        let v = point(points, indices[i])[axis];

        // swap(m, i) unconditionally
        indices.swap(m, i);

        // when v >= pivot, we swap i with m without advancing it, preserving invariants
        m += usize::from(v < pivot);
    }

    m
}

fn kdtree_build_leaf(offset: usize, nodes: &mut [KdNode], indices: &[u32]) -> usize {
    let count = indices.len();

    nodes[offset] = KdNode {
        value: indices[0],
        axis: 3,
        children: count as u32,
    };

    // all remaining points are stored in nodes immediately following the leaf
    for i in 1..count {
        nodes[offset + i] = KdNode {
            value: indices[i],
            axis: 3,
            children: !0u32 >> 2,
        }; // bogus value to prevent misuse
    }

    offset + count
}

fn kdtree_build(offset: usize, nodes: &mut [KdNode], points: &[Cone], indices: &mut [u32], leaf_size: usize, depth: i32) -> usize {
    let count = indices.len();

    if count <= leaf_size {
        return kdtree_build_leaf(offset, nodes, indices);
    }

    let mut mean = [0.0f32; 3];
    let mut vars = [0.0f32; 3];
    let mut runc = 1.0f32;
    let mut runs = 1.0f32;

    // gather statistics on the points in the subtree using Welford's algorithm
    for &index in indices.iter() {
        let p = point(points, index);

        for k in 0..3 {
            let delta = p[k] - mean[k];
            mean[k] += delta * runs;
            vars[k] += delta * (p[k] - mean[k]);
        }

        runc += 1.0;
        runs = 1.0 / runc;
    }

    // split axis is one where the variance is largest
    let axis = if vars[0] >= vars[1] && vars[0] >= vars[2] {
        0
    } else if vars[1] >= vars[2] {
        1
    } else {
        2
    };

    let split = mean[axis];
    let middle = kdtree_partition(indices, points, axis, split);

    // when the partition is degenerate, consolidate the points into a single node; also ensure recursion depth is bounded on pathological inputs
    if middle == 0 || middle == count || depth >= MESHLET_MAX_TREE_DEPTH {
        return kdtree_build_leaf(offset, nodes, indices);
    }

    // when the partition is unbalanced, consolidate the points into a single small node; large nodes are still split to prevent outlier points from creating very large leaves
    if (middle <= leaf_size / 2 || middle >= count - leaf_size / 2) && count <= leaf_size * 8 {
        return kdtree_build_leaf(offset, nodes, indices);
    }

    nodes[offset].value = split.to_bits();
    nodes[offset].axis = axis as u32;

    let (left, right) = indices.split_at_mut(middle);

    // left subtree is right after our node
    let next_offset = kdtree_build(offset + 1, nodes, points, left, leaf_size, depth + 1);

    // distance to the right subtree is represented explicitly
    nodes[offset].children = (next_offset - offset - 1) as u32;

    kdtree_build(next_offset, nodes, points, right, leaf_size, depth + 1)
}

fn kdtree_nearest(
    nodes: &mut [KdNode],
    root: usize,
    points: &[Cone],
    emitted_flags: &[bool],
    position: &[f32; 3],
    result: &mut u32,
    limit: &mut f32,
) {
    let node = nodes[root];

    if node.children == 0 {
        return;
    }

    if node.axis == 3 {
        // leaf
        let mut inactive = true;

        for i in 0..node.children as usize {
            let index = nodes[root + i].value;

            if emitted_flags[index as usize] {
                continue;
            }

            inactive = false;

            let p = point(points, index);

            let (dx, dy, dz) = (p[0] - position[0], p[1] - position[1], p[2] - position[2]);
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();

            if distance < *limit {
                *result = index;
                *limit = distance;
            }
        }

        // deactivate leaves that no longer have items to emit
        if inactive {
            nodes[root].children = 0;
        }
    } else {
        // branch; we order recursion to process the node that search position is in first
        let delta = position[node.axis as usize] - f32::from_bits(node.value);
        let first = if delta <= 0.0 { 0 } else { node.children as usize };
        let second = first ^ node.children as usize;

        // deactivate branches that no longer have items to emit to accelerate traversal
        // note that we do this *before* recursing which delays deactivation but keeps tail calls
        if (nodes[root + 1 + first].children | nodes[root + 1 + second].children) == 0 {
            nodes[root].children = 0;
        }

        // recursion depth is bounded by tree depth (which is limited by construction)
        kdtree_nearest(nodes, root + 1 + first, points, emitted_flags, position, result, limit);

        // only process the other node if it can have a match based on closest distance so far
        if delta.abs() <= *limit {
            kdtree_nearest(nodes, root + 1 + second, points, emitted_flags, position, result, limit);
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct BvhBox {
    min: [f32; 3],
    max: [f32; 3],
}

fn box_merge(acc: &mut BvhBox, other: &BvhBox) -> f32 {
    for k in 0..3 {
        acc.min[k] = if other.min[k] < acc.min[k] { other.min[k] } else { acc.min[k] };
        acc.max[k] = if other.max[k] > acc.max[k] { other.max[k] } else { acc.max[k] };
    }

    let (sx, sy, sz) = (acc.max[0] - acc.min[0], acc.max[1] - acc.min[1], acc.max[2] - acc.min[2]);
    sx * sy + sy * sz + sz * sx
}

fn radix_float(v: u32) -> u32 {
    // if sign bit is 0, flip sign bit
    // if sign bit is 1, flip everything
    let mask = ((v as i32) >> 31) as u32 | 0x8000_0000;
    v ^ mask
}

fn compute_histogram(data: &[f32]) -> Vec<[u32; 3]> {
    let mut hist = vec![[0u32; 3]; 1024];

    // compute 3 10-bit histograms in parallel (dropping 2 LSB)
    for &d in data {
        let id = radix_float(d.to_bits()) as usize;

        hist[(id >> 2) & 1023][0] += 1;
        hist[(id >> 12) & 1023][1] += 1;
        hist[(id >> 22) & 1023][2] += 1;
    }

    let mut sum = [0u32; 3];

    // replace histogram data with prefix histogram sums in-place
    for h in hist.iter_mut() {
        let old = *h;
        *h = sum;

        for k in 0..3 {
            sum[k] += old[k];
        }
    }

    hist
}

fn radix_pass(destination: &mut [u32], source: &[u32], keys: &[f32], hist: &mut [[u32; 3]], pass: usize) {
    let bitoff = pass * 10 + 2; // drop 2 LSB to be able to use 3 10-bit passes

    for &s in source {
        let id = ((radix_float(keys[s as usize].to_bits()) >> bitoff) & 1023) as usize;

        destination[hist[id][pass] as usize] = s;
        hist[id][pass] += 1;
    }
}

fn bvh_prepare(boxes: &mut [BvhBox], centroids: &mut [f32], indices: &[u32], vertex_positions: &[f32], vertex_stride_float: usize) {
    let face_count = boxes.len();

    for (i, (bx, tri)) in boxes.iter_mut().zip(indices.as_chunks::<3>().0).enumerate() {
        let va = &vertex_positions[vertex_stride_float * tri[0] as usize..];
        let vb = &vertex_positions[vertex_stride_float * tri[1] as usize..];
        let vc = &vertex_positions[vertex_stride_float * tri[2] as usize..];

        for k in 0..3 {
            bx.min[k] = if va[k] < vb[k] { va[k] } else { vb[k] };
            bx.min[k] = if vc[k] < bx.min[k] { vc[k] } else { bx.min[k] };

            bx.max[k] = if va[k] > vb[k] { va[k] } else { vb[k] };
            bx.max[k] = if vc[k] > bx.max[k] { vc[k] } else { bx.max[k] };

            centroids[i + face_count * k] = (bx.min[k] + bx.max[k]) / 2.0;
        }
    }
}

fn bvh_count_vertices(order: &[u32], used: &mut [i16], indices: &[u32], mut out: Option<&mut [u32]>) -> usize {
    // count number of unique vertices
    let mut used_vertices = 0;
    for (i, &index) in order.iter().enumerate() {
        let [a, b, c] = triangle(indices, index);

        used_vertices += usize::from(used[a] < 0) + usize::from(used[b] < 0) + usize::from(used[c] < 0);
        used[a] = 1;
        used[b] = 1;
        used[c] = 1;

        if let Some(out) = out.as_deref_mut() {
            out[i] = used_vertices as u32;
        }
    }

    // reset used[] for future invocations
    for &index in order {
        let [a, b, c] = triangle(indices, index);

        used[a] = -1;
        used[b] = -1;
        used[c] = -1;
    }

    used_vertices
}

fn bvh_pack_leaf(boundary: &mut [u8], count: usize) {
    // mark meshlet boundary for future reassembly
    boundary[0] = 1;
    boundary[1..count].fill(0);
}

fn bvh_pack_tail(boundary: &mut [u8], order: &[u32], used: &mut [i16], indices: &[u32], max_vertices: usize, max_triangles: usize) {
    let count = order.len();
    let mut i = 0;

    while i < count {
        let chunk = if i + max_triangles <= count { max_triangles } else { count - i };

        if bvh_count_vertices(&order[i..i + chunk], used, indices, None) <= max_vertices {
            bvh_pack_leaf(&mut boundary[i..], chunk);
            i += chunk;
            continue;
        }

        // chunk is vertex bound, split it into smaller meshlets
        bvh_pack_leaf(&mut boundary[i..], max_vertices / 3);
        i += max_vertices / 3;
    }
}

fn bvh_divisible(count: usize, min: usize, max: usize) -> bool {
    // count is representable as a sum of values in [min..max] if it is in range of [k*min..k*min+k*(max-min)]
    // equivalent to ceil(count / max) <= floor(count / min), but the form below allows using idiv (see nv_cluster_builder)
    // we avoid expensive integer divisions in the common case where min is <= max/2
    if min * 2 <= max {
        count >= min
    } else {
        count % min <= (count / min) * (max - min)
    }
}

fn bvh_compute_area(areas: &mut [f32], boxes: &[BvhBox], order: &[u32]) {
    let count = order.len();

    let mut accuml = BvhBox {
        min: [f32::MAX; 3],
        max: [-f32::MAX; 3],
    };
    let mut accumr = accuml;

    for i in 0..count {
        let larea = box_merge(&mut accuml, &boxes[order[i] as usize]);
        let rarea = box_merge(&mut accumr, &boxes[order[count - 1 - i] as usize]);

        areas[i] = larea;
        areas[i + count] = rarea;
    }
}

fn bvh_pivot(
    areas: &[f32],
    vertices: Option<&[u32]>,
    count: usize,
    step: usize,
    min: usize,
    max: usize,
    fill: f32,
    maxfill: usize,
    out_cost: &mut f32,
) -> usize {
    let aligned = count >= min * 2 && bvh_divisible(count, min, max);
    let end = if aligned { count - min } else { count - 1 };

    let rmaxfill = 1.0 / maxfill as i32 as f32;

    // find best split that minimizes SAH
    let mut bestsplit = 0;
    let mut bestcost = f32::MAX;

    let mut i = min - 1;
    while i < end {
        let lsplit = i + 1;
        let rsplit = count - (i + 1);

        if !bvh_divisible(lsplit, min, max) || (aligned && !bvh_divisible(rsplit, min, max)) {
            i += step;
            continue;
        }

        // areas[x] = inclusive surface area of boxes[0..x]
        // areas[count-1-x] = inclusive surface area of boxes[x..count-1]
        let larea = areas[i];
        let rarea = areas[(count - 1 - (i + 1)) + count];
        let mut cost = larea * lsplit as i32 as f32 + rarea * rsplit as i32 as f32;

        if cost > bestcost {
            i += step;
            continue;
        }

        // use vertex fill when splitting vertex limited clusters; note that we use the same (left->right) vertex count
        // using bidirectional vertex counts is a little more expensive to compute and produces slightly worse results in practice
        let lfill = vertices.map_or(lsplit, |v| v[i] as usize);
        let rfill = vertices.map_or(rsplit, |v| v[i] as usize);

        // fill cost; use floating point math to round up to maxfill to avoid expensive integer modulo
        let lrest = (((lfill + maxfill - 1) as i32 as f32 * rmaxfill) as i32) * maxfill as i32 - lfill as i32;
        let rrest = (((rfill + maxfill - 1) as i32 as f32 * rmaxfill) as i32) * maxfill as i32 - rfill as i32;

        cost += fill * (lrest as f32 * larea + rrest as f32 * rarea);

        if cost < bestcost {
            bestcost = cost;
            bestsplit = i + 1;
        }

        i += step;
    }

    *out_cost = bestcost;
    bestsplit
}

fn bvh_partition(target: &mut [u32], order: &[u32], sides: &[u8], split: usize) {
    let (mut l, mut r) = (0, split);

    for &o in order {
        let side = usize::from(sides[o as usize]);
        target[if side != 0 { r } else { l }] = o;
        l += 1;
        l -= side;
        r += side;
    }
}

struct BvhScratch {
    areas: Vec<f32>,
    vertices: Vec<u32>,
    temp: Vec<u32>,
    sides: Vec<u8>,
}

struct BvhParams<'a> {
    boxes: &'a [BvhBox],
    indices: &'a [u32],
    max_vertices: usize,
    min_triangles: usize,
    max_triangles: usize,
    fill_weight: f32,
}

fn bvh_split(params: &BvhParams, axes: [&mut [u32]; 3], boundary: &mut [u8], depth: i32, scratch: &mut BvhScratch, used: &mut [i16]) {
    let count = axes[0].len();
    let (indices, max_vertices, min_triangles, max_triangles) =
        (params.indices, params.max_vertices, params.min_triangles, params.max_triangles);

    if count <= max_triangles && bvh_count_vertices(axes[0], used, indices, None) <= max_vertices {
        return bvh_pack_leaf(boundary, count);
    }

    // we can use step=1 unconditionally but to reduce the cost for min=max case we use step=max
    let step = if min_triangles == max_triangles && count > max_triangles {
        max_triangles
    } else {
        1
    };

    // if we could not pack the meshlet, we must be vertex bound
    let mint = if count <= max_triangles && max_vertices / 3 < min_triangles {
        max_vertices / 3
    } else {
        min_triangles
    };
    let maxfill = if count <= max_triangles { max_vertices } else { max_triangles };

    // find best split that minimizes SAH
    let mut bestk = None;
    let mut bestsplit = 0;
    let mut bestcost = f32::MAX;

    for (k, axis) in axes.iter().enumerate() {
        bvh_compute_area(&mut scratch.areas, params.boxes, axis);

        let vertices = if count <= max_triangles {
            // for vertex bound clusters, count number of unique vertices for each split
            bvh_count_vertices(axis, used, indices, Some(&mut scratch.vertices));
            Some(&scratch.vertices[..count])
        } else {
            None
        };

        let mut axiscost = f32::MAX;
        let axissplit = bvh_pivot(
            &scratch.areas,
            vertices,
            count,
            step,
            mint,
            max_triangles,
            params.fill_weight,
            maxfill,
            &mut axiscost,
        );

        if axissplit != 0 && axiscost < bestcost {
            bestk = Some(k);
            bestcost = axiscost;
            bestsplit = axissplit;
        }
    }

    // this may happen if SAH costs along the admissible splits are NaN, or due to imbalanced splits on pathological inputs
    let bestk = match bestk {
        Some(k) if depth < MESHLET_MAX_TREE_DEPTH => k,
        _ => return bvh_pack_tail(boundary, axes[0], used, indices, max_vertices, max_triangles),
    };

    // mark sides of split for partitioning
    for (i, &t) in axes[bestk].iter().enumerate() {
        scratch.sides[t as usize] = u8::from(i >= bestsplit);
    }

    // partition all axes into two sides, maintaining order
    let mut axes = axes;

    for (k, axis) in axes.iter_mut().enumerate() {
        if k == bestk {
            continue;
        }

        scratch.temp[..count].copy_from_slice(axis);
        bvh_partition(axis, &scratch.temp[..count], &scratch.sides, bestsplit);
    }

    // recursion depth is bounded due to max depth check above
    let [ax, ay, az] = axes;
    let (lx, rx) = ax.split_at_mut(bestsplit);
    let (ly, ry) = ay.split_at_mut(bestsplit);
    let (lz, rz) = az.split_at_mut(bestsplit);
    let (lb, rb) = boundary.split_at_mut(bestsplit);

    bvh_split(params, [lx, ly, lz], lb, depth + 1, scratch, used);
    bvh_split(params, [rx, ry, rz], rb, depth + 1, scratch, used);
}

/// Worst case number of meshlets produced by the meshlet builders.
pub fn build_meshlets_bound(index_count: usize, max_vertices: usize, max_triangles: usize) -> Result<usize> {
    validate::triangles(index_count)?;
    ensure(
        (3..=256).contains(&max_vertices),
        Error::InvalidArgument("max_vertices must be in 3..=256"),
    )?;
    ensure(
        (1..=512).contains(&max_triangles),
        Error::InvalidArgument("max_triangles must be in 1..=512"),
    )?;

    // meshlet construction is limited by max vertices and max triangles per meshlet
    // the worst case is that the input is an unindexed stream since this equally stresses both limits
    // note that we assume that in the worst case, we leave 2 vertices unpacked in each meshlet - if we have space for 3 we can pack any triangle
    let max_vertices_conservative = max_vertices - 2;
    let meshlet_limit_vertices = index_count.div_ceil(max_vertices_conservative);
    let meshlet_limit_triangles = (index_count / 3).div_ceil(max_triangles);

    Ok(meshlet_limit_vertices.max(meshlet_limit_triangles))
}

/// Validates meshlet limits; output sizes are checked lazily on write (see [`append_meshlet`]).
fn validate_meshlet_output<'a>(
    meshlets: &'a mut [Meshlet],
    meshlet_vertices: &'a mut [u32],
    meshlet_triangles: &'a mut [u8],
    index_count: usize,
    max_vertices: usize,
    min_triangles: usize,
    max_triangles: usize,
) -> Result<MeshletOutput<'a>> {
    ensure(
        (3..=256).contains(&max_vertices),
        Error::InvalidArgument("max_vertices must be in 3..=256"),
    )?;
    ensure(
        min_triangles >= 1 && min_triangles <= max_triangles && max_triangles <= 512,
        Error::InvalidArgument("triangle limits must satisfy 1 <= min_triangles <= max_triangles <= 512"),
    )?;

    validate::triangles(index_count)?;

    Ok(MeshletOutput {
        meshlets,
        meshlet_vertices,
        meshlet_triangles,
    })
}

/// Meshlet builder with flexible cluster sizes.
/// Splits the mesh into a set of meshlets, similarly to [`build_meshlets`], but allows to specify minimum and maximum number of triangles per meshlet.
/// Clusters between min and max triangle counts are split when the cluster size would have exceeded the expected cluster size by more than `split_factor`.
///
/// `meshlets` must hold [`build_meshlets_bound`] (using `min_triangles`) entries; `meshlet_vertices` and `meshlet_triangles` must hold `index_count` entries.
pub fn build_meshlets_flex<T: IndexType>(
    meshlets: &mut [Meshlet],
    meshlet_vertices: &mut [u32],
    meshlet_triangles: &mut [u8],
    indices: &[T],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    max_vertices: usize,
    min_triangles: usize,
    max_triangles: usize,
    cone_weight: f32,
    split_factor: f32,
) -> Result<usize> {
    build_meshlets_flex_u32(
        meshlets,
        meshlet_vertices,
        meshlet_triangles,
        &index::input(indices),
        index_count,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        max_vertices,
        min_triangles,
        max_triangles,
        cone_weight,
        split_factor,
    )
}

fn build_meshlets_flex_u32(
    meshlets: &mut [Meshlet],
    meshlet_vertices: &mut [u32],
    meshlet_triangles: &mut [u8],
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    max_vertices: usize,
    min_triangles: usize,
    max_triangles: usize,
    cone_weight: f32,
    split_factor: f32,
) -> Result<usize> {
    validate::triangles(index_count)?;
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    ensure(
        (0.0..=1.0).contains(&cone_weight),
        Error::InvalidArgument("cone_weight must be in 0..=1"),
    )?;
    ensure(split_factor >= 0.0, Error::InvalidArgument("split_factor must be >= 0"))?;
    let mut out = validate_meshlet_output(
        meshlets,
        meshlet_vertices,
        meshlet_triangles,
        index_count,
        max_vertices,
        min_triangles,
        max_triangles,
    )?;
    let indices = validate::indices(indices, index_count, vertex_count)?;

    if index_count == 0 {
        return Ok(0);
    }

    let mut adjacency = if vertex_count > index_count && index_count < (1 << 31) {
        build_triangle_adjacency_sparse(indices, vertex_count)
    } else {
        build_triangle_adjacency(indices, vertex_count)
    };

    // live triangle counts; note, we alias adjacency.counts as we remove triangles after emitting them so the counts always match

    let face_count = index_count / 3;

    let mut emitted_flags = vec![false; face_count];

    // for each triangle, precompute centroid & normal to use for scoring
    let mut triangles = vec![Cone::default(); face_count];
    let mesh_area = compute_triangle_cones(&mut triangles, indices, vertex_positions, vertex_positions_stride);

    // assuming each meshlet is a square patch, expected radius is sqrt(expected area)
    let triangle_area_avg = if face_count == 0 {
        0.0
    } else {
        mesh_area / face_count as f32 * 0.5
    };
    let meshlet_expected_radius = (triangle_area_avg * max_triangles as f32).sqrt() * 0.5;

    // build a kd-tree for nearest neighbor lookup
    let mut kdindices: Vec<u32> = (0..face_count as u32).collect();

    let mut nodes = vec![KdNode::default(); face_count * 2];
    kdtree_build(0, &mut nodes, &triangles, &mut kdindices, 8, 0);

    // find a specific corner of the mesh to use as a starting point for meshlet flow
    let mut corner = [f32::MAX; 3];

    for tri in &triangles {
        corner[0] = if corner[0] > tri.px { tri.px } else { corner[0] };
        corner[1] = if corner[1] > tri.py { tri.py } else { corner[1] };
        corner[2] = if corner[2] > tri.pz { tri.pz } else { corner[2] };
    }

    // index of the vertex in the meshlet, -1 if the vertex isn't used
    let mut used = vec![-1i16; vertex_count];

    // initial seed triangle is the one closest to the corner
    let mut initial_seed = !0u32;
    let mut initial_score = f32::MAX;

    for (i, tri) in triangles.iter().enumerate() {
        let (dx, dy, dz) = (tri.px - corner[0], tri.py - corner[1], tri.pz - corner[2]);
        let score = (dx * dx + dy * dy + dz * dz).sqrt();

        if initial_seed == !0u32 || score < initial_score {
            initial_seed = i as u32;
            initial_score = score;
        }
    }

    // seed triangles to continue meshlet flow
    let mut seeds = [0u32; MESHLET_MAX_SEEDS];
    let mut seed_count = 0;

    let mut meshlet = Meshlet::default();
    let mut meshlet_offset = 0;

    let mut meshlet_cone_acc = Cone::default();

    loop {
        let meshlet_cone = get_meshlet_cone(&meshlet_cone_acc, meshlet.triangle_count);

        // for the first triangle, we don't have a meshlet cone yet, so we use the initial seed
        // to continue the meshlet, we select an adjacent triangle based on connectivity and spatial scoring
        let mut best_triangle = if meshlet_offset == 0 && meshlet.triangle_count == 0 {
            initial_seed
        } else {
            get_neighbor_triangle(
                &meshlet,
                &meshlet_cone,
                out.meshlet_vertices,
                indices,
                &adjacency,
                &triangles,
                &used,
                meshlet_expected_radius,
                cone_weight,
            )
        };

        let mut split = false;

        // when we run out of adjacent triangles we need to switch to spatial search; we currently just pick the closest triangle irrespective of connectivity
        if best_triangle == !0u32 {
            let position = [meshlet_cone.px, meshlet_cone.py, meshlet_cone.pz];
            let mut index = !0u32;
            let mut distance = f32::MAX;

            kdtree_nearest(&mut nodes, 0, &triangles, &emitted_flags, &position, &mut index, &mut distance);

            best_triangle = index;
            split =
                meshlet.triangle_count as usize >= min_triangles && split_factor > 0.0 && distance > meshlet_expected_radius * split_factor;
        }

        if best_triangle == !0u32 {
            break;
        }

        let [a, b, c] = triangle(indices, best_triangle);
        let best_extra = usize::from(used[a] < 0) + usize::from(used[b] < 0) + usize::from(used[c] < 0);

        // if the best triangle doesn't fit into current meshlet, we re-select using seeds to maintain global flow
        if split || (meshlet.vertex_count as usize + best_extra > max_vertices || meshlet.triangle_count as usize >= max_triangles) {
            seed_count = prune_seed_triangles(&mut seeds, seed_count, &emitted_flags);
            seed_count = if seed_count + MESHLET_ADD_SEEDS <= MESHLET_MAX_SEEDS {
                seed_count
            } else {
                MESHLET_MAX_SEEDS - MESHLET_ADD_SEEDS
            };
            seed_count += append_seed_triangles(
                &mut seeds[seed_count..],
                &meshlet,
                out.meshlet_vertices,
                indices,
                &adjacency,
                &triangles,
                corner,
            );

            let best_seed = select_seed_triangle(&seeds[..seed_count], indices, &triangles, &adjacency.counts, corner);

            // we may not find a valid seed triangle if the mesh is disconnected as seeds are based on adjacency
            best_triangle = if best_seed != !0u32 { best_seed } else { best_triangle };
        }

        let [a, b, c] = triangle(indices, best_triangle);

        // add meshlet to the output; when the current meshlet is full we reset the accumulated bounds
        if append_meshlet(
            &mut meshlet,
            a as u32,
            b as u32,
            c as u32,
            &mut used,
            &mut out,
            meshlet_offset,
            max_vertices,
            max_triangles,
            split,
        )? {
            meshlet_offset += 1;
            meshlet_cone_acc = Cone::default();
        }

        // remove emitted triangle from adjacency data
        // this makes sure that we spend less time traversing these lists on subsequent iterations
        // live triangle counts are updated as a byproduct of these adjustments
        for index in [a, b, c] {
            let offset = adjacency.offsets[index] as usize;
            let neighbors = &mut adjacency.data[offset..offset + adjacency.counts[index] as usize];

            if let Some(i) = neighbors.iter().position(|&tri| tri == best_triangle) {
                let last = neighbors.len() - 1;
                neighbors[i] = neighbors[last];
                adjacency.counts[index] -= 1;
            }
        }

        // update aggregated meshlet cone data for scoring subsequent triangles
        let t = &triangles[best_triangle as usize];
        meshlet_cone_acc.px += t.px;
        meshlet_cone_acc.py += t.py;
        meshlet_cone_acc.pz += t.pz;
        meshlet_cone_acc.nx += t.nx;
        meshlet_cone_acc.ny += t.ny;
        meshlet_cone_acc.nz += t.nz;

        emitted_flags[best_triangle as usize] = true;
    }

    flush_meshlet(&meshlet, &mut out, &mut meshlet_offset)?;

    Ok(meshlet_offset)
}

/// Meshlet builder.
/// Splits the mesh into a set of meshlets where each meshlet has a micro index buffer indexing into meshlet vertices that refer to the original vertex buffer.
/// `meshlets` must hold [`build_meshlets_bound`] entries; `meshlet_vertices` and `meshlet_triangles` must hold `index_count` entries.
/// `cone_weight` should be set to 0 when cone culling is not used, and a value between 0 and 1 otherwise.
pub fn build_meshlets<T: IndexType>(
    meshlets: &mut [Meshlet],
    meshlet_vertices: &mut [u32],
    meshlet_triangles: &mut [u8],
    indices: &[T],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    max_vertices: usize,
    max_triangles: usize,
    cone_weight: f32,
) -> Result<usize> {
    build_meshlets_u32(
        meshlets,
        meshlet_vertices,
        meshlet_triangles,
        &index::input(indices),
        index_count,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        max_vertices,
        max_triangles,
        cone_weight,
    )
}

fn build_meshlets_u32(
    meshlets: &mut [Meshlet],
    meshlet_vertices: &mut [u32],
    meshlet_triangles: &mut [u8],
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    max_vertices: usize,
    max_triangles: usize,
    cone_weight: f32,
) -> Result<usize> {
    build_meshlets_flex(
        meshlets,
        meshlet_vertices,
        meshlet_triangles,
        indices,
        index_count,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        max_vertices,
        max_triangles,
        max_triangles,
        cone_weight,
        0.0,
    )
}

/// Meshlet builder that scans the index buffer in order; for maximum efficiency the index buffer has to be optimized for vertex cache first.
pub fn build_meshlets_scan<T: IndexType>(
    meshlets: &mut [Meshlet],
    meshlet_vertices: &mut [u32],
    meshlet_triangles: &mut [u8],
    indices: &[T],
    index_count: usize,
    vertex_count: usize,
    max_vertices: usize,
    max_triangles: usize,
) -> Result<usize> {
    build_meshlets_scan_u32(
        meshlets,
        meshlet_vertices,
        meshlet_triangles,
        &index::input(indices),
        index_count,
        vertex_count,
        max_vertices,
        max_triangles,
    )
}

fn build_meshlets_scan_u32(
    meshlets: &mut [Meshlet],
    meshlet_vertices: &mut [u32],
    meshlet_triangles: &mut [u8],
    indices: &[u32],
    index_count: usize,
    vertex_count: usize,
    max_vertices: usize,
    max_triangles: usize,
) -> Result<usize> {
    validate::triangles(index_count)?;
    let mut out = validate_meshlet_output(
        meshlets,
        meshlet_vertices,
        meshlet_triangles,
        index_count,
        max_vertices,
        max_triangles,
        max_triangles,
    )?;
    let indices = validate::indices(indices, index_count, vertex_count)?;

    // index of the vertex in the meshlet, -1 if the vertex isn't used
    let mut used = vec![-1i16; vertex_count];

    let mut meshlet = Meshlet::default();
    let mut meshlet_offset = 0;

    for &[a, b, c] in indices.as_chunks::<3>().0 {
        // appends triangle to the meshlet and writes previous meshlet to the output if full
        meshlet_offset += usize::from(append_meshlet(
            &mut meshlet,
            a,
            b,
            c,
            &mut used,
            &mut out,
            meshlet_offset,
            max_vertices,
            max_triangles,
            false,
        )?);
    }

    flush_meshlet(&meshlet, &mut out, &mut meshlet_offset)?;

    Ok(meshlet_offset)
}

/// Meshlet builder that produces clusters optimized for raytracing.
/// Splits the mesh into a set of meshlets, similarly to [`build_meshlets`], but optimizes cluster subdivision for raytracing and allows to specify minimum and maximum number of triangles per meshlet.
/// `meshlets` must hold [`build_meshlets_bound`] (using `min_triangles`) entries. `fill_weight` of 0.5 is a safe default.
pub fn build_meshlets_spatial<T: IndexType>(
    meshlets: &mut [Meshlet],
    meshlet_vertices: &mut [u32],
    meshlet_triangles: &mut [u8],
    indices: &[T],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    max_vertices: usize,
    min_triangles: usize,
    max_triangles: usize,
    fill_weight: f32,
) -> Result<usize> {
    build_meshlets_spatial_u32(
        meshlets,
        meshlet_vertices,
        meshlet_triangles,
        &index::input(indices),
        index_count,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        max_vertices,
        min_triangles,
        max_triangles,
        fill_weight,
    )
}

fn build_meshlets_spatial_u32(
    meshlets: &mut [Meshlet],
    meshlet_vertices: &mut [u32],
    meshlet_triangles: &mut [u8],
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    max_vertices: usize,
    min_triangles: usize,
    max_triangles: usize,
    fill_weight: f32,
) -> Result<usize> {
    validate::triangles(index_count)?;
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    ensure(fill_weight >= 0.0, Error::InvalidArgument("fill_weight must be >= 0"))?;
    let mut out = validate_meshlet_output(
        meshlets,
        meshlet_vertices,
        meshlet_triangles,
        index_count,
        max_vertices,
        min_triangles,
        max_triangles,
    )?;
    let indices = validate::indices(indices, index_count, vertex_count)?;

    if index_count == 0 {
        return Ok(0);
    }

    let face_count = index_count / 3;
    let vertex_stride_float = vertex_positions_stride / 4;

    // compute bounding boxes and centroids for sorting
    let mut boxes = vec![
        BvhBox {
            min: [0.0; 3],
            max: [0.0; 3]
        };
        face_count
    ];
    let mut centroids = vec![0.0f32; face_count * 3];
    bvh_prepare(&mut boxes, &mut centroids, indices, vertex_positions, vertex_stride_float);

    let mut axes = vec![0u32; face_count * 3];
    let mut temp = vec![0u32; face_count];

    for (k, order) in axes.chunks_exact_mut(face_count).enumerate() {
        let keys = &centroids[k * face_count..(k + 1) * face_count];

        let mut hist = compute_histogram(keys);

        // 3-pass radix sort computes the resulting order into axes
        for (i, t) in temp.iter_mut().enumerate() {
            *t = i as u32;
        }

        radix_pass(order, &temp, keys, &mut hist, 0);
        radix_pass(&mut temp, order, keys, &mut hist, 1);
        radix_pass(order, &temp, keys, &mut hist, 2);
    }

    // index of the vertex in the meshlet, -1 if the vertex isn't used
    let mut used = vec![-1i16; vertex_count];

    let mut boundary = vec![0u8; face_count];

    let mut scratch = BvhScratch {
        areas: vec![0.0; face_count * 2],
        vertices: vec![0; face_count],
        temp,
        sides: vec![0; face_count],
    };
    let params = BvhParams {
        boxes: &boxes,
        indices,
        max_vertices,
        min_triangles,
        max_triangles,
        fill_weight,
    };

    {
        let (ax, rest) = axes.split_at_mut(face_count);
        let (ay, az) = rest.split_at_mut(face_count);
        bvh_split(&params, [ax, ay, az], &mut boundary, 0, &mut scratch, &mut used);
    }

    // compute the desired number of meshlets; note that on some meshes with a lot of vertex bound clusters this might go over the bound
    let meshlet_count: usize = boundary.iter().map(|&b| usize::from(b)).sum();

    let meshlet_bound = build_meshlets_bound(index_count, max_vertices, min_triangles)?;

    // pack triangles into meshlets according to the order and boundaries marked by bvhSplit
    let mut meshlet = Meshlet::default();
    let mut meshlet_offset = 0;
    let mut meshlet_pending = meshlet_count;

    for i in 0..face_count {
        let mut split = i > 0 && boundary[i] == 1;

        // while we are over the limit, we ignore boundary[] data and disable splits until we free up enough space
        if split && meshlet_count > meshlet_bound && meshlet_offset + meshlet_pending >= meshlet_bound {
            split = false;
        }

        let [a, b, c] = triangle(indices, axes[i]);

        // appends triangle to the meshlet and writes previous meshlet to the output if full
        meshlet_offset += usize::from(append_meshlet(
            &mut meshlet,
            a as u32,
            b as u32,
            c as u32,
            &mut used,
            &mut out,
            meshlet_offset,
            max_vertices,
            max_triangles,
            split,
        )?);
        meshlet_pending -= usize::from(boundary[i]);
    }

    flush_meshlet(&meshlet, &mut out, &mut meshlet_offset)?;

    Ok(meshlet_offset)
}
