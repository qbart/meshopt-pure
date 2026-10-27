// This work is based on:
// Takio Kurita. An efficient agglomerative clustering algorithm using a heap. 1991

use crate::error::{Error, Result, ensure};
use crate::validate;

// To avoid excessive recursion for malformed inputs, we switch to bisection after some depth
const MERGE_DEPTH_CUTOFF: i32 = 40;

struct ClusterAdjacency {
    offsets: Vec<u32>,
    clusters: Vec<u32>,
    shared: Vec<u32>,
}

/// Copies cluster indices skipping duplicates within each cluster; returns the new data and cluster offsets.
fn filter_cluster_indices(cluster_indices: &[u32], cluster_index_counts: &[u32], vertex_count: usize) -> (Vec<u32>, Vec<u32>) {
    let mut data = vec![0u32; cluster_indices.len()];
    let mut offsets = vec![0u32; cluster_index_counts.len() + 1];
    let mut used = vec![false; vertex_count];

    let mut cluster_start = 0;
    let mut cluster_write = 0;

    for (i, &count) in cluster_index_counts.iter().enumerate() {
        offsets[i] = cluster_write as u32;

        // copy cluster indices, skipping duplicates
        for &v in &cluster_indices[cluster_start..cluster_start + count as usize] {
            data[cluster_write] = v;
            cluster_write += usize::from(!used[v as usize]);
            used[v as usize] = true;
        }

        // reset used flags for the next cluster
        for &v in &data[offsets[i] as usize..cluster_write] {
            used[v as usize] = false;
        }

        cluster_start += count as usize;
    }

    offsets[cluster_index_counts.len()] = cluster_write as u32;

    (data, offsets)
}

fn compute_cluster_bounds(indices: &[u32], vertex_positions: &[f32], vertex_positions_stride: usize) -> ([f32; 3], f32) {
    let vertex_stride_float = vertex_positions_stride / 4;

    let mut center = [0.0f32; 3];

    // approximate center of the cluster by averaging all vertex positions
    for &j in indices {
        let p = &vertex_positions[j as usize * vertex_stride_float..];

        center[0] += p[0];
        center[1] += p[1];
        center[2] += p[2];
    }

    // note: technically clusters can't be empty per meshopt_partitionCluster but we check for a division by zero in case that changes
    if !indices.is_empty() {
        center[0] /= indices.len() as f32;
        center[1] /= indices.len() as f32;
        center[2] /= indices.len() as f32;
    }

    // compute radius of the bounding sphere for each cluster
    let mut radiussq = 0.0f32;

    for &j in indices {
        let p = &vertex_positions[j as usize * vertex_stride_float..];

        let d2 =
            (p[0] - center[0]) * (p[0] - center[0]) + (p[1] - center[1]) * (p[1] - center[1]) + (p[2] - center[2]) * (p[2] - center[2]);

        radiussq = if radiussq < d2 { d2 } else { radiussq };
    }

    (center, radiussq.sqrt())
}

fn build_cluster_adjacency(cluster_indices: &[u32], cluster_offsets: &[u32], vertex_count: usize) -> ClusterAdjacency {
    let cluster_count = cluster_offsets.len() - 1;
    let cluster = |i: usize| &cluster_indices[cluster_offsets[i] as usize..cluster_offsets[i + 1] as usize];

    let mut ref_offsets = vec![0u32; vertex_count + 1];

    // compute number of clusters referenced by each vertex
    for i in 0..cluster_count {
        for &v in cluster(i) {
            ref_offsets[v as usize] += 1;
        }
    }

    // compute (worst-case) number of adjacent clusters for each cluster
    let mut total_adjacency = 0;

    for i in 0..cluster_count {
        // worst case is every vertex has a disjoint cluster list
        let count: usize = cluster(i).iter().map(|&v| ref_offsets[v as usize] as usize - 1).sum();

        // ... but only every other cluster can be adjacent in the end
        total_adjacency += count.min(cluster_count - 1);
    }

    // we can now allocate adjacency buffers
    let mut adjacency = ClusterAdjacency {
        offsets: vec![0; cluster_count + 1],
        clusters: vec![0; total_adjacency],
        shared: vec![0; total_adjacency],
    };

    // convert ref counts to offsets
    let mut total_refs = 0u32;

    for r in ref_offsets.iter_mut().take(vertex_count) {
        let count = *r;
        *r = total_refs;
        total_refs += count;
    }

    let mut ref_data = vec![0u32; total_refs as usize];

    // fill cluster refs for each vertex
    for i in 0..cluster_count {
        for &v in cluster(i) {
            ref_data[ref_offsets[v as usize] as usize] = i as u32;
            ref_offsets[v as usize] += 1;
        }
    }

    // after the previous pass, ref_offsets contain the end of the data for each vertex; shift it forward to get the start
    ref_offsets.copy_within(0..vertex_count, 1);
    ref_offsets[0] = 0;

    let mut slots = vec![!0u32; cluster_count];

    // fill cluster adjacency for each cluster...
    adjacency.offsets[0] = 0;

    for i in 0..cluster_count {
        let base = adjacency.offsets[i] as usize;
        let mut count = 0usize;

        for &v in cluster(i) {
            // merge the entire cluster list of each vertex into current list
            for &c in &ref_data[ref_offsets[v as usize] as usize..ref_offsets[v as usize + 1] as usize] {
                if c == i as u32 {
                    continue;
                }

                // if the cluster is already in the list, increment the shared count
                // we do not need to clear slots across clusters due to adj[slot] confirmation check
                let slot = slots[c as usize] as usize;
                if slot < count && adjacency.clusters[base + slot] == c {
                    adjacency.shared[base + slot] += 1;
                } else {
                    slots[c as usize] = count as u32;
                    adjacency.clusters[base + count] = c;
                    adjacency.shared[base + count] = 1;
                    count += 1;
                }
            }
        }

        // mark the end of the adjacency list; the next cluster will start there as well
        adjacency.offsets[i + 1] = adjacency.offsets[i] + count as u32;
    }

    adjacency
}

#[derive(Debug, Clone, Copy, Default)]
struct ClusterGroup {
    group: i32,
    next: i32,
    size: u32, // 0 unless root
    vertices: u32,

    center: [f32; 3],
    radius: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct GroupOrder {
    id: u32,
    order: i32,
}

fn heap_push(heap: &mut [GroupOrder], size: usize, item: GroupOrder) {
    // insert a new element at the end (breaks heap invariant)
    heap[size] = item;

    // bubble up the new element to its correct position
    let mut i = size;
    while i > 0 && heap[i].order < heap[(i - 1) / 2].order {
        let p = (i - 1) / 2;

        heap.swap(i, p);
        i = p;
    }
}

fn heap_pop(heap: &mut [GroupOrder], size: usize) -> GroupOrder {
    let top = heap[0];

    // move the last element to the top (breaks heap invariant)
    let size = size - 1;
    heap[0] = heap[size];

    // bubble down the new top element to its correct position
    let mut i = 0;
    while i * 2 + 1 < size {
        // find the smallest child
        let mut j = i * 2 + 1;
        j += usize::from(j + 1 < size && heap[j + 1].order < heap[j].order);

        // if the parent is already smaller than both children, we're done
        if heap[j].order >= heap[i].order {
            break;
        }

        // otherwise, swap the parent and child and continue
        heap.swap(i, j);
        i = j;
    }

    top
}

fn merge_bounds(target: &mut ClusterGroup, source: &ClusterGroup) {
    let (r1, r2) = (target.radius, source.radius);
    let (dx, dy, dz) = (
        source.center[0] - target.center[0],
        source.center[1] - target.center[1],
        source.center[2] - target.center[2],
    );
    let d = (dx * dx + dy * dy + dz * dz).sqrt();

    if d + r1 < r2 {
        target.center = source.center;
        target.radius = source.radius;
        return;
    }

    if d + r2 > r1 {
        let k = if d > 0.0 { (d + r2 - r1) / (2.0 * d) } else { 0.0 };

        target.center[0] += dx * k;
        target.center[1] += dy * k;
        target.center[2] += dz * k;
        target.radius = (d + r2 + r1) / 2.0;
    }
}

fn bounds_score(target: &ClusterGroup, source: &ClusterGroup) -> f32 {
    let (r1, r2) = (target.radius, source.radius);
    let (dx, dy, dz) = (
        source.center[0] - target.center[0],
        source.center[1] - target.center[1],
        source.center[2] - target.center[2],
    );
    let d = (dx * dx + dy * dy + dz * dz).sqrt();

    let mr = if d + r1 < r2 {
        r2
    } else if d + r2 < r1 {
        r1
    } else {
        (d + r2 + r1) / 2.0
    };

    if mr > 0.0 { r1 / mr } else { 0.0 }
}

fn merge_groups(groups: &mut [ClusterGroup], target: usize, source: usize) {
    let source_group = groups[source];
    merge_bounds(&mut groups[target], &source_group);
}

fn pick_group_to_merge(
    groups: &[ClusterGroup],
    id: usize,
    adjacency: &ClusterAdjacency,
    max_partition_size: usize,
    use_bounds: bool,
    shared_acc: &mut [u32],
    out_shared: &mut u32,
) -> i32 {
    let group_rsqrt = 1.0 / (groups[id].vertices as i32 as f32).sqrt();

    let mut best_group = -1;
    let mut best_score = 0.0f32;
    let mut best_shared = 0;

    let adjacent = |ci: usize| adjacency.offsets[ci] as usize..adjacency.offsets[ci + 1] as usize;

    // compute shared count based on underlying clusters for every adjacent group
    // note: we assume shared_acc only contains zeros on input
    let mut ci = id as i32;
    while ci >= 0 {
        for adj in adjacent(ci as usize) {
            let other = groups[adjacency.clusters[adj] as usize].group;
            if other >= 0 {
                shared_acc[other as usize] += adjacency.shared[adj];
            }
        }
        ci = groups[ci as usize].next;
    }

    // score each adjacent group and pick the best one; resets shared_acc in the process
    let mut ci = id as i32;
    while ci >= 0 {
        for adj in adjacent(ci as usize) {
            let other = groups[adjacency.clusters[adj] as usize].group;
            if other < 0 || shared_acc[other as usize] == 0 {
                continue;
            }

            // note: we reset shared_acc here to make sure the next invocation sees zero inputs; it's important that this is done before any early outs
            let shared = shared_acc[other as usize];
            shared_acc[other as usize] = 0;

            let og = &groups[other as usize];
            if (groups[id].size + og.size) as usize > max_partition_size {
                continue;
            }

            let other_rsqrt = 1.0 / (og.vertices as i32 as f32).sqrt();

            // normalize shared count by the expected boundary of each group (+ keeps scoring symmetric)
            let mut score = shared as i32 as f32 * (group_rsqrt + other_rsqrt);

            // incorporate spatial score to favor merging nearby groups
            if use_bounds {
                score *= 1.0 + 0.4 * bounds_score(&groups[id], og);
            }

            if score > best_score {
                best_group = other;
                best_score = score;
                best_shared = shared;
            }
        }
        ci = groups[ci as usize].next;
    }

    *out_shared = best_shared;
    best_group
}

fn link_groups(groups: &mut [ClusterGroup], head: usize, other: usize) {
    // combine groups by linking them together
    let mut tail = head;
    while groups[tail].next >= 0 {
        tail = groups[tail].next as usize;
    }

    groups[tail].next = other as i32;
}

fn merge_leaf(groups: &mut [ClusterGroup], order: &[u32], target_partition_size: usize, max_partition_size: usize) {
    for &id in order {
        let id = id as usize;
        if groups[id].size == 0 || groups[id].size as usize >= target_partition_size {
            continue;
        }

        let mut best_score = -1.0f32;
        let mut best_group = -1i32;

        for &other in order {
            let other = other as usize;
            if id == other || groups[other].size == 0 {
                continue;
            }

            if (groups[id].size + groups[other].size) as usize > max_partition_size {
                continue;
            }

            // favor merging nearby groups
            let score = bounds_score(&groups[id], &groups[other]);

            if score > best_score {
                best_score = score;
                best_group = other as i32;
            }
        }

        // merge id *into* best_group; that way, we may merge more groups into the same best_group, maximizing the chance of reaching target
        if best_group != -1 {
            let best_group = best_group as usize;

            link_groups(groups, best_group, id);

            // update group sizes; note, we omit vertices update for simplicity as it's not used for spatial merge
            groups[best_group].size += groups[id].size;
            groups[id].size = 0;

            // merge bounding spheres
            merge_groups(groups, best_group, id);
            groups[id].radius = 0.0;
        }
    }
}

fn merge_partition(order: &mut [u32], groups: &[ClusterGroup], axis: usize, pivot: f32) -> usize {
    let mut m = 0;

    // invariant: elements in range [0, m) are < pivot, elements in range [m, i) are >= pivot
    for i in 0..order.len() {
        let v = groups[order[i] as usize].center[axis];

        // swap(m, i) unconditionally
        order.swap(m, i);

        // when v >= pivot, we swap i with m without advancing it, preserving invariants
        m += usize::from(v < pivot);
    }

    m
}

fn merge_spatial(
    groups: &mut [ClusterGroup],
    order: &mut [u32],
    target_partition_size: usize,
    max_partition_size: usize,
    leaf_size: usize,
    depth: i32,
) {
    let count = order.len();
    let total: usize = order.iter().map(|&o| groups[o as usize].size as usize).sum();

    if total <= max_partition_size || count <= leaf_size {
        return merge_leaf(groups, order, target_partition_size, max_partition_size);
    }

    let mut mean = [0.0f32; 3];
    let mut vars = [0.0f32; 3];
    let mut runc = 1.0f32;
    let mut runs = 1.0f32;

    // gather statistics on the points in the subtree using Welford's algorithm
    for &o in order.iter() {
        let point = groups[o as usize].center;

        for k in 0..3 {
            let delta = point[k] - mean[k];
            mean[k] += delta * runs;
            vars[k] += delta * (point[k] - mean[k]);
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
    let mut middle = merge_partition(order, groups, axis, split);

    // enforce balance for degenerate partitions
    // this also ensures recursion depth is bounded on pathological inputs
    if middle <= leaf_size / 2 || count - middle <= leaf_size / 2 || depth >= MERGE_DEPTH_CUTOFF {
        middle = count / 2;
    }

    // recursion depth is logarithmic and bounded due to max depth check above
    let (left, right) = order.split_at_mut(middle);
    merge_spatial(groups, left, target_partition_size, max_partition_size, leaf_size, depth + 1);
    merge_spatial(groups, right, target_partition_size, max_partition_size, leaf_size, depth + 1);
}

/// Cluster partitioner.
/// Partitions clusters into groups of similar size, prioritizing grouping clusters that share vertices or are close to each other.
/// When `vertex_positions` is `None`, only clustering connectivity is considered.
///
/// `destination` must hold `cluster_count` entries; returns the number of partitions.
/// `cluster_indices` holds all cluster indices back to back; `cluster_index_counts` holds the number of indices per cluster.
pub fn partition_clusters(
    destination: &mut [u32],
    cluster_indices: &[u32],
    total_index_count: usize,
    cluster_index_counts: &[u32],
    cluster_count: usize,
    vertex_positions: Option<&[f32]>,
    vertex_count: usize,
    vertex_positions_stride: usize,
    target_partition_size: usize,
) -> Result<usize> {
    if let Some(vp) = vertex_positions {
        validate::floats(vp, vertex_count, vertex_positions_stride, 12)?;
    } else {
        ensure(
            vertex_positions_stride <= 256 && vertex_positions_stride.is_multiple_of(4),
            Error::InvalidArgument("vertex stride out of range"),
        )?;
    }
    ensure(
        target_partition_size > 0,
        Error::InvalidArgument("target_partition_size must be > 0"),
    )?;
    let cluster_indices = validate::indices(cluster_indices, total_index_count, vertex_count)?;
    let cluster_index_counts = cluster_index_counts
        .get(..cluster_count)
        .ok_or(Error::BufferTooSmall("cluster_index_counts"))?;
    ensure(
        cluster_index_counts.iter().map(|&c| c as usize).sum::<usize>() == total_index_count,
        Error::InvalidArgument("cluster_index_counts must sum to total_index_count"),
    )?;
    ensure(
        cluster_index_counts.iter().all(|&c| c > 0),
        Error::InvalidArgument("clusters must not be empty"),
    )?;
    let destination = validate::output(destination, cluster_count)?;

    let max_partition_size = target_partition_size + target_partition_size / 3;

    // make new cluster index list that filters out duplicate indices
    let (cluster_indices, cluster_offsets) = filter_cluster_indices(cluster_indices, cluster_index_counts, vertex_count);

    // build cluster adjacency along with edge weights (shared vertex count)
    let adjacency = build_cluster_adjacency(&cluster_indices, &cluster_offsets, vertex_count);

    let mut groups = vec![ClusterGroup::default(); cluster_count];

    let mut order = vec![GroupOrder::default(); cluster_count];
    let mut pending = 0;

    let mut shared_acc = vec![0u32; cluster_count];

    // create a singleton group for each cluster and order them by priority
    for (i, g) in groups.iter_mut().enumerate() {
        g.group = i as i32;
        g.next = -1;
        g.size = 1;
        g.vertices = cluster_offsets[i + 1] - cluster_offsets[i];

        // compute bounding sphere for each cluster if positions are provided
        if let Some(vp) = vertex_positions {
            let (center, radius) = compute_cluster_bounds(
                &cluster_indices[cluster_offsets[i] as usize..cluster_offsets[i + 1] as usize],
                vp,
                vertex_positions_stride,
            );
            g.center = center;
            g.radius = radius;
        }

        heap_push(
            &mut order,
            pending,
            GroupOrder {
                id: i as u32,
                order: g.vertices as i32,
            },
        );
        pending += 1;
    }

    // iteratively merge the smallest group with the best group
    while pending > 0 {
        let mut top = heap_pop(&mut order, pending);
        pending -= 1;

        let id = top.id as usize;

        // this group was merged into another group earlier
        if groups[id].size == 0 {
            continue;
        }

        // disassociate clusters from the group to prevent them from being merged again; we will re-associate them if the group is reinserted
        let mut i = id as i32;
        while i >= 0 {
            groups[i as usize].group = -1;
            i = groups[i as usize].next;
        }

        // the group is large enough, emit as is
        if groups[id].size as usize >= target_partition_size {
            continue;
        }

        let mut best_shared = 0;
        let best_group = pick_group_to_merge(
            &groups,
            id,
            &adjacency,
            max_partition_size,
            vertex_positions.is_some(),
            &mut shared_acc,
            &mut best_shared,
        );

        // we can't grow the group any more, emit as is
        if best_group == -1 {
            continue;
        }

        let best_group = best_group as usize;

        link_groups(&mut groups, id, best_group);

        // update group sizes; note, the vertex update is a O(1) approximation which avoids recomputing the true size
        groups[id].size += groups[best_group].size;
        groups[id].vertices += groups[best_group].vertices;
        groups[id].vertices = if groups[id].vertices > best_shared {
            groups[id].vertices - best_shared
        } else {
            1
        };

        groups[best_group].size = 0;
        groups[best_group].vertices = 0;

        // merge bounding spheres if bounds are available
        if vertex_positions.is_some() {
            merge_groups(&mut groups, id, best_group);
            groups[best_group].radius = 0.0;
        }

        // re-associate all clusters back to the merged group
        let mut i = id as i32;
        while i >= 0 {
            groups[i as usize].group = id as i32;
            i = groups[i as usize].next;
        }

        top.order = groups[id].vertices as i32;
        heap_push(&mut order, pending, top);
        pending += 1;
    }

    // if vertex positions are provided, we do a final pass to see if we can merge small groups based on spatial locality alone
    if vertex_positions.is_some() {
        let mut merge_order: Vec<u32> = (0..cluster_count as u32).filter(|&i| groups[i as usize].size != 0).collect();

        merge_spatial(&mut groups, &mut merge_order, target_partition_size, max_partition_size, 8, 0);
    }

    // output each remaining group
    let mut next_group = 0u32;

    for i in 0..cluster_count {
        if groups[i].size == 0 {
            continue;
        }

        let mut j = i as i32;
        while j >= 0 {
            destination[j as usize] = next_group;
            j = groups[j as usize].next;
        }

        next_group += 1;
    }

    Ok(next_group as usize)
}
