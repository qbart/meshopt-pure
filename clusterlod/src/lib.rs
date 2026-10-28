//! clusterlod - a small "library"/example built on top of meshoptimizer to generate cluster LOD hierarchies.
//! Pure Rust port of `demo/clusterlod.h`; intended to either be used as is, or as a reference for implementing similar
//! functionality in your engine.
//!
//! For reference, see the original Nanite paper: Brian Karis. Nanite: A Deep Dive. 2021
#![deny(unsafe_code)]

use std::borrow::Cow;

use meshoptimizer::{
    Error, Meshlet, Result, SIMPLIFY_ERROR_ABSOLUTE, SIMPLIFY_ERROR_CLAMPED, SIMPLIFY_PERMISSIVE, SIMPLIFY_PRESERVE_FOLDS,
    SIMPLIFY_REGULARIZE, SIMPLIFY_SPARSE, SIMPLIFY_VERTEX_PROTECT,
};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ClodConfig {
    /// configuration of each cluster; maps to meshopt_buildMeshlets* parameters
    pub max_vertices: usize,
    pub min_triangles: usize,
    pub max_triangles: usize,

    /// partitioning setup; maps to meshopt_partitionClusters parameters (plus optional partition sorting)
    /// note: partition size is the target size, not maximum; actual partitions may be up to 1/3 larger (e.g. target 24 results in maximum 32)
    pub partition_spatial: bool,
    pub partition_sort: bool,
    pub partition_size: usize,

    /// clusterization setup; maps to meshopt_buildMeshletsSpatial / meshopt_buildMeshletsFlex
    pub cluster_spatial: bool,
    pub cluster_fill_weight: f32,
    pub cluster_split_factor: f32,

    /// every level aims to reduce the number of triangles by ratio, and considers clusters that don't reach the threshold stuck
    pub simplify_ratio: f32,
    pub simplify_threshold: f32,

    /// to compute the error of simplified clusters, we use the formula that combines previous accumulated error as follows:
    /// max(previous_error * simplify_error_merge_previous, current_error) + current_error * simplify_error_merge_additive
    pub simplify_error_merge_previous: f32,
    pub simplify_error_merge_additive: f32,

    /// amplify the error of clusters that go through sloppy simplification to account for appearance degradation
    pub simplify_error_factor_sloppy: f32,

    /// limit error by edge length, aiming to remove subpixel triangles even if the attribute error is high
    pub simplify_error_edge_limit: f32,

    /// clamp attribute error to match position error scale and avoid overly conservative LOD selection
    pub simplify_error_clamped: bool,

    /// use permissive simplification instead of regular simplification (make sure to use attribute_protect_mask if this is set!)
    pub simplify_permissive: bool,

    /// use permissive or sloppy simplification but only if regular simplification gets stuck
    pub simplify_fallback_permissive: bool,
    pub simplify_fallback_sloppy: bool,

    /// use regularization during simplification to make triangle density more uniform, at some cost to overall triangle count; recommended for deformable objects
    pub simplify_regularize: bool,

    /// try to preserve fold lines between opposite-facing triangles during simplification, at a small performance cost
    pub simplify_preserve_folds: bool,

    /// dilate open borders for each cluster to try to mitigate area loss; most useful for foliage, should not be enabled everywhere
    /// note: positions are dilated in a working copy that is passed to the output callback; the caller should copy positions
    /// for each cluster in the output callback as they will be overwritten by simplified clusters
    pub simplify_dilate_borders: bool,

    /// should ClodCluster::bounds be computed based on the geometry of each cluster
    pub optimize_bounds: bool,

    /// should ClodCluster::indices be optimized for locality; helps with rasterization performance and ray tracing performance in fast-build modes
    pub optimize_clusters: bool,
    pub optimize_clusters_level: i32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ClodMesh<'a> {
    /// input triangle indices
    pub indices: &'a [u32],
    pub index_count: usize,

    /// total vertex count
    pub vertex_count: usize,

    /// input vertex positions; must be 3 floats per vertex
    pub vertex_positions: &'a [f32],
    pub vertex_positions_stride: usize,

    /// input vertex attributes; used for attribute-aware simplification and permissive simplification
    pub vertex_attributes: Option<&'a [f32]>,
    pub vertex_attributes_stride: usize,

    /// input vertex locks; allows to preserve additional seams (when not using attribute_protect_mask) or lock vertices via SIMPLIFY_VERTEX_* flags
    pub vertex_lock: Option<&'a [u8]>,

    /// attribute weights for attribute-aware simplification; maps to meshopt_simplifyWithAttributes parameters
    pub attribute_weights: Option<&'a [f32]>,
    pub attribute_count: usize,

    /// attribute mask to flag attribute discontinuities for permissive simplification; mask (1<<K) corresponds to attribute K
    pub attribute_protect_mask: u32,
}

/// To compute approximate (perspective) projection error of a cluster in screen space (0..1; multiply by screen height to get pixels):
/// - camera_proj is projection\[1\]\[1\], or cot(fovy/2); camera_znear is *positive* near plane distance
/// - for simplicity, we ignore perspective distortion and use rotationally invariant projection size estimation
/// - return: bounds.error / max(distance(bounds.center, camera_position) - bounds.radius, camera_znear) * (camera_proj * 0.5f)
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ClodBounds {
    /// sphere bounds, in mesh coordinate space
    pub center: [f32; 3],
    pub radius: f32,

    /// combined simplification error, in mesh coordinate space
    pub error: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct ClodCluster<'a> {
    /// index of more refined group (with more triangles) that produced this cluster during simplification, or -1 for original geometry
    pub refined: i32,

    /// cluster bounds; should only be used for culling, as bounds.error is not monotonic across DAG
    pub bounds: ClodBounds,

    /// cluster indices; refer to the original mesh vertex buffer
    pub indices: &'a [u32],
    pub index_count: usize,

    /// cluster vertex count; indices[] has vertex_count unique entries
    pub vertex_count: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ClodGroup {
    /// DAG level the group was generated at
    pub depth: i32,

    /// simplified group bounds (reflects error for clusters with ClodCluster::refined == group id; error is f32::MAX for terminal groups)
    /// cluster should be rendered if:
    /// 1. ClodGroup::simplified for the group it's in is over error threshold
    /// 2. cluster.refined is -1 *or* ClodGroup::simplified for groups\[cluster.refined\].simplified is at or under error threshold
    pub simplified: ClodBounds,
}

/// when using BVH to determine cluster visibility, traverse the tree into nodes while bounds is over error threshold; for leaves,
/// group clusters should be rendered if cluster.refined is -1 *or* ClodGroup::simplified for groups\[cluster.refined\].simplified is at or under error threshold
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ClodNode {
    /// BVH node bounds, reflecting worst case error for all groups in the subtree
    pub bounds: ClodBounds,

    /// leaf node: index of the group this node represents (or -1 for internal nodes)
    pub group: i32,

    /// internal node: children are at nodes\[child_offset..child_offset+child_count)
    pub child_offset: u32,
    pub child_count: u32,
}

#[derive(Clone, Copy)]
struct Cluster {
    vertices: usize,

    // offset into indices vector in the processing loop
    index_offset: usize,
    index_count: usize,

    refined: i32,

    bounds: ClodBounds,
}

// std::max / std::min
fn fmax(a: f32, b: f32) -> f32 {
    if a < b { b } else { a }
}

fn fmin(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}

fn position(positions: &[f32], stride: usize, v: u32) -> [f32; 3] {
    let p = v as usize * stride;
    [positions[p], positions[p + 1], positions[p + 2]]
}

fn bounds_compute(mesh: &ClodMesh, positions: &[f32], indices: &[u32], error: f32) -> Result<ClodBounds> {
    let bounds = meshoptimizer::compute_cluster_bounds(indices, indices.len(), positions, mesh.vertex_count, mesh.vertex_positions_stride)?;

    Ok(ClodBounds {
        center: bounds.center,
        radius: bounds.radius,
        error,
    })
}

fn bounds_merge<'b>(bounds: impl Iterator<Item = &'b ClodBounds>) -> Result<ClodBounds> {
    let mut spheres = Vec::new();

    // merged bounds error must be conservative wrt cluster errors
    let mut error = 0.0f32;
    for b in bounds {
        spheres.extend_from_slice(&[b.center[0], b.center[1], b.center[2], b.radius]);
        error = fmax(error, b.error);
    }

    let merged = meshoptimizer::compute_sphere_bounds(&spheres, spheres.len() / 4, 16, spheres.get(3..), 16)?;

    Ok(ClodBounds {
        center: merged.center,
        radius: merged.radius,
        error,
    })
}

fn clusterize(
    clusters: &mut Vec<Cluster>,
    cluster_indices: &mut Vec<u32>,
    config: &ClodConfig,
    mesh: &ClodMesh,
    positions: &[f32],
    indices: &[u32],
    refined: i32,
) -> Result<()> {
    let index_count = indices.len();
    let max_meshlets = meshoptimizer::build_meshlets_bound(index_count, config.max_vertices, config.min_triangles)?;

    let mut meshlets = vec![Meshlet::default(); max_meshlets];
    let mut meshlet_vertices = vec![0u32; index_count];
    let mut meshlet_triangles = vec![0u8; index_count];

    let meshlet_count = if config.cluster_spatial {
        meshoptimizer::build_meshlets_spatial(
            &mut meshlets,
            &mut meshlet_vertices,
            &mut meshlet_triangles,
            indices,
            index_count,
            positions,
            mesh.vertex_count,
            mesh.vertex_positions_stride,
            config.max_vertices,
            config.min_triangles,
            config.max_triangles,
            config.cluster_fill_weight,
        )?
    } else {
        meshoptimizer::build_meshlets_flex(
            &mut meshlets,
            &mut meshlet_vertices,
            &mut meshlet_triangles,
            indices,
            index_count,
            positions,
            mesh.vertex_count,
            mesh.vertex_positions_stride,
            config.max_vertices,
            config.min_triangles,
            config.max_triangles,
            0.0,
            config.cluster_split_factor,
        )?
    };

    let mut index_offset = cluster_indices.len();
    cluster_indices.resize(index_offset + index_count, 0);

    for meshlet in &meshlets[..meshlet_count] {
        let vertex_offset = meshlet.vertex_offset as usize;
        let vertex_count = meshlet.vertex_count as usize;
        let triangle_offset = meshlet.triangle_offset as usize;
        let triangle_count = meshlet.triangle_count as usize;

        let vertices = &mut meshlet_vertices[vertex_offset..vertex_offset + vertex_count];
        let triangles = &mut meshlet_triangles[triangle_offset..triangle_offset + triangle_count * 3];

        if config.optimize_clusters {
            meshoptimizer::optimize_meshlet_level(vertices, vertex_count, triangles, triangle_count, config.optimize_clusters_level)?;
        }

        // note: we discard meshlet-local indices; they can be recovered by the caller using clod_local_indices
        for (j, &t) in triangles.iter().enumerate() {
            cluster_indices[index_offset + j] = vertices[t as usize];
        }

        clusters.push(Cluster {
            vertices: vertex_count,
            index_offset,
            index_count: triangle_count * 3,
            refined,
            bounds: ClodBounds::default(),
        });

        index_offset += triangle_count * 3;
    }

    Ok(())
}

fn partition(
    clusters: &mut Vec<Cluster>,
    config: &ClodConfig,
    mesh: &ClodMesh,
    positions: &[f32],
    indices: &[u32],
    remap: &[u32],
) -> Result<Vec<u32>> {
    if clusters.len() <= config.partition_size {
        return Ok(vec![0, clusters.len() as u32]);
    }

    // copy remapped cluster index data into a flat array for partitioning
    let mut cluster_counts = vec![0u32; clusters.len()];
    let mut cluster_indices = Vec::with_capacity(indices.len());

    for (i, cluster) in clusters.iter().enumerate() {
        cluster_counts[i] = cluster.index_count as u32;

        for &v in &indices[cluster.index_offset..cluster.index_offset + cluster.index_count] {
            cluster_indices.push(remap[v as usize]);
        }
    }

    // partition clusters into groups; the output is a partition id per cluster
    let mut cluster_part = vec![0u32; clusters.len()];
    let partition_count = meshoptimizer::partition_clusters(
        &mut cluster_part,
        &cluster_indices,
        cluster_indices.len(),
        &cluster_counts,
        cluster_counts.len(),
        if config.partition_spatial { Some(positions) } else { None },
        remap.len(),
        mesh.vertex_positions_stride,
        config.partition_size,
    )?;

    if config.partition_sort {
        // compute partition points for sorting; any representative point will do, we use last cluster center for simplicity
        let mut partition_point = vec![0.0f32; partition_count * 3];
        for (i, cluster) in clusters.iter().enumerate() {
            let p = cluster_part[i] as usize * 3;
            partition_point[p..p + 3].copy_from_slice(&cluster.bounds.center);
        }

        // sort partitions spatially; the output is a remap table from old index (partition id) to new index
        let mut partition_remap = vec![0u32; partition_count];
        meshoptimizer::spatial_sort_remap(&mut partition_remap, &partition_point, partition_count, 12)?;

        for part in cluster_part.iter_mut() {
            *part = partition_remap[*part as usize];
        }
    }

    // count clusters in each partition and compute offsets via prefix sum
    let mut partition_offsets = vec![0u32; partition_count + 1];
    for &part in &cluster_part {
        partition_offsets[part as usize + 1] += 1;
    }

    for i in 1..=partition_count {
        partition_offsets[i] += partition_offsets[i - 1];
    }

    // distribute clusters into contiguous partitions
    let mut write = partition_offsets.clone();
    let mut result = clusters.clone();
    for (i, cluster) in clusters.iter().enumerate() {
        let w = &mut write[cluster_part[i] as usize];
        result[*w as usize] = *cluster;
        *w += 1;
    }

    *clusters = result;
    Ok(partition_offsets)
}

fn lock_boundary(
    locks: &mut [u8],
    clusters: &[Cluster],
    group_offsets: &[u32],
    cluster_indices: &[u32],
    remap: &[u32],
    vertex_lock: Option<&[u8]>,
) {
    // for each remapped vertex, use bit 7 as temporary storage to indicate that the vertex has been used by a different group previously
    for lock in locks.iter_mut() {
        *lock &= !((1 << 0) | (1 << 7));
    }

    for offsets in group_offsets.windows(2) {
        let group = &clusters[offsets[0] as usize..offsets[1] as usize];

        // mark all remapped vertices as locked if seen by a prior group
        for cluster in group {
            for &v in &cluster_indices[cluster.index_offset..cluster.index_offset + cluster.index_count] {
                let r = remap[v as usize] as usize;

                locks[r] |= locks[r] >> 7;
            }
        }

        // mark all remapped vertices as seen
        for cluster in group {
            for &v in &cluster_indices[cluster.index_offset..cluster.index_offset + cluster.index_count] {
                let r = remap[v as usize] as usize;

                locks[r] |= 1 << 7;
            }
        }
    }

    for i in 0..locks.len() {
        let r = remap[i] as usize;

        // consistently lock all vertices with the same position; keep protect bit if set
        locks[i] = (locks[r] & 1) | (locks[i] & SIMPLIFY_VERTEX_PROTECT);

        if let Some(vertex_lock) = vertex_lock {
            locks[i] |= vertex_lock[i];
        }
    }
}

fn simplify_fallback(
    lod: &mut Vec<u32>,
    mesh: &ClodMesh,
    positions: &[f32],
    indices: &[u32],
    locks: &[u8],
    target_count: usize,
    error: &mut f32,
) -> Result<()> {
    let positions_stride = mesh.vertex_positions_stride / 4;

    let mut subset = Vec::with_capacity(indices.len() * 3);
    let mut subset_locks = Vec::with_capacity(indices.len());
    let subset_indices: Vec<u32> = (0..indices.len() as u32).collect();

    // deindex the mesh subset to avoid calling simplify_sloppy on the entire vertex buffer (which is prohibitively expensive without sparsity)
    for &v in indices {
        subset.extend_from_slice(&position(positions, positions_stride, v));
        subset_locks.push(locks[v as usize]);
    }

    lod.resize(indices.len(), 0);
    let count = meshoptimizer::simplify_sloppy(
        lod.as_mut_slice(),
        &subset_indices,
        indices.len(),
        &subset,
        indices.len(),
        12,
        Some(&subset_locks),
        target_count,
        f32::MAX,
        Some(&mut *error),
    )?;
    lod.truncate(count);

    // convert error to absolute
    *error *= meshoptimizer::simplify_scale(&subset, indices.len(), 12)?;

    // restore original vertex indices
    for v in lod.iter_mut() {
        *v = indices[*v as usize];
    }

    Ok(())
}

fn simplify(
    lod: &mut Vec<u32>,
    config: &ClodConfig,
    mesh: &ClodMesh,
    positions: &[f32],
    indices: &[u32],
    locks: &[u8],
    target_count: usize,
    error: &mut f32,
) -> Result<()> {
    let mut options = SIMPLIFY_SPARSE | SIMPLIFY_ERROR_ABSOLUTE;

    if config.simplify_error_clamped {
        options |= SIMPLIFY_ERROR_CLAMPED;
    }
    if config.simplify_permissive {
        options |= SIMPLIFY_PERMISSIVE;
    }
    if config.simplify_regularize {
        options |= SIMPLIFY_REGULARIZE;
    }
    if config.simplify_preserve_folds {
        options |= SIMPLIFY_PRESERVE_FOLDS;
    }

    let attributes = mesh.vertex_attributes.unwrap_or(&[]);
    let weights = mesh.attribute_weights.unwrap_or(&[]);

    let simplify_with = |lod: &mut Vec<u32>, error: &mut f32, options: u32| -> Result<()> {
        lod.resize(indices.len(), 0);
        let count = meshoptimizer::simplify_with_attributes(
            lod.as_mut_slice(),
            indices,
            indices.len(),
            positions,
            mesh.vertex_count,
            mesh.vertex_positions_stride,
            attributes,
            mesh.vertex_attributes_stride,
            weights,
            mesh.attribute_count,
            Some(locks),
            target_count,
            f32::MAX,
            options,
            Some(error),
        )?;
        lod.truncate(count);
        Ok(())
    };

    simplify_with(lod, error, options)?;

    if lod.len() > target_count && config.simplify_fallback_permissive && !config.simplify_permissive {
        simplify_with(lod, error, options | SIMPLIFY_PERMISSIVE)?;
    }

    // while it's possible to call simplify_sloppy directly, it doesn't support sparsity or absolute error, so we need to do some extra work
    if lod.len() > target_count && config.simplify_fallback_sloppy {
        simplify_fallback(lod, mesh, positions, indices, locks, target_count, error)?;
        *error *= config.simplify_error_factor_sloppy; // scale error up to account for appearance degradation
    }

    // replace degenerate outputs with the first triangle; this ensures we do not produce empty clusters that break DAG invariants
    if lod.is_empty() && !indices.is_empty() {
        lod.extend_from_slice(&indices[..3]);
    }

    // optionally limit error by edge length, aiming to remove subpixel triangles even if the attribute error is high
    if config.simplify_error_edge_limit > 0.0 {
        let positions_stride = mesh.vertex_positions_stride / 4;
        let mut max_edge_sq = 0.0f32;

        for tri in indices.as_chunks::<3>().0 {
            let va = position(positions, positions_stride, tri[0]);
            let vb = position(positions, positions_stride, tri[1]);
            let vc = position(positions, positions_stride, tri[2]);

            // compute squared edge lengths
            let eab = (va[0] - vb[0]) * (va[0] - vb[0]) + (va[1] - vb[1]) * (va[1] - vb[1]) + (va[2] - vb[2]) * (va[2] - vb[2]);
            let eac = (va[0] - vc[0]) * (va[0] - vc[0]) + (va[1] - vc[1]) * (va[1] - vc[1]) + (va[2] - vc[2]) * (va[2] - vc[2]);
            let ebc = (vb[0] - vc[0]) * (vb[0] - vc[0]) + (vb[1] - vc[1]) * (vb[1] - vc[1]) + (vb[2] - vc[2]) * (vb[2] - vc[2]);

            let emax = fmax(fmax(eab, eac), ebc);
            let emin = fmin(fmin(eab, eac), ebc);

            // we prefer using min edge length to reduce the number of triangles <1px thick, but need some stopgap for thin and long triangles like wires
            max_edge_sq = fmax(max_edge_sq, fmax(emin, emax / 4.0));
        }

        // adjust the error to limit it for dense clusters based on edge lengths
        *error = fmin(*error, max_edge_sq.sqrt() * config.simplify_error_edge_limit);
    }

    Ok(())
}

fn output_group<F>(
    config: &ClodConfig,
    mesh: &ClodMesh,
    positions: &[f32],
    group: &[Cluster],
    cluster_indices: &[u32],
    simplified: ClodBounds,
    depth: i32,
    output: &mut F,
) -> Result<i32>
where
    F: FnMut(ClodGroup, &[ClodCluster<'_>], &[f32]) -> i32,
{
    let mut group_clusters = Vec::with_capacity(group.len());

    for cluster in group {
        let indices = &cluster_indices[cluster.index_offset..cluster.index_offset + cluster.index_count];

        group_clusters.push(ClodCluster {
            refined: cluster.refined,
            bounds: if config.optimize_bounds && cluster.refined != -1 {
                bounds_compute(mesh, positions, indices, cluster.bounds.error)?
            } else {
                cluster.bounds
            },
            indices,
            index_count: cluster.index_count,
            vertex_count: cluster.vertices,
        });
    }

    Ok(output(ClodGroup { depth, simplified }, &group_clusters, positions))
}

fn edge_lookup(table: &[u64], key: u64) -> usize {
    let mask = table.len() - 1;
    let mut h = ((key.wrapping_mul(0x9E3779B97F4A7C15) >> 32) as usize) & mask;

    while table[h] != !0u64 && table[h] != key {
        h = (h + 1) & mask;
    }

    h
}

fn edge_key(a: u32, b: u32) -> u64 {
    (a as u64) << 32 | b as u64
}

fn boundary_area(mesh: &ClodMesh, positions: &[f32], indices: &[u32], locks: &[u8], remap: &[u32], table: &mut [u64]) -> f32 {
    table.fill(!0u64);

    for tri in indices.as_chunks::<3>().0 {
        for e in 0..3 {
            let a = remap[tri[e] as usize];
            let b = remap[tri[if e == 2 { 0 } else { e + 1 }] as usize];
            let id = edge_key(a, b);
            let h = edge_lookup(table, id);
            table[h] = id;
        }
    }

    let positions_stride = mesh.vertex_positions_stride / 4;
    let mut area = 0.0f32;

    for tri in indices.as_chunks::<3>().0 {
        let mut border = false;

        for e in 0..3 {
            let a = remap[tri[e] as usize];
            let b = remap[tri[if e == 2 { 0 } else { e + 1 }] as usize];
            let id = edge_key(b, a);

            // we only track border triangles - at least one edge should be unlocked and open (the reverse edge does not exist)
            border |= (locks[a as usize] & locks[b as usize] & 1) == 0 && table[edge_lookup(table, id)] == !0u64;
        }

        if !border {
            continue;
        }

        let va = position(positions, positions_stride, tri[0]);
        let vb = position(positions, positions_stride, tri[1]);
        let vc = position(positions, positions_stride, tri[2]);

        let ab = [vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]];
        let ac = [vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]];
        let nt = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];

        area += (nt[0] * nt[0] + nt[1] * nt[1] + nt[2] * nt[2]).sqrt() * 0.5;
    }

    area
}

fn dilate_borders(
    mesh: &ClodMesh,
    positions: &mut [f32],
    merged: &[u32],
    simplified: &[u32],
    locks: &[u8],
    remap: &[u32],
    bounds: &mut ClodBounds,
    dilate_offsets: &mut [f32],
    edge_table: &mut Vec<u64>,
) {
    // hash table is sized to be able to track all edges
    let mut table_size = 16;
    while table_size < merged.len() + merged.len() / 2 {
        table_size *= 2;
    }

    edge_table.resize(table_size, 0);

    let old_area = boundary_area(mesh, positions, merged, locks, remap, edge_table);
    if old_area == 0.0 {
        return;
    }

    // limit expansion to cases of moderate area shrinkage; it may be unsafe to dilate otherwise
    let new_area = boundary_area(mesh, positions, simplified, locks, remap, edge_table);
    if new_area >= old_area || new_area < old_area / 4.0 {
        return;
    }

    let positions_stride = mesh.vertex_positions_stride / 4;
    let mut perimeter = 0.0f32;

    for tri in simplified.as_chunks::<3>().0 {
        for e in 0..3 {
            let a = remap[tri[e] as usize] as usize;
            let b = remap[tri[if e == 2 { 0 } else { e + 1 }] as usize] as usize;
            let id = edge_key(b as u32, a as u32);

            // the border check must match the one in boundary_area (we also rely on table being filled by boundary_area to avoid extra work)
            if (locks[a] & locks[b] & 1) != 0 || edge_table[edge_lookup(edge_table, id)] != !0u64 {
                continue;
            }

            let va = position(positions, positions_stride, tri[e]);
            let vb = position(positions, positions_stride, tri[if e == 2 { 0 } else { e + 1 }]);
            let vc = position(positions, positions_stride, tri[(e + 2) % 3]);

            let ev = [vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]];
            let el = (ev[0] * ev[0] + ev[1] * ev[1] + ev[2] * ev[2]).sqrt();

            // compute edge normal by projecting C onto AB
            let cv = [va[0] - vc[0], va[1] - vc[1], va[2] - vc[2]];
            let cp = (cv[0] * ev[0] + cv[1] * ev[1] + cv[2] * ev[2]) / (if el > 0.0 { el * el } else { 1.0 });
            let nv = [cv[0] - ev[0] * cp, cv[1] - ev[1] * cp, cv[2] - ev[2] * cp];
            let nl = (nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]).sqrt();

            // accumulate edge normals weighted by edge length; we will renormalize by perimeter before applying
            let ns = if nl > 0.0 { el / nl } else { 0.0 };

            for v in [a, b] {
                if (locks[v] & 1) == 0 {
                    let n = &mut dilate_offsets[v * 4..v * 4 + 4];
                    n[0] += nv[0] * ns;
                    n[1] += nv[1] * ns;
                    n[2] += nv[2] * ns;
                    n[3] += el;
                }
            }

            perimeter += el;
        }
    }

    let distance = if perimeter > 0.0 { (old_area - new_area) / perimeter } else { 0.0 };

    for &v in simplified {
        let r = remap[v as usize] as usize;
        if locks[r] & 1 != 0 {
            continue;
        }

        let vi = v as usize * positions_stride;
        let vr = r * positions_stride;
        let n = &mut dilate_offsets[r * 4..r * 4 + 4];

        if n[3] > 0.0 {
            // we average the normals and then divide by length twice: once to normalize, and once to extend the offset into a miter
            let nx = n[0] / n[3];
            let ny = n[1] / n[3];
            let nz = n[2] / n[3];
            let nn = nx * nx + ny * ny + nz * nz;
            let ns = distance / (if nn > 0.15 { nn } else { 0.15 });

            positions[vr] += nx * ns;
            positions[vr + 1] += ny * ns;
            positions[vr + 2] += nz * ns;
            n.fill(0.0);

            // expand group bounds to accommodate the offset vertex
            let dx = positions[vr] - bounds.center[0];
            let dy = positions[vr + 1] - bounds.center[1];
            let dz = positions[vr + 2] - bounds.center[2];
            bounds.radius = fmax(bounds.radius, (dx * dx + dy * dy + dz * dz).sqrt());
        }

        // copy dilated positions back to all referencing wedges from the canonical copy updated above; no-op for vertices that weren't dilated
        positions.copy_within(vr..vr + 3, vi);
    }
}

fn merge_nodes(nodes: &[ClodNode], offset: usize, count: usize) -> Result<ClodNode> {
    Ok(ClodNode {
        bounds: bounds_merge(nodes[offset..offset + count].iter().map(|n| &n.bounds))?,
        group: -1,
        child_offset: offset as u32,
        child_count: count as u32,
    })
}

/// Default configuration optimized for rasterization; `max_triangles` must be in 4..=256.
pub fn clod_default_config(max_triangles: usize) -> Result<ClodConfig> {
    if !(4..=256).contains(&max_triangles) {
        return Err(Error::InvalidArgument("max_triangles must be in 4..=256"));
    }

    Ok(ClodConfig {
        max_vertices: max_triangles,
        min_triangles: max_triangles / 3,
        max_triangles,

        partition_spatial: true,
        partition_size: 16,

        cluster_spatial: false,
        cluster_split_factor: 2.0,

        optimize_clusters: true,
        optimize_clusters_level: 1,

        simplify_ratio: 0.5,
        simplify_threshold: 0.85,
        simplify_error_merge_previous: 1.0,
        simplify_error_factor_sloppy: 2.0,
        simplify_error_clamped: true,
        simplify_permissive: true,
        simplify_fallback_permissive: false, // note: by default we run in permissive mode, but it's also possible to disable that and use it only as a fallback
        simplify_fallback_sloppy: true,

        ..ClodConfig::default()
    })
}

/// Default configuration optimized for ray tracing; `max_triangles` must be in 4..=256.
pub fn clod_default_config_rt(max_triangles: usize) -> Result<ClodConfig> {
    let mut config = clod_default_config(max_triangles)?;

    // for ray tracing, we may want smaller clusters when that improves BVH quality further; for maximum ray tracing performance this could be reduced even further
    config.min_triangles = max_triangles / 4;

    // by default, we use larger max_vertices for RT; the vertex count is not important for ray tracing performance, and this helps improve cluster utilization
    config.max_vertices = 256.min(max_triangles * 2);

    config.cluster_spatial = true;
    config.cluster_fill_weight = 0.5;

    Ok(config)
}

fn validate_mesh(mesh: &ClodMesh) -> Result<()> {
    let ensure = |cond: bool, err: Error| if cond { Ok(()) } else { Err(err) };

    ensure(
        mesh.vertex_attributes_stride.is_multiple_of(4),
        Error::InvalidArgument("vertex_attributes_stride must be a multiple of 4"),
    )?;
    ensure(
        mesh.attribute_count * 4 <= mesh.vertex_attributes_stride,
        Error::InvalidArgument("attribute_count must fit in vertex_attributes_stride"),
    )?;

    let max_attributes = mesh.vertex_attributes_stride / 4;
    ensure(
        1u32.checked_shl(max_attributes as u32)
            .is_none_or(|limit| mesh.attribute_protect_mask < limit),
        Error::InvalidArgument("attribute_protect_mask must only refer to attributes within vertex_attributes_stride"),
    )?;

    ensure(
        mesh.vertex_positions_stride >= 12 && mesh.vertex_positions_stride.is_multiple_of(4),
        Error::InvalidArgument("vertex_positions_stride must be >= 12 and a multiple of 4"),
    )?;
    if mesh.vertex_count > 0 {
        ensure(
            mesh.vertex_positions.len() >= (mesh.vertex_count - 1) * (mesh.vertex_positions_stride / 4) + 3,
            Error::BufferTooSmall("vertex_positions"),
        )?;
    }

    let indices = mesh.indices.get(..mesh.index_count).ok_or(Error::BufferTooSmall("indices"))?;
    ensure(indices.iter().all(|&v| (v as usize) < mesh.vertex_count), Error::IndexOutOfRange)?;

    if let Some(lock) = mesh.vertex_lock {
        ensure(lock.len() >= mesh.vertex_count, Error::BufferTooSmall("vertex_lock"))?;
    }

    if mesh.attribute_protect_mask != 0 && mesh.vertex_count > 0 {
        let attributes = mesh.vertex_attributes.unwrap_or(&[]);
        let highest = (32 - mesh.attribute_protect_mask.leading_zeros()) as usize;
        ensure(
            attributes.len() >= (mesh.vertex_count - 1) * max_attributes + highest,
            Error::BufferTooSmall("vertex_attributes"),
        )?;
    }

    Ok(())
}

/// Build cluster LOD hierarchy, calling `output` as new clusters and groups are generated.
/// `output` gets called for each group in sequence with the group, its clusters and the current vertex positions (which
/// differ from `mesh.vertex_positions` only when `simplify_dilate_borders` is set); the returned value gets saved for
/// clusters emitted from this group (ClodCluster::refined). Pass `|_, _, _| -1` to ignore the output.
pub fn clod_build<F>(config: ClodConfig, mesh: ClodMesh, mut output: F) -> Result<()>
where
    F: FnMut(ClodGroup, &[ClodCluster<'_>], &[f32]) -> i32,
{
    validate_mesh(&mesh)?;

    let mut locks = vec![0u8; mesh.vertex_count];

    // for cluster connectivity, we need a position-only remap that maps vertices with the same position to the same index
    let mut remap = vec![0u32; mesh.vertex_count];
    meshoptimizer::generate_position_remap(&mut remap, mesh.vertex_positions, mesh.vertex_count, mesh.vertex_positions_stride)?;

    // set up protect bits on UV seams for permissive mode
    if mesh.attribute_protect_mask != 0 {
        let max_attributes = mesh.vertex_attributes_stride / 4;
        let attributes = mesh.vertex_attributes.unwrap_or(&[]);

        for i in 0..mesh.vertex_count {
            let r = remap[i] as usize; // canonical vertex with the same position

            for j in 0..max_attributes {
                if r != i
                    && (mesh.attribute_protect_mask & (1u32 << j)) != 0
                    && attributes[i * max_attributes + j] != attributes[r * max_attributes + j]
                {
                    locks[i] |= SIMPLIFY_VERTEX_PROTECT;
                }
            }
        }
    }

    // dilation updates positions in place; C++ writes through the const input pointer, we use a working copy
    let mut positions = if config.simplify_dilate_borders {
        Cow::Owned(mesh.vertex_positions.to_vec())
    } else {
        Cow::Borrowed(mesh.vertex_positions)
    };

    let mut dilate_offsets = vec![0.0f32; if config.simplify_dilate_borders { mesh.vertex_count * 4 } else { 0 }];
    let mut dilate_table = Vec::new();

    // initial clusterization splits the original mesh
    let mut cluster_indices = Vec::new();
    let mut clusters = Vec::new();
    clusterize(
        &mut clusters,
        &mut cluster_indices,
        &config,
        &mesh,
        &positions,
        &mesh.indices[..mesh.index_count],
        -1,
    )?;

    // compute initial precise bounds; subsequent bounds will be using group-merged bounds
    for cluster in clusters.iter_mut() {
        cluster.bounds = bounds_compute(
            &mesh,
            &positions,
            &cluster_indices[cluster.index_offset..cluster.index_offset + cluster.index_count],
            0.0,
        )?;
    }

    let mut pending = Vec::with_capacity(clusters.len());
    let mut pending_indices = Vec::new();

    let mut merged = Vec::new();
    let mut simplified = Vec::new();

    let mut depth = 0;

    // merge and simplify clusters until we can't merge anymore
    while clusters.len() > 1 {
        let group_offsets = partition(&mut clusters, &config, &mesh, &positions, &cluster_indices, &remap)?;

        pending.clear();
        pending_indices.clear();

        // mark boundaries between groups with a lock bit to avoid gaps in simplified result
        lock_boundary(&mut locks, &clusters, &group_offsets, &cluster_indices, &remap, mesh.vertex_lock);

        // every group needs to be simplified now
        for offsets in group_offsets.windows(2) {
            let group = &clusters[offsets[0] as usize..offsets[1] as usize];

            merged.clear();
            for cluster in group {
                merged.extend_from_slice(&cluster_indices[cluster.index_offset..cluster.index_offset + cluster.index_count]);
            }

            let target_size = (((merged.len() / 3) as f32 * config.simplify_ratio) as usize) * 3;

            // enforce bounds and error monotonicity
            // note: it is incorrect to use the precise bounds of the merged or simplified mesh, because this may violate monotonicity
            let mut bounds = bounds_merge(group.iter().map(|c| &c.bounds))?;

            let mut error = 0.0f32;
            simplify(
                &mut simplified,
                &config,
                &mesh,
                &positions,
                &merged,
                &locks,
                target_size,
                &mut error,
            )?;
            if simplified.len() as f32 > merged.len() as f32 * config.simplify_threshold {
                bounds.error = f32::MAX; // terminal group, won't simplify further
                output_group(&config, &mesh, &positions, group, &cluster_indices, bounds, depth, &mut output)?;
                continue; // simplification is stuck; abandon the merge
            }

            // enforce error monotonicity (with an optional hierarchical factor to separate transitions more)
            bounds.error = fmax(bounds.error * config.simplify_error_merge_previous, error) + error * config.simplify_error_merge_additive;

            // output the new group with all clusters; the resulting id will be recorded in new clusters as ClodCluster::refined
            let refined = output_group(&config, &mesh, &positions, group, &cluster_indices, bounds, depth, &mut output)?;

            // now that we've output the group with the original clusters, we need to dilate simplified clusters if requested
            if config.simplify_dilate_borders {
                dilate_borders(
                    &mesh,
                    positions.to_mut(),
                    &merged,
                    &simplified,
                    &locks,
                    &remap,
                    &mut bounds,
                    &mut dilate_offsets,
                    &mut dilate_table,
                );
            }

            // enqueue new clusters for further processing
            let cluster_offset = pending.len();
            clusterize(&mut pending, &mut pending_indices, &config, &mesh, &positions, &simplified, refined)?;

            // update cluster group bounds to the group-merged bounds; this ensures that we compute the group bounds for whatever group this cluster will be part of conservatively
            for cluster in &mut pending[cluster_offset..] {
                cluster.bounds = bounds;
            }
        }

        std::mem::swap(&mut clusters, &mut pending);
        std::mem::swap(&mut cluster_indices, &mut pending_indices);
        depth += 1;
    }

    if let Some(cluster) = clusters.first() {
        let mut bounds = cluster.bounds;
        bounds.error = f32::MAX; // terminal group, won't simplify further

        output_group(&config, &mesh, &positions, &clusters, &cluster_indices, bounds, depth, &mut output)?;
    }

    Ok(())
}

/// Extract meshlet-local indices from cluster indices produced by clod_build.
/// Fills triangles[] and vertices[] such that vertices\[triangles\[i\]\] == indices\[i\];
/// returns number of unique vertices (which will be equal to ClodCluster::vertex_count).
pub fn clod_local_indices(vertices: &mut [u32], triangles: &mut [u8], indices: &[u32], index_count: usize) -> Result<usize> {
    meshoptimizer::extract_meshlet_indices(vertices, triangles, indices, index_count)
}

/// Upper bound on the number of nodes clod_build_hierarchy can produce.
/// level_count is the number of levels in the group DAG (max(ClodGroup::depth) + 1); `node_width` must be >= 2.
pub fn clod_build_hierarchy_bound(group_count: usize, node_width: usize, level_count: usize) -> Result<usize> {
    if node_width < 2 {
        return Err(Error::InvalidArgument("node_width must be >= 2"));
    }

    // count nodes for each tree depth; we pad by level_count at each iteration to account for unknown level distribution in the forest
    let mut total = level_count;
    let mut frontier = group_count;
    while frontier > 1 {
        total += frontier + level_count;
        frontier = frontier.div_ceil(node_width);
    }

    Ok(total)
}

/// Build a spatial hierarchy over the groups produced by clod_build for accelerating cluster visibility.
/// level_count is the number of levels in the group DAG (max(ClodGroup::depth) + 1); every level must have a group.
/// The result is a forest with one tree per DAG level:
/// - nodes 0..level_count-1 are the roots of per-level trees, each covering groups with ClodGroup::depth == level
/// - remaining nodes are the tree nodes; each internal node has up to node_width children, and each leaf node refers to a single group
pub fn clod_build_hierarchy(
    nodes: &mut [ClodNode],
    groups: &[ClodGroup],
    group_count: usize,
    node_width: usize,
    level_count: usize,
) -> Result<usize> {
    let bound = clod_build_hierarchy_bound(group_count, node_width, level_count)?;
    let groups = groups.get(..group_count).ok_or(Error::BufferTooSmall("groups"))?;
    let nodes = nodes.get_mut(..bound).ok_or(Error::BufferTooSmall("nodes"))?;

    if !(0..level_count).all(|level| groups.iter().any(|g| g.depth as i64 == level as i64)) {
        return Err(Error::InvalidArgument("every level must have at least one group"));
    }

    // reserve space for per-level roots
    let mut offset = level_count;

    let mut row: Vec<ClodNode> = Vec::with_capacity(group_count);
    let mut order = vec![0u32; group_count];
    let mut centers = Vec::with_capacity(group_count * 3);

    for level in 0..level_count {
        // start each tree hierarchy from the groups of that level as leaves
        row.clear();
        for (i, group) in groups.iter().enumerate() {
            if group.depth as i64 == level as i64 {
                row.push(ClodNode {
                    bounds: group.simplified,
                    group: i as i32,
                    child_offset: 0,
                    child_count: 0,
                });
            }
        }

        // build the tree going up one level at a time, using spatial grouping of centers as a proxy for locality
        while row.len() > 1 {
            let count = row.len();

            centers.clear();
            for node in &row {
                centers.extend_from_slice(&node.bounds.center);
            }
            meshoptimizer::spatial_cluster_points(&mut order, &centers, count, 12, node_width)?;

            for i in 0..count {
                nodes[offset + i] = row[order[i] as usize];
            }

            row.clear();
            for i in (0..count).step_by(node_width) {
                // spatial_cluster_points guarantees that each cluster except for last one is full
                let children = node_width.min(count - i);
                row.push(merge_nodes(nodes, offset + i, children)?);
            }

            offset += count;
        }

        // the root of the current level goes into the fixed section at the beginning
        nodes[level] = row[0];
    }

    Ok(offset)
}
