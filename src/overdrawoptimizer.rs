// This work is based on:
// Pedro Sander, Diego Nehab and Joshua Barczak. Fast Triangle Reordering for Vertex Locality and Reduced Overdraw. 2007

use crate::error::{Error, Result};
use crate::quantization::quantize_unorm;
use crate::validate;

fn calculate_sort_data(
    sort_data: &mut [f32],
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    clusters: &[u32],
) {
    let vertex_stride_float = vertex_positions_stride / 4;
    let index_count = indices.len();
    let cluster_count = clusters.len();

    let mut mesh_centroid = [0.0f32; 3];

    for i in 0..vertex_count {
        let p = &vertex_positions[vertex_stride_float * i..];

        mesh_centroid[0] += p[0];
        mesh_centroid[1] += p[1];
        mesh_centroid[2] += p[2];
    }

    mesh_centroid[0] /= vertex_count as f32;
    mesh_centroid[1] /= vertex_count as f32;
    mesh_centroid[2] /= vertex_count as f32;

    for cluster in 0..cluster_count {
        let cluster_begin = clusters[cluster] as usize * 3;
        let cluster_end = if cluster + 1 < cluster_count {
            clusters[cluster + 1] as usize * 3
        } else {
            index_count
        };

        let mut cluster_area = 0.0f32;
        let mut cluster_centroid = [0.0f32; 3];
        let mut cluster_normal = [0.0f32; 3];

        for tri in indices[cluster_begin..cluster_end].as_chunks::<3>().0 {
            let p0 = &vertex_positions[vertex_stride_float * tri[0] as usize..];
            let p1 = &vertex_positions[vertex_stride_float * tri[1] as usize..];
            let p2 = &vertex_positions[vertex_stride_float * tri[2] as usize..];

            let p10 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let p20 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

            let normalx = p10[1] * p20[2] - p10[2] * p20[1];
            let normaly = p10[2] * p20[0] - p10[0] * p20[2];
            let normalz = p10[0] * p20[1] - p10[1] * p20[0];

            let area = (normalx * normalx + normaly * normaly + normalz * normalz).sqrt();

            cluster_centroid[0] += (p0[0] + p1[0] + p2[0]) * (area / 3.0);
            cluster_centroid[1] += (p0[1] + p1[1] + p2[1]) * (area / 3.0);
            cluster_centroid[2] += (p0[2] + p1[2] + p2[2]) * (area / 3.0);
            cluster_normal[0] += normalx;
            cluster_normal[1] += normaly;
            cluster_normal[2] += normalz;
            cluster_area += area;
        }

        let inv_cluster_area = if cluster_area == 0.0 { 0.0 } else { 1.0 / cluster_area };

        cluster_centroid[0] *= inv_cluster_area;
        cluster_centroid[1] *= inv_cluster_area;
        cluster_centroid[2] *= inv_cluster_area;

        let cluster_normal_length =
            (cluster_normal[0] * cluster_normal[0] + cluster_normal[1] * cluster_normal[1] + cluster_normal[2] * cluster_normal[2]).sqrt();
        let inv_cluster_normal_length = if cluster_normal_length == 0.0 {
            0.0
        } else {
            1.0 / cluster_normal_length
        };

        cluster_normal[0] *= inv_cluster_normal_length;
        cluster_normal[1] *= inv_cluster_normal_length;
        cluster_normal[2] *= inv_cluster_normal_length;

        let centroid_vector = [
            cluster_centroid[0] - mesh_centroid[0],
            cluster_centroid[1] - mesh_centroid[1],
            cluster_centroid[2] - mesh_centroid[2],
        ];

        sort_data[cluster] =
            centroid_vector[0] * cluster_normal[0] + centroid_vector[1] * cluster_normal[1] + centroid_vector[2] * cluster_normal[2];
    }
}

fn calculate_sort_order_radix(sort_order: &mut [u32], sort_data: &[f32], sort_keys: &mut [u16]) {
    // compute sort data bounds and renormalize, using fixed point snorm
    let mut sort_data_max = 1e-3f32;

    for &d in sort_data {
        let dpa = d.abs();

        sort_data_max = if sort_data_max < dpa { dpa } else { sort_data_max };
    }

    const SORT_BITS: i32 = 11;

    for (key, &d) in sort_keys.iter_mut().zip(sort_data) {
        // note that we flip distribution since high dot product should come first
        let sort_key = 0.5 - 0.5 * (d / sort_data_max);

        *key = (quantize_unorm(sort_key, SORT_BITS) & ((1 << SORT_BITS) - 1)) as u16;
    }

    // fill histogram for counting sort
    let mut histogram = [0u32; 1 << SORT_BITS];

    for &key in sort_keys.iter() {
        histogram[key as usize] += 1;
    }

    // compute offsets based on histogram data
    let mut histogram_sum = 0u32;

    for h in histogram.iter_mut() {
        let count = *h;
        *h = histogram_sum;
        histogram_sum += count;
    }

    // compute sort order based on offsets
    for (i, &key) in sort_keys.iter().enumerate() {
        sort_order[histogram[key as usize] as usize] = i as u32;
        histogram[key as usize] += 1;
    }
}

fn update_cache(tri: &[u32; 3], cache_size: u32, cache_timestamps: &mut [u32], timestamp: &mut u32) -> u32 {
    let mut cache_misses = 0;

    // if vertex is not in cache, put it in cache
    for &v in tri {
        if timestamp.wrapping_sub(cache_timestamps[v as usize]) > cache_size {
            cache_timestamps[v as usize] = *timestamp;
            *timestamp = timestamp.wrapping_add(1);
            cache_misses += 1;
        }
    }

    cache_misses
}

fn generate_hard_boundaries(destination: &mut [u32], indices: &[u32], cache_size: u32, cache_timestamps: &mut [u32]) -> usize {
    cache_timestamps.fill(0);

    let mut timestamp = cache_size + 1;

    let mut result = 0;

    for (i, tri) in indices.as_chunks::<3>().0.iter().enumerate() {
        let m = update_cache(tri, cache_size, cache_timestamps, &mut timestamp);

        // when all three vertices are not in the cache it's usually relatively safe to assume that this is a new patch in the mesh
        // that is disjoint from previous vertices; sometimes it might come back to reference existing vertices but that frequently
        // suggests an inefficiency in the vertex cache optimization algorithm
        // usually the first triangle has 3 misses unless it's degenerate - thus we make sure the first cluster always starts with 0
        if i == 0 || m == 3 {
            destination[result] = i as u32;
            result += 1;
        }
    }

    result
}

fn generate_soft_boundaries(
    destination: &mut [u32],
    indices: &[u32],
    clusters: &[u32],
    cache_size: u32,
    threshold: f32,
    cache_timestamps: &mut [u32],
) -> usize {
    cache_timestamps.fill(0);

    let triangles = indices.as_chunks::<3>().0;
    let cluster_count = clusters.len();

    let mut timestamp = 0u32;

    let mut result = 0;

    for it in 0..cluster_count {
        let start = clusters[it] as usize;
        let end = if it + 1 < cluster_count {
            clusters[it + 1] as usize
        } else {
            triangles.len()
        };

        // reset cache
        timestamp = timestamp.wrapping_add(cache_size + 1);

        // measure cluster ACMR
        let mut cluster_misses = 0u32;

        for tri in &triangles[start..end] {
            cluster_misses += update_cache(tri, cache_size, cache_timestamps, &mut timestamp);
        }

        let cluster_threshold = threshold * (cluster_misses as f32 / (end - start) as f32);

        // first cluster always starts from the hard cluster boundary
        destination[result] = start as u32;
        result += 1;

        // reset cache
        timestamp = timestamp.wrapping_add(cache_size + 1);

        let mut running_misses = 0u32;
        let mut running_faces = 0u32;

        for (i, tri) in triangles.iter().enumerate().take(end).skip(start) {
            let m = update_cache(tri, cache_size, cache_timestamps, &mut timestamp);

            running_misses += m;
            running_faces += 1;

            if running_misses as f32 / running_faces as f32 <= cluster_threshold {
                // we have reached the target ACMR with the current triangle so we need to start a new cluster on the next one
                // note that this may mean that we add 'end` to destination for the last triangle, which will imply that the last
                // cluster is empty; however, the 'pop_back' after the loop will clean it up
                destination[result] = (i + 1) as u32;
                result += 1;

                // reset cache
                timestamp = timestamp.wrapping_add(cache_size + 1);

                running_misses = 0;
                running_faces = 0;
            }
        }

        // each time we reach the target ACMR we flush the cluster
        // this means that the last cluster is by definition not very good - there are frequent cases where we are left with a few triangles
        // in the last cluster, producing a very bad ACMR and significantly penalizing the overall results
        // thus we remove the last cluster boundary, merging the last complete cluster with the last incomplete one
        // there are sometimes cases when the last cluster is actually good enough - in which case the code above would have added 'end'
        // to the cluster boundary array which we need to remove anyway - this code will do that automatically
        if destination[result - 1] as usize != start {
            result -= 1;
        }
    }

    result
}

/// Overdraw optimizer.
/// Reorders indices to reduce the number of GPU vertex shader invocations and the pixel overdraw.
/// `indices` must have been optimized for vertex cache. `threshold` is the allowed ACMR degradation (1.05 = up to 5% worse).
/// See [`optimize_overdraw_in_place`] for C in-place mode.
pub fn optimize_overdraw(
    destination: &mut [u32],
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    threshold: f32,
) -> Result<()> {
    validate::triangles(index_count)?;
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;

    // guard for empty meshes
    if index_count == 0 || vertex_count == 0 {
        return Ok(());
    }

    let indices = validate::indices(indices, index_count, vertex_count)?;
    let destination = validate::output(destination, index_count)?;

    let cache_size = 16u32;

    let mut cache_timestamps = vec![0u32; vertex_count];

    // generate hard boundaries from full-triangle cache misses
    let mut hard_clusters = vec![0u32; index_count / 3];
    let hard_cluster_count = generate_hard_boundaries(&mut hard_clusters, indices, cache_size, &mut cache_timestamps);

    // generate soft boundaries
    let mut soft_clusters = vec![0u32; index_count / 3 + 1];
    let soft_cluster_count = generate_soft_boundaries(
        &mut soft_clusters,
        indices,
        &hard_clusters[..hard_cluster_count],
        cache_size,
        threshold,
        &mut cache_timestamps,
    );

    let clusters = &soft_clusters[..soft_cluster_count];
    let cluster_count = soft_cluster_count;

    // fill sort data
    let mut sort_data = vec![0.0f32; cluster_count];
    calculate_sort_data(
        &mut sort_data,
        indices,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        clusters,
    );

    // sort clusters using sort data
    let mut sort_keys = vec![0u16; cluster_count];
    let mut sort_order = vec![0u32; cluster_count];
    calculate_sort_order_radix(&mut sort_order, &sort_data, &mut sort_keys);

    // fill output buffer
    let mut offset = 0;

    for &cluster in &sort_order {
        let cluster = cluster as usize;

        let cluster_begin = clusters[cluster] as usize * 3;
        let cluster_end = if cluster + 1 < cluster_count {
            clusters[cluster + 1] as usize * 3
        } else {
            index_count
        };

        destination[offset..offset + cluster_end - cluster_begin].copy_from_slice(&indices[cluster_begin..cluster_end]);
        offset += cluster_end - cluster_begin;
    }

    Ok(())
}

/// In-place variant of [`optimize_overdraw`] (C: `destination == indices`).
pub fn optimize_overdraw_in_place(
    indices: &mut [u32],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    threshold: f32,
) -> Result<()> {
    let copy = indices.get(..index_count).ok_or(Error::BufferTooSmall("indices"))?.to_vec();
    optimize_overdraw(
        indices,
        &copy,
        index_count,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
        threshold,
    )
}
