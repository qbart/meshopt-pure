// This work is based on:
// Tom Forsyth. Linear-Speed Vertex Cache Optimisation. 2006
// Pedro Sander, Diego Nehab and Joshua Barczak. Fast Triangle Reordering for Vertex Locality and Reduced Overdraw. 2007

use crate::error::{Error, Result, ensure};
use crate::index::{self, IndexType};
use crate::validate;

const CACHE_SIZE_MAX: usize = 16;
const VALENCE_MAX: usize = 8;

struct VertexScoreTable {
    cache: [f32; 1 + CACHE_SIZE_MAX],
    live: [f32; 1 + VALENCE_MAX],
}

// Tuned to minimize the ACMR of a GPU that has a cache profile similar to NVidia and AMD
#[rustfmt::skip]
static VERTEX_SCORE_TABLE: VertexScoreTable = VertexScoreTable {
    cache: [0.0, 0.779, 0.791, 0.789, 0.981, 0.843, 0.726, 0.847, 0.882, 0.867, 0.799, 0.642, 0.613, 0.600, 0.568, 0.372, 0.234],
    live: [0.0, 0.995, 0.713, 0.450, 0.404, 0.059, 0.005, 0.147, 0.006],
};

// Tuned to minimize the encoded index buffer size
#[rustfmt::skip]
static VERTEX_SCORE_TABLE_STRIP: VertexScoreTable = VertexScoreTable {
    cache: [0.0, 1.000, 1.000, 1.000, 0.453, 0.561, 0.490, 0.459, 0.179, 0.526, 0.000, 0.227, 0.184, 0.490, 0.112, 0.050, 0.131],
    live: [0.0, 0.956, 0.786, 0.577, 0.558, 0.618, 0.549, 0.499, 0.489],
};

/// Per-vertex triangle lists (C: `TriangleAdjacency`).
pub(crate) struct TriangleAdjacency {
    pub(crate) counts: Vec<u32>,
    pub(crate) offsets: Vec<u32>,
    pub(crate) data: Vec<u32>,
}

/// Indices must be validated by the caller.
pub(crate) fn build_triangle_adjacency(indices: &[u32], vertex_count: usize) -> TriangleAdjacency {
    let index_count = indices.len();

    // fill triangle counts
    let mut counts = vec![0u32; vertex_count];

    for &index in indices {
        counts[index as usize] += 1;
    }

    // fill offset table
    let mut offsets = vec![0u32; vertex_count];
    let mut offset = 0u32;

    for (o, &c) in offsets.iter_mut().zip(&counts) {
        *o = offset;
        offset += c;
    }

    // fill triangle data
    let mut data = vec![0u32; index_count];

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

    TriangleAdjacency { counts, offsets, data }
}

fn get_next_vertex_dead_end(
    dead_end: &[u32],
    dead_end_top: &mut usize,
    input_cursor: &mut u32,
    live_triangles: &[u32],
    vertex_count: usize,
) -> u32 {
    // check dead-end stack
    while *dead_end_top > 0 {
        *dead_end_top -= 1;
        let vertex = dead_end[*dead_end_top];

        if live_triangles[vertex as usize] > 0 {
            return vertex;
        }
    }

    // input order
    while (*input_cursor as usize) < vertex_count {
        if live_triangles[*input_cursor as usize] > 0 {
            return *input_cursor;
        }

        *input_cursor += 1;
    }

    !0u32
}

fn get_next_vertex_neighbor(
    next_candidates: &[u32],
    live_triangles: &[u32],
    cache_timestamps: &[u32],
    timestamp: u32,
    cache_size: u32,
) -> u32 {
    let mut best_candidate = !0u32;
    let mut best_priority = -1i32;

    for &vertex in next_candidates {
        let v = vertex as usize;

        // otherwise we don't need to process it
        if live_triangles[v] > 0 {
            let mut priority = 0i32;

            // will it be in cache after fanning?
            if live_triangles[v]
                .wrapping_mul(2)
                .wrapping_add(timestamp)
                .wrapping_sub(cache_timestamps[v])
                <= cache_size
            {
                priority = timestamp.wrapping_sub(cache_timestamps[v]) as i32; // position in cache
            }

            if priority > best_priority {
                best_candidate = vertex;
                best_priority = priority;
            }
        }
    }

    best_candidate
}

fn vertex_score(table: &VertexScoreTable, cache_position: i32, live_triangles: u32) -> f32 {
    let live_triangles_clamped = (live_triangles as usize).min(VALENCE_MAX);

    table.cache[(1 + cache_position) as usize] + table.live[live_triangles_clamped]
}

fn get_next_triangle_dead_end(input_cursor: &mut u32, emitted_flags: &[bool]) -> u32 {
    // input order
    while (*input_cursor as usize) < emitted_flags.len() {
        if !emitted_flags[*input_cursor as usize] {
            return *input_cursor;
        }

        *input_cursor += 1;
    }

    !0u32
}

fn optimize_vertex_cache_table(
    destination: &mut [u32],
    indices: &[u32],
    index_count: usize,
    vertex_count: usize,
    table: &VertexScoreTable,
) -> Result<()> {
    validate::triangles(index_count)?;

    // guard for empty meshes
    if index_count == 0 || vertex_count == 0 {
        return Ok(());
    }

    let indices = validate::indices(indices, index_count, vertex_count)?;
    let destination = validate::output(destination, index_count)?;

    let cache_size = 16usize;

    let face_count = index_count / 3;

    // build adjacency information
    // live triangle counts; note, we alias adjacency.counts as we remove triangles after emitting them so the counts always match
    let TriangleAdjacency {
        counts: mut live_triangles,
        offsets,
        mut data,
    } = build_triangle_adjacency(indices, vertex_count);

    // emitted flags
    let mut emitted_flags = vec![false; face_count];

    // compute initial vertex scores
    let mut vertex_scores: Vec<f32> = live_triangles.iter().map(|&l| vertex_score(table, -1, l)).collect();

    // compute triangle scores
    let mut triangle_scores: Vec<f32> = indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|&[a, b, c]| vertex_scores[a as usize] + vertex_scores[b as usize] + vertex_scores[c as usize])
        .collect();

    let mut cache = [0u32; CACHE_SIZE_MAX + 4];
    let mut cache_new = [0u32; CACHE_SIZE_MAX + 4];
    let mut cache_count = 0usize;

    let mut current_triangle = 0u32;
    let mut input_cursor = 1u32;

    let mut output_triangle = 0usize;

    while current_triangle != !0u32 && output_triangle < face_count {
        let ct = current_triangle as usize;
        let [a, b, c] = [indices[ct * 3], indices[ct * 3 + 1], indices[ct * 3 + 2]];

        // output indices
        destination[output_triangle * 3..output_triangle * 3 + 3].copy_from_slice(&[a, b, c]);
        output_triangle += 1;

        // update emitted flags
        emitted_flags[ct] = true;
        triangle_scores[ct] = 0.0;

        // new triangle
        let mut cache_write = 0;
        cache_new[0] = a;
        cache_new[1] = b;
        cache_new[2] = c;
        cache_write += 3;

        // old triangles
        for &index in &cache[..cache_count] {
            cache_new[cache_write] = index;
            cache_write += usize::from((index != a) & (index != b) & (index != c));
        }

        core::mem::swap(&mut cache, &mut cache_new);
        cache_count = cache_write.min(cache_size);

        // remove emitted triangle from adjacency data
        // this makes sure that we spend less time traversing these lists on subsequent iterations
        // live triangle counts are updated as a byproduct of these adjustments
        for index in [a, b, c] {
            let offset = offsets[index as usize] as usize;
            let neighbors = &mut data[offset..offset + live_triangles[index as usize] as usize];

            if let Some(i) = neighbors.iter().position(|&tri| tri == current_triangle) {
                let last = neighbors.len() - 1;
                neighbors[i] = neighbors[last];
                live_triangles[index as usize] -= 1;
            }
        }

        let mut best_triangle = !0u32;
        let mut best_score = 0.0f32;

        // update cache positions, vertex scores and triangle scores, and find next best triangle
        for (i, &index) in cache[..cache_write].iter().enumerate() {
            let index = index as usize;

            // no need to update scores if we are never going to use this vertex
            if live_triangles[index] == 0 {
                continue;
            }

            let cache_position = if i >= cache_size { -1 } else { i as i32 };

            // update vertex score
            let score = vertex_score(table, cache_position, live_triangles[index]);
            let score_diff = score - vertex_scores[index];

            vertex_scores[index] = score;

            // update scores of vertex triangles
            let offset = offsets[index] as usize;

            for &tri in &data[offset..offset + live_triangles[index] as usize] {
                let tri_score = triangle_scores[tri as usize] + score_diff;

                best_triangle = if best_score < tri_score { tri } else { best_triangle };
                best_score = if best_score < tri_score { tri_score } else { best_score };

                triangle_scores[tri as usize] = tri_score;
            }
        }

        // step through input triangles in order if we hit a dead-end
        current_triangle = best_triangle;

        if current_triangle == !0u32 {
            current_triangle = get_next_triangle_dead_end(&mut input_cursor, &emitted_flags);
        }
    }

    Ok(())
}

/// Vertex transform cache optimizer.
/// Reorders indices to reduce the number of GPU vertex shader invocations.
/// `destination` must hold `index_count` entries; see [`optimize_vertex_cache_in_place`] for C in-place mode.
pub fn optimize_vertex_cache<T: IndexType>(destination: &mut [T], indices: &[T], index_count: usize, vertex_count: usize) -> Result<()> {
    let indices = index::input(indices);
    index::output(destination, |destination| {
        optimize_vertex_cache_u32(destination, &indices, index_count, vertex_count)
    })
}

fn optimize_vertex_cache_u32(destination: &mut [u32], indices: &[u32], index_count: usize, vertex_count: usize) -> Result<()> {
    optimize_vertex_cache_table(destination, indices, index_count, vertex_count, &VERTEX_SCORE_TABLE)
}

/// In-place variant of [`optimize_vertex_cache`] (C: `destination == indices`).
pub fn optimize_vertex_cache_in_place<T: IndexType>(indices: &mut [T], index_count: usize, vertex_count: usize) -> Result<()> {
    let copy = indices.get(..index_count).ok_or(Error::BufferTooSmall("indices"))?.to_vec();
    optimize_vertex_cache(indices, &copy, index_count, vertex_count)
}

/// Vertex transform cache optimizer for strip-like caches.
/// Produces inferior results to [`optimize_vertex_cache`] from the GPU vertex cache perspective,
/// but can be used to generate more efficient index buffer compression or triangle strips.
pub fn optimize_vertex_cache_strip<T: IndexType>(
    destination: &mut [T],
    indices: &[T],
    index_count: usize,
    vertex_count: usize,
) -> Result<()> {
    let indices = index::input(indices);
    index::output(destination, |destination| {
        optimize_vertex_cache_strip_u32(destination, &indices, index_count, vertex_count)
    })
}

fn optimize_vertex_cache_strip_u32(destination: &mut [u32], indices: &[u32], index_count: usize, vertex_count: usize) -> Result<()> {
    optimize_vertex_cache_table(destination, indices, index_count, vertex_count, &VERTEX_SCORE_TABLE_STRIP)
}

/// In-place variant of [`optimize_vertex_cache_strip`] (C: `destination == indices`).
pub fn optimize_vertex_cache_strip_in_place<T: IndexType>(indices: &mut [T], index_count: usize, vertex_count: usize) -> Result<()> {
    let copy = indices.get(..index_count).ok_or(Error::BufferTooSmall("indices"))?.to_vec();
    optimize_vertex_cache_strip(indices, &copy, index_count, vertex_count)
}

/// Vertex transform cache optimizer for FIFO caches.
/// Reorders indices to reduce the number of GPU vertex shader invocations; generally takes ~2x less time than [`optimize_vertex_cache`].
/// `cache_size` should be less than the actual GPU cache size to avoid cache thrashing.
pub fn optimize_vertex_cache_fifo<T: IndexType>(
    destination: &mut [T],
    indices: &[T],
    index_count: usize,
    vertex_count: usize,
    cache_size: u32,
) -> Result<()> {
    let indices = index::input(indices);
    index::output(destination, |destination| {
        optimize_vertex_cache_fifo_u32(destination, &indices, index_count, vertex_count, cache_size)
    })
}

fn optimize_vertex_cache_fifo_u32(
    destination: &mut [u32],
    indices: &[u32],
    index_count: usize,
    vertex_count: usize,
    cache_size: u32,
) -> Result<()> {
    validate::triangles(index_count)?;
    ensure(cache_size >= 3, Error::InvalidArgument("cache_size must be >= 3"))?;

    // guard for empty meshes
    if index_count == 0 || vertex_count == 0 {
        return Ok(());
    }

    let indices = validate::indices(indices, index_count, vertex_count)?;
    let destination = validate::output(destination, index_count)?;

    let face_count = index_count / 3;

    // build adjacency information
    let adjacency = build_triangle_adjacency(indices, vertex_count);

    // live triangle counts
    let mut live_triangles = adjacency.counts.clone();

    // cache time stamps
    let mut cache_timestamps = vec![0u32; vertex_count];

    // dead-end stack
    let mut dead_end = vec![0u32; index_count];
    let mut dead_end_top = 0usize;

    // emitted flags
    let mut emitted_flags = vec![false; face_count];

    let mut current_vertex = 0u32;

    let mut timestamp = cache_size + 1;
    let mut input_cursor = 1u32; // vertex to restart from in case of dead-end

    let mut output_triangle = 0usize;

    while current_vertex != !0u32 {
        let next_candidates_begin = dead_end_top;

        // emit all vertex neighbors
        let offset = adjacency.offsets[current_vertex as usize] as usize;
        let count = adjacency.counts[current_vertex as usize] as usize;

        for &triangle in &adjacency.data[offset..offset + count] {
            let t = triangle as usize;

            if !emitted_flags[t] {
                let tri = [indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]];

                // output indices
                destination[output_triangle * 3..output_triangle * 3 + 3].copy_from_slice(&tri);
                output_triangle += 1;

                // update dead-end stack
                dead_end[dead_end_top..dead_end_top + 3].copy_from_slice(&tri);
                dead_end_top += 3;

                for v in tri {
                    // update live triangle counts
                    live_triangles[v as usize] -= 1;

                    // update cache info
                    // if vertex is not in cache, put it in cache
                    if timestamp.wrapping_sub(cache_timestamps[v as usize]) > cache_size {
                        cache_timestamps[v as usize] = timestamp;
                        timestamp = timestamp.wrapping_add(1);
                    }
                }

                // update emitted flags
                emitted_flags[t] = true;
            }
        }

        // next candidates are the ones we pushed to dead-end stack just now
        let next_candidates = &dead_end[next_candidates_begin..dead_end_top];

        // get next vertex
        current_vertex = get_next_vertex_neighbor(next_candidates, &live_triangles, &cache_timestamps, timestamp, cache_size);

        if current_vertex == !0u32 {
            current_vertex = get_next_vertex_dead_end(&dead_end, &mut dead_end_top, &mut input_cursor, &live_triangles, vertex_count);
        }
    }

    Ok(())
}

/// In-place variant of [`optimize_vertex_cache_fifo`] (C: `destination == indices`).
pub fn optimize_vertex_cache_fifo_in_place<T: IndexType>(
    indices: &mut [T],
    index_count: usize,
    vertex_count: usize,
    cache_size: u32,
) -> Result<()> {
    let copy = indices.get(..index_count).ok_or(Error::BufferTooSmall("indices"))?.to_vec();
    optimize_vertex_cache_fifo(indices, &copy, index_count, vertex_count, cache_size)
}
