use crate::error::{Error, Result, ensure};
use crate::index::{self, IndexType};
use crate::validate;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VertexCacheStatistics {
    pub vertices_transformed: u32,
    pub warps_executed: u32,
    /// transformed vertices / triangle count; best case 0.5, worst case 3.0, optimum depends on topology
    pub acmr: f32,
    /// transformed vertices / vertex count; best case 1.0, worst case 6.0, optimum is 1.0 (each vertex is transformed once)
    pub atvr: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VertexFetchStatistics {
    pub bytes_fetched: u32,
    /// fetched bytes / vertex buffer size; best case 1.0 (each byte is fetched once)
    pub overfetch: f32,
}

/// Vertex transform cache analyzer.
/// Returns cache hit statistics using a simplified FIFO model; results may not match actual GPU performance.
pub fn analyze_vertex_cache<T: IndexType>(
    indices: &[T],
    index_count: usize,
    vertex_count: usize,
    cache_size: u32,
    warp_size: u32,
    primgroup_size: u32,
) -> Result<VertexCacheStatistics> {
    analyze_vertex_cache_u32(
        &index::input(indices),
        index_count,
        vertex_count,
        cache_size,
        warp_size,
        primgroup_size,
    )
}

fn analyze_vertex_cache_u32(
    indices: &[u32],
    index_count: usize,
    vertex_count: usize,
    cache_size: u32,
    warp_size: u32,
    primgroup_size: u32,
) -> Result<VertexCacheStatistics> {
    validate::triangles(index_count)?;
    ensure(cache_size >= 3, Error::InvalidArgument("cache_size must be >= 3"))?;
    ensure(
        warp_size == 0 || warp_size >= 3,
        Error::InvalidArgument("warp_size must be 0 or >= 3"),
    )?;
    let indices = validate::indices(indices, index_count, vertex_count)?;

    let mut result = VertexCacheStatistics::default();

    let mut warp_offset = 0u32;
    let mut primgroup_offset = 0u32;

    let mut cache_timestamps = vec![0u32; vertex_count];

    let mut timestamp = cache_size + 1;

    for tri in indices.as_chunks::<3>().0 {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);

        let ac = timestamp.wrapping_sub(cache_timestamps[a]) > cache_size;
        let bc = timestamp.wrapping_sub(cache_timestamps[b]) > cache_size;
        let cc = timestamp.wrapping_sub(cache_timestamps[c]) > cache_size;

        // flush cache if triangle doesn't fit into warp or into the primitive buffer
        if (primgroup_size != 0 && primgroup_offset == primgroup_size)
            || (warp_size != 0 && warp_offset + u32::from(ac) + u32::from(bc) + u32::from(cc) > warp_size)
        {
            result.warps_executed += u32::from(warp_offset > 0);

            warp_offset = 0;
            primgroup_offset = 0;

            // reset cache
            timestamp = timestamp.wrapping_add(cache_size + 1);
        }

        // update cache and add vertices to warp
        for &index in tri {
            let index = index as usize;

            if timestamp.wrapping_sub(cache_timestamps[index]) > cache_size {
                cache_timestamps[index] = timestamp;
                timestamp = timestamp.wrapping_add(1);
                result.vertices_transformed = result.vertices_transformed.wrapping_add(1);
                warp_offset += 1;
            }
        }

        primgroup_offset += 1;
    }

    let unique_vertex_count = cache_timestamps.iter().filter(|&&t| t > 0).count();

    result.warps_executed += u32::from(warp_offset > 0);

    result.acmr = if index_count == 0 {
        0.0
    } else {
        result.vertices_transformed as f32 / (index_count / 3) as f32
    };
    result.atvr = if unique_vertex_count == 0 {
        0.0
    } else {
        result.vertices_transformed as f32 / unique_vertex_count as f32
    };

    Ok(result)
}

/// Vertex fetch cache analyzer.
/// Returns cache hit statistics using a simplified direct mapped model; results may not match actual GPU performance.
pub fn analyze_vertex_fetch<T: IndexType>(
    indices: &[T],
    index_count: usize,
    vertex_count: usize,
    vertex_size: usize,
) -> Result<VertexFetchStatistics> {
    analyze_vertex_fetch_u32(&index::input(indices), index_count, vertex_count, vertex_size)
}

fn analyze_vertex_fetch_u32(indices: &[u32], index_count: usize, vertex_count: usize, vertex_size: usize) -> Result<VertexFetchStatistics> {
    ensure(
        vertex_size > 0 && vertex_size <= 256,
        Error::InvalidArgument("vertex_size must be in 1..=256"),
    )?;
    let indices = validate::indices(indices, index_count, vertex_count)?;

    let mut result = VertexFetchStatistics::default();

    let mut vertex_visited = vec![false; vertex_count];

    const CACHE_LINE: usize = 64;
    const CACHE_SIZE: usize = 128 * 1024;

    // simple direct mapped cache; on typical mesh data this is close to 4-way cache, and this model is a gross approximation anyway
    let mut cache = [0usize; CACHE_SIZE / CACHE_LINE];

    for &index in indices {
        vertex_visited[index as usize] = true;

        let start_address = index as usize * vertex_size;
        let end_address = start_address + vertex_size;

        let start_tag = start_address / CACHE_LINE;
        let end_tag = end_address.div_ceil(CACHE_LINE);

        for tag in start_tag..end_tag {
            let line = tag % cache.len();

            // we store +1 since cache is filled with 0 by default
            result.bytes_fetched = result
                .bytes_fetched
                .wrapping_add(u32::from(cache[line] != tag + 1) * CACHE_LINE as u32);
            cache[line] = tag + 1;
        }
    }

    let unique_vertex_count = vertex_visited.iter().filter(|&&v| v).count();

    result.overfetch = if unique_vertex_count == 0 {
        0.0
    } else {
        result.bytes_fetched as f32 / (unique_vertex_count * vertex_size) as f32
    };

    Ok(result)
}
