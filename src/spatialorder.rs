// This work is based on:
// Fabian Giesen. Decoding Morton codes. 2009

use crate::error::{Error, Result, ensure};
use crate::validate;

// "Insert" two 0 bits after each of the 20 low bits of x
fn part1_by2(mut x: u64) -> u64 {
    x &= 0x000fffff; // x = ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- jihg fedc ba98 7654 3210
    x = (x ^ (x << 32)) & 0x000f00000000ffff; // x = ---- ---- ---- jihg ---- ---- ---- ---- ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x << 16)) & 0x000f0000ff0000ff; // x = ---- ---- ---- jihg ---- ---- ---- ---- fedc ba98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x << 8)) & 0x000f00f00f00f00f; // x = ---- ---- ---- jihg ---- ---- fedc ---- ---- ba98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x << 4)) & 0x00c30c30c30c30c3; // x = ---- ---- ji-- --hg ---- fe-- --dc ---- ba-- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x << 2)) & 0x0249249249249249; // x = ---- --j- -i-- h--g --f- -e-- d--c --b- -a-- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x
}

fn compute_order(result: &mut [u64], vertex_positions_data: &[f32], vertex_count: usize, vertex_positions_stride: usize, morton: bool) {
    let vertex_stride_float = vertex_positions_stride / 4;

    let mut minv = [f32::MAX; 3];
    let mut maxv = [-f32::MAX; 3];

    for i in 0..vertex_count {
        let v = &vertex_positions_data[i * vertex_stride_float..];

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

    // rescale each axis to 16 bits to get 48-bit Morton codes
    let scale = if extent == 0.0 { 0.0 } else { 65535.0 / extent };

    // generate Morton order based on the position inside a unit cube
    for (i, r) in result.iter_mut().enumerate().take(vertex_count) {
        let v = &vertex_positions_data[i * vertex_stride_float..];

        let x = ((v[0] - minv[0]) * scale + 0.5) as i32 as u64;
        let y = ((v[1] - minv[1]) * scale + 0.5) as i32 as u64;
        let z = ((v[2] - minv[2]) * scale + 0.5) as i32 as u64;

        *r = if morton {
            part1_by2(x) | (part1_by2(y) << 1) | (part1_by2(z) << 2)
        } else {
            x | (y << 20) | (z << 40)
        };
    }
}

fn radix_sort10(destination: &mut [u32], source: &[u32], keys: &[u16]) {
    let mut hist = [0u32; 1024];

    // compute histogram (assume keys are 10-bit)
    for &k in keys {
        hist[usize::from(k)] += 1;
    }

    let mut sum = 0u32;

    // replace histogram data with prefix histogram sums in-place
    for h in hist.iter_mut() {
        let count = *h;
        *h = sum;
        sum += count;
    }

    // reorder values
    for &s in source {
        let id = usize::from(keys[s as usize]);

        destination[hist[id] as usize] = s;
        hist[id] += 1;
    }
}

fn compute_histogram(data: &[u16]) -> [[u32; 2]; 256] {
    let mut hist = [[0u32; 2]; 256];

    // compute 2 8-bit histograms in parallel
    for &id in data {
        hist[usize::from(id & 255)][0] += 1;
        hist[usize::from(id >> 8)][1] += 1;
    }

    let (mut sum0, mut sum1) = (0u32, 0u32);

    // replace histogram data with prefix histogram sums in-place
    for h in hist.iter_mut() {
        let (h0, h1) = (h[0], h[1]);

        *h = [sum0, sum1];

        sum0 += h0;
        sum1 += h1;
    }

    hist
}

fn radix_pass(destination: &mut [u32], source: &[u32], keys: &[u16], hist: &mut [[u32; 2]; 256], pass: usize) {
    let bitoff = pass * 8;

    for &s in source {
        let id = usize::from((keys[s as usize] >> bitoff) & 255);

        destination[hist[id][pass] as usize] = s;
        hist[id][pass] += 1;
    }
}

fn partition_points(target: &mut [u32], order: &[u32], sides: &[u8], split: usize) {
    let (mut l, mut r) = (0, split);

    for &o in order {
        let side = usize::from(sides[o as usize]);
        target[if side != 0 { r } else { l }] = o;
        l += 1;
        l -= side;
        r += side;
    }
}

fn split_points(destination: &mut [u32], axes: [&mut [u32]; 3], keys: &[u64], temp: &mut [u32], sides: &mut [u8], cluster_size: usize) {
    let count = axes[0].len();

    if count <= cluster_size {
        destination[..count].copy_from_slice(axes[0]);
        return;
    }

    let mut bestk = 0;
    let mut bestdim = 0u32;

    for (k, axis) in axes.iter().enumerate() {
        let mask = (1u32 << 20) - 1;
        let dim =
            ((keys[axis[count - 1] as usize] >> (k * 20)) as u32 & mask).wrapping_sub((keys[axis[0] as usize] >> (k * 20)) as u32 & mask);

        if dim >= bestdim {
            bestk = k;
            bestdim = dim;
        }
    }

    // split roughly in half, with the left split always being aligned to cluster size
    let split = (count / 2).div_ceil(cluster_size) * cluster_size;

    // mark sides of split for partitioning
    for (i, &v) in axes[bestk].iter().enumerate() {
        sides[v as usize] = u8::from(i >= split);
    }

    // partition all axes into two sides, maintaining order
    let [ax, ay, az] = axes;
    let mut axes = [ax, ay, az];

    for (k, axis) in axes.iter_mut().enumerate() {
        if k == bestk {
            continue;
        }

        temp[..count].copy_from_slice(axis);
        partition_points(axis, &temp[..count], sides, split);
    }

    // recursion depth is logarithmic and bounded as we always split in approximately half
    let [ax, ay, az] = axes;
    let (lx, rx) = ax.split_at_mut(split);
    let (ly, ry) = ay.split_at_mut(split);
    let (lz, rz) = az.split_at_mut(split);
    let (ld, rd) = destination.split_at_mut(split);

    split_points(ld, [lx, ly, lz], keys, temp, sides, cluster_size);
    split_points(rd, [rx, ry, rz], keys, temp, sides, cluster_size);
}

/// Spatial sorter.
/// Generates a remap table that can be used to reorder points for spatial locality.
/// Resulting remap table maps old vertices to new vertices and can be used in [`crate::remap_vertex_buffer`].
/// `destination` must hold `vertex_count` entries.
pub fn spatial_sort_remap(
    destination: &mut [u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> Result<()> {
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    let destination = validate::output(destination, vertex_count)?;

    let mut keys = vec![0u64; vertex_count];
    compute_order(&mut keys, vertex_positions, vertex_count, vertex_positions_stride, true);

    let mut scratch = vec![0u32; vertex_count];
    let mut keyk = vec![0u16; vertex_count];

    for (i, d) in destination.iter_mut().enumerate() {
        *d = i as u32;
    }

    // 5-pass radix sort computes the resulting order into scratch
    for k in 0..5 {
        // copy 10-bit key segments into keyk to reduce cache pressure during radix pass
        for (kk, &key) in keyk.iter_mut().zip(&keys) {
            *kk = ((key >> (k * 10)) & 1023) as u16;
        }

        if k % 2 == 0 {
            radix_sort10(&mut scratch, destination, &keyk);
        } else {
            radix_sort10(destination, &scratch, &keyk);
        }
    }

    // since our remap table is mapping old=>new, we need to reverse it
    for (i, &s) in scratch.iter().enumerate() {
        destination[s as usize] = i as u32;
    }

    Ok(())
}

/// Spatial sorter.
/// Reorders triangles for spatial locality, and generates a new index buffer. The resulting index buffer can be used with other functions like optimize_vertex_cache.
/// `destination` must hold `index_count` entries; see [`spatial_sort_triangles_in_place`] for C in-place mode.
pub fn spatial_sort_triangles(
    destination: &mut [u32],
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> Result<()> {
    validate::triangles(index_count)?;
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    let indices = validate::indices(indices, index_count, vertex_count)?;
    let destination = validate::output(destination, index_count)?;

    let face_count = index_count / 3;
    let vertex_stride_float = vertex_positions_stride / 4;

    let mut centroids = vec![0.0f32; face_count * 3];

    for (c, tri) in centroids.as_chunks_mut::<3>().0.iter_mut().zip(indices.as_chunks::<3>().0) {
        let va = &vertex_positions[tri[0] as usize * vertex_stride_float..];
        let vb = &vertex_positions[tri[1] as usize * vertex_stride_float..];
        let vc = &vertex_positions[tri[2] as usize * vertex_stride_float..];

        c[0] = (va[0] + vb[0] + vc[0]) / 3.0;
        c[1] = (va[1] + vb[1] + vc[1]) / 3.0;
        c[2] = (va[2] + vb[2] + vc[2]) / 3.0;
    }

    let mut remap = vec![0u32; face_count];

    spatial_sort_remap(&mut remap, &centroids, face_count, 12)?;

    for (tri, &r) in indices.as_chunks::<3>().0.iter().zip(&remap) {
        let r = r as usize;
        destination[r * 3..r * 3 + 3].copy_from_slice(tri);
    }

    Ok(())
}

/// In-place variant of [`spatial_sort_triangles`] (C: `destination == indices`).
pub fn spatial_sort_triangles_in_place(
    indices: &mut [u32],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> Result<()> {
    let copy = indices.get(..index_count).ok_or(Error::BufferTooSmall("indices"))?.to_vec();
    spatial_sort_triangles(indices, &copy, index_count, vertex_positions, vertex_count, vertex_positions_stride)
}

/// Spatial clusterizer.
/// Reorders points into clusters optimized for spatial locality, and generates a new index buffer.
/// Ensures the output can be split into `cluster_size` chunks where each chunk has good positional locality.
/// Only the last chunk will be smaller than `cluster_size`. `destination` must hold `vertex_count` entries.
pub fn spatial_cluster_points(
    destination: &mut [u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    cluster_size: usize,
) -> Result<()> {
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    ensure(cluster_size > 0, Error::InvalidArgument("cluster_size must be > 0"))?;
    let destination = validate::output(destination, vertex_count)?;

    let mut keys = vec![0u64; vertex_count];
    compute_order(&mut keys, vertex_positions, vertex_count, vertex_positions_stride, false);

    let mut order = vec![0u32; vertex_count * 3];
    let mut scratch = vec![0u32; vertex_count];
    let mut keyk = vec![0u16; vertex_count];

    for (k, axis) in order.chunks_exact_mut(vertex_count.max(1)).take(3).enumerate() {
        // copy 16-bit key segments into keyk to reduce cache pressure during radix pass
        for (kk, &key) in keyk.iter_mut().zip(&keys) {
            *kk = (key >> (k * 20)) as u16;
        }

        let mut hist = compute_histogram(&keyk);

        for (i, o) in axis.iter_mut().enumerate() {
            *o = i as u32;
        }

        radix_pass(&mut scratch, axis, &keyk, &mut hist, 0);
        radix_pass(axis, &scratch, &keyk, &mut hist, 1);
    }

    let mut sides = vec![0u8; vertex_count];

    let (ox, rest) = order.split_at_mut(vertex_count);
    let (oy, oz) = rest.split_at_mut(vertex_count);

    split_points(destination, [ox, oy, oz], &keys, &mut scratch, &mut sides, cluster_size);

    Ok(())
}
