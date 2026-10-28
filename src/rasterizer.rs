// This work is based on:
// Nicolas Capens. Advanced Rasterization. 2004

use crate::error::Result;
use crate::index::{self, IndexType};
use crate::validate;

const VIEWPORT: usize = 256;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OverdrawStatistics {
    pub pixels_covered: u32,
    pub pixels_shaded: u32,
    /// shaded pixels / covered pixels; best case 1.0
    pub overdraw: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CoverageStatistics {
    pub coverage: [f32; 3],
    /// viewport size in mesh coordinates
    pub extent: f32,
}

/// `[y][x][sign]` buffers stored flat.
struct OverdrawBuffer {
    z: Vec<f32>,
    overdraw: Vec<u32>,
}

impl OverdrawBuffer {
    fn new() -> Self {
        OverdrawBuffer {
            z: vec![0.0; VIEWPORT * VIEWPORT * 2],
            overdraw: vec![0; VIEWPORT * VIEWPORT * 2],
        }
    }

    fn clear(&mut self) {
        self.z.fill(0.0);
        self.overdraw.fill(0);
    }
}

fn compute_depth_gradients(x1: f32, y1: f32, z1: f32, x2: f32, y2: f32, z2: f32, x3: f32, y3: f32, z3: f32) -> (f32, f32, f32) {
    // z2 = z1 + dzdx * (x2 - x1) + dzdy * (y2 - y1)
    // z3 = z1 + dzdx * (x3 - x1) + dzdy * (y3 - y1)
    // (x2-x1 y2-y1)(dzdx) = (z2-z1)
    // (x3-x1 y3-y1)(dzdy)   (z3-z1)
    // we'll solve it with Cramer's rule
    let det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
    let invdet = if det == 0.0 { 0.0 } else { 1.0 / det };

    let dzdx = ((z2 - z1) * (y3 - y1) - (y2 - y1) * (z3 - z1)) * invdet;
    let dzdy = ((x2 - x1) * (z3 - z1) - (z2 - z1) * (x3 - x1)) * invdet;

    (det, dzdx, dzdy)
}

// half-space fixed point triangle rasterizer
fn rasterize(buffer: &mut OverdrawBuffer, v1: [f32; 3], v2: [f32; 3], v3: [f32; 3]) {
    let [v1x, v1y, mut v1z] = v1;
    let [mut v2x, mut v2y, v2z] = v2;
    let [mut v3x, mut v3y, v3z] = v3;

    // compute depth gradients
    let (det, mut dzx, mut dzy) = compute_depth_gradients(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z);
    let sign = usize::from(det > 0.0);

    // flip backfacing triangles to simplify rasterization logic
    if sign != 0 {
        // flipping v2 & v3 preserves depth gradients since they're based on v1; only v1z is used below
        core::mem::swap(&mut v2x, &mut v3x);
        core::mem::swap(&mut v2y, &mut v3y);

        // flip depth since we rasterize backfacing triangles to second buffer with reverse Z; only v1z is used below
        v1z = VIEWPORT as f32 - v1z;
        dzx = -dzx;
        dzy = -dzy;
    }

    // coordinates, 28.4 fixed point
    let x1 = (16.0 * v1x + 0.5) as i32;
    let x2 = (16.0 * v2x + 0.5) as i32;
    let x3 = (16.0 * v3x + 0.5) as i32;

    let y1 = (16.0 * v1y + 0.5) as i32;
    let y2 = (16.0 * v2y + 0.5) as i32;
    let y3 = (16.0 * v3y + 0.5) as i32;

    // bounding rectangle, clipped against viewport
    // since we rasterize pixels with covered centers, min >0.5 should round up
    // as for max, due to top-left filling convention we will never rasterize right/bottom edges
    // so max >= 0.5 should round down for inclusive bounds, and up for exclusive (in our case)
    let minx = (x1.min(x2).min(x3).wrapping_add(7) >> 4).max(0);
    let miny = (y1.min(y2).min(y3).wrapping_add(7) >> 4).max(0);
    let maxx = (x1.max(x2).max(x3).wrapping_add(7) >> 4).min(VIEWPORT as i32);
    let maxy = (y1.max(y2).max(y3).wrapping_add(7) >> 4).min(VIEWPORT as i32);

    // deltas, 28.4 fixed point
    let dx12 = x1.wrapping_sub(x2);
    let dx23 = x2.wrapping_sub(x3);
    let dx31 = x3.wrapping_sub(x1);

    let dy12 = y1.wrapping_sub(y2);
    let dy23 = y2.wrapping_sub(y3);
    let dy31 = y3.wrapping_sub(y1);

    // fill convention correction
    let tl1 = i32::from(dy12 < 0 || (dy12 == 0 && dx12 > 0));
    let tl2 = i32::from(dy23 < 0 || (dy23 == 0 && dx23 > 0));
    let tl3 = i32::from(dy31 < 0 || (dy31 == 0 && dx31 > 0));

    // half edge equations, 24.8 fixed point
    // note that we offset minx/miny by half pixel since we want to rasterize pixels with covered centers
    let fx = (minx << 4) + 8;
    let fy = (miny << 4) + 8;
    let edge = |dx: i32, dy: i32, x: i32, y: i32, tl: i32| {
        dx.wrapping_mul(fy.wrapping_sub(y))
            .wrapping_sub(dy.wrapping_mul(fx.wrapping_sub(x)))
            .wrapping_add(tl - 1)
    };
    let mut cy1 = edge(dx12, dy12, x1, y1, tl1);
    let mut cy2 = edge(dx23, dy23, x2, y2, tl2);
    let mut cy3 = edge(dx31, dy31, x3, y3, tl3);
    let mut zy = v1z + (dzx * fx.wrapping_sub(x1) as f32 + dzy * fy.wrapping_sub(y1) as f32) * (1.0 / 16.0);

    for y in miny..maxy {
        let mut cx1 = cy1;
        let mut cx2 = cy2;
        let mut cx3 = cy3;
        let mut zx = zy;

        for x in minx..maxx {
            // check if all CXn are non-negative
            if (cx1 | cx2 | cx3) >= 0 {
                let i = (y as usize * VIEWPORT + x as usize) * 2 + sign;

                if zx >= buffer.z[i] {
                    buffer.z[i] = zx;
                    buffer.overdraw[i] = buffer.overdraw[i].wrapping_add(1);
                }
            }

            cx1 = cx1.wrapping_sub(((dy12 as u32) << 4) as i32);
            cx2 = cx2.wrapping_sub(((dy23 as u32) << 4) as i32);
            cx3 = cx3.wrapping_sub(((dy31 as u32) << 4) as i32);
            zx += dzx;
        }

        cy1 = cy1.wrapping_add(((dx12 as u32) << 4) as i32);
        cy2 = cy2.wrapping_add(((dx23 as u32) << 4) as i32);
        cy3 = cy3.wrapping_add(((dx31 as u32) << 4) as i32);
        zy += dzy;
    }
}

fn transform_triangles(
    triangles: &mut [f32],
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> f32 {
    let vertex_stride_float = vertex_positions_stride / 4;

    let mut minv = [f32::MAX; 3];
    let mut maxv = [-f32::MAX; 3];

    for i in 0..vertex_count {
        let v = &vertex_positions[i * vertex_stride_float..];

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

    let scale = if extent == 0.0 { 0.0 } else { VIEWPORT as f32 / extent };

    for (t, &index) in triangles.as_chunks_mut::<3>().0.iter_mut().zip(indices) {
        let v = &vertex_positions[index as usize * vertex_stride_float..];

        t[0] = (v[0] - minv[0]) * scale;
        t[1] = (v[1] - minv[1]) * scale;
        t[2] = (v[2] - minv[2]) * scale;
    }

    extent
}

fn rasterize_triangles(buffer: &mut OverdrawBuffer, triangles: &[f32], axis: usize) {
    for tri in triangles.as_chunks::<9>().0 {
        let vn0 = [tri[0], tri[1], tri[2]];
        let vn1 = [tri[3], tri[4], tri[5]];
        let vn2 = [tri[6], tri[7], tri[8]];

        let swizzle = |v: [f32; 3]| match axis {
            0 => [v[2], v[1], v[0]],
            1 => [v[0], v[2], v[1]],
            _ => [v[1], v[0], v[2]],
        };

        rasterize(buffer, swizzle(vn0), swizzle(vn1), swizzle(vn2));
    }
}

/// Overdraw analyzer.
/// Returns overdraw statistics using a software rasterizer; results may not match actual GPU performance.
pub fn analyze_overdraw<T: IndexType>(
    indices: &[T],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> Result<OverdrawStatistics> {
    analyze_overdraw_u32(
        &index::input(indices),
        index_count,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
    )
}

fn analyze_overdraw_u32(
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> Result<OverdrawStatistics> {
    validate::triangles(index_count)?;
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    let indices = validate::indices(indices, index_count, vertex_count)?;

    let mut result = OverdrawStatistics::default();

    let mut triangles = vec![0.0f32; index_count * 3];
    transform_triangles(&mut triangles, indices, vertex_positions, vertex_count, vertex_positions_stride);

    let mut buffer = OverdrawBuffer::new();

    for axis in 0..3 {
        buffer.clear();
        rasterize_triangles(&mut buffer, &triangles, axis);

        for &overdraw in &buffer.overdraw {
            result.pixels_covered += u32::from(overdraw > 0);
            result.pixels_shaded = result.pixels_shaded.wrapping_add(overdraw);
        }
    }

    result.overdraw = if result.pixels_covered != 0 {
        result.pixels_shaded as f32 / result.pixels_covered as f32
    } else {
        0.0
    };

    Ok(result)
}

/// Coverage analyzer.
/// Returns coverage statistics (ratio of viewport pixels covered from each axis) using a software rasterizer.
pub fn analyze_coverage<T: IndexType>(
    indices: &[T],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> Result<CoverageStatistics> {
    analyze_coverage_u32(
        &index::input(indices),
        index_count,
        vertex_positions,
        vertex_count,
        vertex_positions_stride,
    )
}

fn analyze_coverage_u32(
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
) -> Result<CoverageStatistics> {
    validate::triangles(index_count)?;
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    let indices = validate::indices(indices, index_count, vertex_count)?;

    let mut result = CoverageStatistics::default();

    let mut triangles = vec![0.0f32; index_count * 3];
    let extent = transform_triangles(&mut triangles, indices, vertex_positions, vertex_count, vertex_positions_stride);

    let mut buffer = OverdrawBuffer::new();

    for axis in 0..3 {
        buffer.clear();
        rasterize_triangles(&mut buffer, &triangles, axis);

        let covered = buffer.overdraw.as_chunks::<2>().0.iter().filter(|o| (o[0] | o[1]) > 0).count();

        result.coverage[axis] = covered as f32 / (VIEWPORT * VIEWPORT) as f32;
    }

    result.extent = extent;

    Ok(result)
}
