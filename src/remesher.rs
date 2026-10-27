// This work is based on:
// William Lorensen, Harvey Cline. Marching Cubes: A High Resolution 3D Surface Construction Algorithm. 1987
// Michael Garland and Paul S. Heckbert. Surface simplification using quadric error metrics. 1997

use std::sync::OnceLock;

use crate::error::{Error, Result, ensure};
use crate::validate;

/// Produce a two-sided shell that wraps around surfaces of the original mesh, instead of a solid mesh.
pub const REMESH_SHELL: u32 = 1 << 0;
/// Compute optimal output positions that approximate the original surface as closely as possible.
pub const REMESH_SOLVE: u32 = 1 << 1;
/// Development-only option: place output vertices at voxel centers. Do *not* use.
#[doc(hidden)]
pub const REMESH_INTERNAL_DEBUG: u32 = 1 << 30;

// triangles are encoded as three vertex codes in three nibbles
// vertex codes match the bit order XYZ (e.g. 3 maps to X=1 Y=1 Z=0)
struct Case {
    code: u8,
    altmode: u8,
    triangles: [u16; 5], // up to 4 triangles and a null terminator
    alternate: [u16; 5], // alternate triangulation, restricted to have <= triangles vs primary
}

const fn case(code: u8, altmode: u8, t: &[u16], a: &[u16]) -> Case {
    let mut triangles = [0u16; 5];
    let mut alternate = [0u16; 5];
    let mut i = 0;
    while i < t.len() {
        triangles[i] = t[i];
        i += 1;
    }
    let mut i = 0;
    while i < a.len() {
        alternate[i] = a[i];
        i += 1;
    }
    Case {
        code,
        altmode,
        triangles,
        alternate,
    }
}

// the base cases establish a canonical triangulation; unlike classical Marching Cubes, we connect cube *corners* instead of edges
// each case may have an alternate triangulation, used based on deciders; crucially, open boundary edges must stay consistent as decider varies per cell
const BASE_CASES: [Case; 24] = [
    case(0x00, 0, &[], &[]),
    case(0x01, 0, &[], &[]),
    case(0x03, 0, &[], &[]),
    case(0x06, 0, &[], &[]),
    case(0x07, 0, &[0x120], &[]),
    case(0x0f, 2, &[0x130, 0x032], &[0x321, 0x120]),
    case(0x16, 1, &[0x124, 0x142], &[]),
    case(0x17, 0, &[0x124], &[]),
    case(0x18, 0, &[], &[]),
    case(0x19, 1, &[0x043, 0x403], &[]),
    case(0x1b, 1, &[0x134, 0x430], &[0x410, 0x013]),
    case(0x1d, 1, &[0x243, 0x340], &[0x320, 0x024]),
    case(0x1e, 1, &[0x142, 0x432, 0x341], &[0x132]),
    case(0x1f, 2, &[0x412, 0x213], &[0x134, 0x432]),
    case(0x3c, 1, &[0x345, 0x324, 0x543, 0x423], &[]),
    case(0x3d, 1, &[0x253, 0x524, 0x350], &[0x240, 0x504, 0x320]),
    case(0x3f, 2, &[0x325, 0x524], &[0x243, 0x345]),
    case(0x69, 1, &[0x035, 0x560, 0x306, 0x536], &[]),
    case(0x6b, 1, &[0x063, 0x605, 0x365], &[0x305]),
    case(0x6f, 1, &[0x365, 0x560], &[0x530, 0x036]),
    case(0x7e, 0, &[0x142, 0x536], &[]),
    case(0x7f, 0, &[0x365], &[]),
    case(0xf0, 2, &[0x547, 0x746], &[0x465, 0x567]),
    case(0xff, 0, &[], &[]),
];

/// For each cube case, a triangulation is derived from the base cases via 90 degree rotations.
struct RemeshTables {
    triangles: [[[u16; 5]; 2]; 256],
    count: [u8; 256],
    alt: [u8; 256],
}

fn build_remesh_tables() -> RemeshTables {
    // corner remap for 90 degree rotations around X and Y
    const ROTATIONS: [[u16; 8]; 2] = [[2, 3, 6, 7, 0, 1, 4, 5], [4, 0, 6, 2, 5, 1, 7, 3]];

    let mut t = RemeshTables {
        triangles: [[[0; 5]; 2]; 256],
        count: [0; 256],
        alt: [0; 256],
    };
    let mut filled = [0u8; 256];

    // copy base cases as is; we will generate the rest via rotations
    for c in &BASE_CASES {
        let code = usize::from(c.code);
        t.triangles[code][0] = c.triangles;
        t.triangles[code][1] = if c.altmode != 0 { c.alternate } else { c.triangles };
        t.alt[code] = c.altmode;
        filled[code] = 1;
    }

    // propagate base cases iteratively to fill all cube cases via rotations
    // note: the worst case is 5 passes but in practice we only need 3 to fill all cases
    for _ in 0..3 {
        for code in 0..256usize {
            if filled[code] != 1 {
                continue;
            }

            for rotation in &ROTATIONS {
                let mut rotated = 0usize;
                for (i, &r) in rotation.iter().enumerate() {
                    rotated |= ((code >> i) & 1) << r;
                }

                if filled[rotated] != 0 {
                    continue;
                }

                for k in 0..2 {
                    for i in 0..5 {
                        let tri = t.triangles[code][k][i];
                        if tri == 0 {
                            break;
                        }

                        let corner = |shift: u16| rotation[usize::from((tri >> shift) & 0xf)];
                        t.triangles[rotated][k][i] = (corner(8) << 8) | (corner(4) << 4) | corner(0);
                    }
                }

                t.alt[rotated] = t.alt[code];
                filled[rotated] = 1; // mark as pending
            }

            filled[code] = 2; // mark as processed
        }
    }

    // finalize auxiliary tables
    for code in 0..256 {
        t.count[code] = t.triangles[code][0].iter().take_while(|&&tri| tri != 0).count() as u8;
    }

    t
}

fn remesh_tables() -> &'static RemeshTables {
    static TABLES: OnceLock<RemeshTables> = OnceLock::new();
    TABLES.get_or_init(build_remesh_tables)
}

#[derive(Debug, Clone, Copy, Default)]
struct Voxel {
    coord: u32,
    octants: u8,

    px: f32,
    py: f32,
    pz: f32,
    w: f32,

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
}

fn voxel_accumulate(vox: &mut Voxel, px: f32, py: f32, pz: f32, w: f32) {
    vox.px += px * w;
    vox.py += py * w;
    vox.pz += pz * w;
    vox.w += w;
}

fn voxel_accumulate_q(vox: &mut Voxel, a: f32, b: f32, c: f32, d: f32, w: f32) {
    let aw = a * w;
    let bw = b * w;
    let cw = c * w;
    let dw = d * w;

    vox.a00 += a * aw;
    vox.a11 += b * bw;
    vox.a22 += c * cw;
    vox.a10 += a * bw;
    vox.a20 += a * cw;
    vox.a21 += b * cw;
    vox.b0 += a * dw;
    vox.b1 += b * dw;
    vox.b2 += c * dw;
    vox.c += d * dw;
}

fn voxel_error(vox: &Voxel, x: f32, y: f32, z: f32) -> f32 {
    let rx = (vox.b0 + vox.a10 * y) * 2.0 + vox.a00 * x;
    let ry = (vox.b1 + vox.a21 * z) * 2.0 + vox.a11 * y;
    let rz = (vox.b2 + vox.a20 * x) * 2.0 + vox.a22 * z;

    (vox.c + rx * x + ry * y + rz * z).abs()
}

fn voxel_solve(vox: &Voxel, lambda: f32) -> Option<(f32, f32, f32)> {
    // solve A*p = -b where A is the quadric matrix and b is the linear term
    let rw = lambda * vox.w;
    let (a00, a11, a22) = (rw + vox.a00, rw + vox.a11, rw + vox.a22);
    let (a10, a20, a21) = (vox.a10, vox.a20, vox.a21);
    let (x0, x1, x2) = (lambda * vox.px - vox.b0, lambda * vox.py - vox.b1, lambda * vox.pz - vox.b2);

    let eps = 1e-6 * vox.w;

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

    // substitute L^T*p = z
    let rz = z2;
    let ry = z1 - l21 * rz;
    let rx = z0 - l10 * ry - l20 * rz;

    (d0.abs() > eps && d1.abs() > eps && d2.abs() > eps).then_some((rx, ry, rz))
}

fn measure_grid(vertex_positions: &[f32], vertex_count: usize, vertex_stride_float: usize, resolution: i32) -> (f32, [f32; 3]) {
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

    // small subvoxel padding keeps vertices inside the last valid voxel to avoid clamping correctness issues
    extent *= (resolution as f32 + 1e-2) / resolution as f32;

    // rescale extents to [0..resolution - 2], because the first and last voxels are used as empty padding
    let scale = if extent == 0.0 { 0.0 } else { (resolution - 2) as f32 / extent };

    // center mesh in the grid; this improves voxelization for symmetric inputs
    let mut offset = [0.0f32; 3];
    for j in 0..3 {
        offset[j] = minv[j] - (extent - (maxv[j] - minv[j])) * 0.5;
    }

    (scale, offset)
}

/// Shared voxel grid state: occupancy grid (0 = empty, 0xff = inside, else 1-based offset in row) and row offsets.
struct Grid {
    data: Vec<u8>,
    rowmap: Vec<u32>,
    resolution: usize,
}

impl Grid {
    /// Occupied voxel at the given cell; `None` for empty or inside voxels.
    fn voxel_index(&self, x: usize, y: usize, z: usize) -> Option<usize> {
        let row = y + self.resolution * z;
        let g = self.data[x + self.resolution * row];
        (g != 0 && g != 0xff).then(|| self.rowmap[row] as usize + usize::from(g) - 1)
    }
}

fn voxelize(
    grid: &mut Grid,
    mut voxels: Option<&mut [Voxel]>,
    indices: &[u32],
    vertex_positions: &[f32],
    vertex_stride_float: usize,
    scale: f32,
    offset: &[f32; 3],
    options: u32,
) {
    let resolution = grid.resolution as i32;

    for tri in indices.as_chunks::<3>().0 {
        let va = &vertex_positions[tri[0] as usize * vertex_stride_float..];
        let vb = &vertex_positions[tri[1] as usize * vertex_stride_float..];
        let vc = &vertex_positions[tri[2] as usize * vertex_stride_float..];

        let (ex, ey, ez) = (vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]);
        let (fx, fy, fz) = (vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]);
        let (gx, gy, gz) = (vc[0] - vb[0], vc[1] - vb[1], vc[2] - vb[2]);

        // use maximum edge length to establish sampling rate
        let el2 = ex * ex + ey * ey + ez * ez;
        let fl2 = fx * fx + fy * fy + fz * fz;
        let gl2 = gx * gx + gy * gy + gz * gz;

        let mut max_edge = if el2 > fl2 { el2 } else { fl2 };
        max_edge = if max_edge > gl2 { max_edge } else { gl2 };
        max_edge = max_edge.sqrt();

        // we target 2 samples per voxel edge which should be enough to hit all voxels at any rotation
        let samples = ((max_edge * scale * 2.0) as i32).max(1).min(resolution * 2);

        // normal is used to compute area for weighting as well as for the quadric
        let (mut nx, mut ny, mut nz) = (ey * fz - ez * fy, ez * fx - ex * fz, ex * fy - ey * fx);
        let area = (nx * nx + ny * ny + nz * nz).sqrt();

        // skip degenerate triangles; they don't contribute to voxelization and may result in NaN when computing voxel centroid
        if area == 0.0 {
            continue;
        }

        let ns = 1.0 / area;
        nx *= ns;
        ny *= ns;
        nz *= ns;

        let (sx, sy, sz) = (va[0] - offset[0], va[1] - offset[1], va[2] - offset[2]);
        let mut sr = 1.0f32;
        let weight = area / ((samples + 1) * (samples + 2)) as f32;

        // this division is on a critical path, and for dense input meshes the branch is almost never taken so it's worth skipping it
        if samples > 1 {
            sr = 1.0 / samples as f32;
        }

        for u in 0..=samples {
            for v in 0..=samples - u {
                let (su, sv) = (u as f32 * sr, v as f32 * sr);
                let px = sx + su * ex + sv * fx;
                let py = sy + su * ey + sv * fy;
                let pz = sz + su * ez + sv * fz;

                // compute coordinates with an extra bit to use in octant mask
                let hx = (px * (scale * 2.0)) as i32;
                let hy = (py * (scale * 2.0)) as i32;
                let hz = (pz * (scale * 2.0)) as i32;
                let (mut x, mut y, mut z) = (hx >> 1, hy >> 1, hz >> 1);

                // safety: rounding errors and non-finite inputs may produce out of bounds coordinates, so we clamp them
                let cutoff = resolution - 3;
                x = if (x as u32) < cutoff as u32 { x } else { cutoff };
                y = if (y as u32) < cutoff as u32 { y } else { cutoff };
                z = if (z as u32) < cutoff as u32 { z } else { cutoff };

                let row = (y + 1) as usize + grid.resolution * (z + 1) as usize;
                let idx = (x + 1) as usize + grid.resolution * row;

                if let Some(voxels) = voxels.as_deref_mut() {
                    let vox = &mut voxels[grid.rowmap[row] as usize + usize::from(grid.data[idx]) - 1];

                    vox.coord = ((x as u32) << 20) | ((y as u32) << 10) | z as u32;
                    vox.octants |= 1 << ((hx & 1) | ((hy & 1) << 1) | ((hz & 1) << 2));

                    voxel_accumulate(vox, px, py, pz, weight);

                    if options & REMESH_SOLVE != 0 {
                        let distance = nx * px + ny * py + nz * pz;
                        voxel_accumulate_q(vox, nx, ny, nz, -distance, weight);
                    }
                } else {
                    grid.data[idx] = 1;
                }
            }
        }
    }
}

fn rowpack(grid: &mut Grid) -> usize {
    let resolution = grid.resolution;
    let mut result = 0;

    for (i, data) in grid.data.chunks_exact_mut(resolution).enumerate() {
        if data.iter().all(|&d| d == 0) {
            // mark empty rows with a sentinel to accelerate further processing
            grid.rowmap[i] = !0u32;
            continue;
        }

        let mut count = 0u8;

        for d in data.iter_mut() {
            count += *d;
            *d = if *d != 0 { count } else { 0 };
        }

        grid.rowmap[i] = result as u32;
        result += usize::from(count);
    }

    result
}

fn solidify_queue(row: usize, worklist: &mut [u32], queued: &mut [bool], pending: &mut usize) {
    if queued[row] {
        return;
    }

    queued[row] = true;
    worklist[*pending] = row as u32;
    *pending += 1;
}

fn solidify(grid: &mut Grid) {
    let resolution = grid.resolution;
    let slice = resolution * resolution;

    let mut worklist = vec![0u32; slice];
    let mut queued = vec![false; slice];
    let mut pending = 0;

    // mark the interior empty voxels as 'inside'; we will propagate 'empty' state from the boundary
    for z in 1..resolution - 1 {
        for y in 1..resolution - 1 {
            let row = y + resolution * z;

            // rows without occupied voxels shortcircuit the interior propagation and are kept as 'empty'
            if grid.rowmap[row] == !0u32 {
                continue;
            }

            for d in &mut grid.data[row * resolution + 1..row * resolution + resolution - 1] {
                *d = if *d == 0 { 0xff } else { *d };
            }
        }
    }

    // queue all rows for processing; note that boundary rows are always empty but we need them to propagate into neighboring rows
    for row in 0..slice {
        solidify_queue(row, &mut worklist, &mut queued, &mut pending);
    }

    while pending > 0 {
        pending -= 1;
        let row = worklist[pending] as usize;
        queued[row] = false;

        let base = row * resolution;

        // propagate outside state to the interior within row; rows without occupied voxels have no 'inside' voxels
        if grid.rowmap[row] != !0u32 {
            let data = &mut grid.data[base..base + resolution];

            for x in 1..resolution - 1 {
                data[x] = if data[x] == 0xff && data[x - 1] == 0 { 0 } else { data[x] };
            }

            for x in (1..resolution - 1).rev() {
                data[x] = if data[x] == 0xff && data[x + 1] == 0 { 0 } else { data[x] };
            }
        }

        // propagate outside state to the interior into neighboring rows
        let (y, z) = ((row % resolution) as i32, (row / resolution) as i32);

        for k in 0..4 {
            let yn = y + if k == 0 {
                -1
            } else if k == 1 {
                1
            } else {
                0
            };
            let zn = z + if k == 2 {
                -1
            } else if k == 3 {
                1
            } else {
                0
            };

            if yn < 1 || yn >= resolution as i32 - 1 || zn < 1 || zn >= resolution as i32 - 1 {
                continue;
            }

            let rown = yn as usize + resolution * zn as usize;

            // neighboring rows without occupied voxels have no 'inside' voxels and can be skipped
            if grid.rowmap[rown] == !0u32 {
                continue;
            }

            let basen = rown * resolution;
            let mut changed = 0u8;

            for x in 1..resolution - 1 {
                let d = grid.data[base + x];
                let dn = grid.data[basen + x];
                let rep = if d == 0 && dn == 0xff { 0 } else { dn };
                changed |= rep ^ dn;
                grid.data[basen + x] = rep;
            }

            if changed != 0 {
                solidify_queue(rown, &mut worklist, &mut queued, &mut pending);
            }
        }
    }
}

fn solve(voxels: &mut [Voxel], scale: f32, options: u32) {
    // regularization factor is a tradeoff between solve stability/uniformity and adherence to the original surface
    let factor = 3e-2;

    // cutoff is a squared distance in model space; sqrt(3)/scale is the voxel diagonal
    let cutoff = 3.0 / (scale * scale);
    let rscale = 1.0 / scale;

    for vox in voxels.iter_mut() {
        let (mut px, mut py, mut pz) = (vox.px / vox.w, vox.py / vox.w, vox.pz / vox.w);

        // voxel corner, used for clamping and debug visualization
        let cx = ((vox.coord >> 20) & 1023) as i32 as f32 * rscale;
        let cy = ((vox.coord >> 10) & 1023) as i32 as f32 * rscale;
        let cz = (vox.coord & 1023) as i32 as f32 * rscale;

        if options & REMESH_SOLVE != 0 {
            // compute minimizing point; this includes regularization to stabilize the solve
            if let Some((sx, sy, sz)) = voxel_solve(vox, factor) {
                let d = (sx - px) * (sx - px) + (sy - py) * (sy - py) + (sz - pz) * (sz - pz);

                // reject solutions that move the vertex too far from voxel centroid; this is a safety net in case regularization is insufficient
                if d < cutoff {
                    // use a clamped solution to reduce intersections
                    let clamp = |s: f32, c: f32| {
                        let p = if s > c { s } else { c };
                        if p < c + rscale { p } else { c + rscale }
                    };

                    px = clamp(sx, cx);
                    py = clamp(sy, cy);
                    pz = clamp(sz, cz);
                }
            }
        }

        if options & REMESH_INTERNAL_DEBUG != 0 {
            px = cx + 0.5 * rscale;
            py = cy + 0.5 * rscale;
            pz = cz + 0.5 * rscale;
        }

        vox.px = px;
        vox.py = py;
        vox.pz = pz;
    }
}

fn emit_vertex(result: &mut [f32], x: usize, y: usize, z: usize, corner: u16, grid: &Grid, voxels: &[Voxel], offset: &[f32; 3]) {
    let (ox, oy, oz) = (
        usize::from(corner & 1),
        usize::from((corner >> 1) & 1),
        usize::from((corner >> 2) & 1),
    );

    if let Some(i) = grid.voxel_index(x + ox, y + oy, z + oz) {
        let vox = &voxels[i];

        result[0] = vox.px + offset[0];
        result[1] = vox.py + offset[1];
        result[2] = vox.pz + offset[2];
    }
}

fn octant_decider(x: usize, y: usize, z: usize, cube: usize, grid: &Grid, voxels: &[Voxel]) -> bool {
    for c in 0..8 {
        if cube & (1 << c) != 0 {
            let (ox, oy, oz) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);

            // note: unlike vertex emission, which never consults interior occupied corners, here we need to skip corners without an occupied voxel
            let Some(i) = grid.voxel_index(x + ox, y + oy, z + oz) else {
                continue;
            };

            // test octant contained within the cell (opposite of the corner index)
            if voxels[i].octants & (1 << (7 - c)) != 0 {
                return false;
            }
        }
    }

    // select alternate configuration if *all* octants are empty
    true
}

fn quadric_decider(x: usize, y: usize, z: usize, cube: usize, grid: &Grid, voxels: &[Voxel], rscale: f32, tables: &RemeshTables) -> bool {
    // quads are encoded implicitly as 0xabc 0xcbd
    let quad = (u32::from(tables.triangles[cube][0][0]) << 4) | u32::from(tables.triangles[cube][0][1] & 0xf);

    let mut corner = [Voxel::default(); 4];

    for (i, cv) in corner.iter_mut().enumerate() {
        let c = ((quad >> (12 - i * 4)) & 0xf) as usize;

        if let Some(vi) = grid.voxel_index(x + (c & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1)) {
            *cv = voxels[vi];
        }
    }

    // accumulate the quadrics of all four corners; we sum (a+d)+(b+c) to get the same results on both sides of thin sheets
    let [va, vb, vc, vd] = corner;
    let sum4 = |f: fn(&Voxel) -> f32| (f(&va) + f(&vd)) + (f(&vb) + f(&vc));

    let sum = Voxel {
        a00: sum4(|v| v.a00),
        a11: sum4(|v| v.a11),
        a22: sum4(|v| v.a22),
        a10: sum4(|v| v.a10),
        a20: sum4(|v| v.a20),
        a21: sum4(|v| v.a21),
        b0: sum4(|v| v.b0),
        b1: sum4(|v| v.b1),
        b2: sum4(|v| v.b2),
        c: sum4(|v| v.c),
        w: sum4(|v| v.w),
        ..Voxel::default()
    };

    // evaluate error for midpoints of primary (bc) and alternate (ad) diagonal
    let error0 = voxel_error(&sum, (vb.px + vc.px) * 0.5, (vb.py + vc.py) * 0.5, (vb.pz + vc.pz) * 0.5);
    let error1 = voxel_error(&sum, (va.px + vd.px) * 0.5, (va.py + vd.py) * 0.5, (va.pz + vd.pz) * 0.5);

    // select alternate configuration if it is clearly better and if the primary error is not too small, to avoid excessive flips
    error0 > sum.w * (0.1 * 0.1 * rscale * rscale) && error1 < error0 * (0.85 * 0.85)
}

fn polygonize(
    mut destination: Option<&mut [f32]>,
    max_triangle_count: usize,
    grid: &Grid,
    voxels: &[Voxel],
    scale: f32,
    offset: &[f32; 3],
    options: u32,
) -> usize {
    let tables = remesh_tables();
    let resolution = grid.resolution;
    let slice = resolution * resolution;
    let rscale = 1.0 / scale;

    let mut result = 0;

    for z in 0..resolution - 1 {
        for y in 0..resolution - 1 {
            let row = y + resolution * z;
            let rowmap = &grid.rowmap;

            // skip processing if all four rows are entirely empty as that guarantees empty output (even three out of four would be enough but that's more expensive to check)
            if (rowmap[row] & rowmap[row + 1] & rowmap[row + resolution] & rowmap[row + resolution + 1]) == !0u32 {
                continue;
            }

            let data = &grid.data[resolution * row..];
            let occupied = |i: usize| usize::from(data[i] != 0);

            // we track each slice as a 8-bit code (matching cube indexing) as we iterate through the row to avoid extra lookups
            let mut last = occupied(0) | (occupied(resolution) << 2) | (occupied(slice) << 4) | (occupied(slice + resolution) << 6);

            for x in 0..resolution - 1 {
                let next = occupied(x + 1)
                    | (occupied(x + 1 + resolution) << 2)
                    | (occupied(x + 1 + slice) << 4)
                    | (occupied(x + 1 + slice + resolution) << 6);

                // next slice bits are in position for the next iteration so we need to shift them to retrieve current cube code
                let cube = last | (next << 1);
                last = next;

                if cube == 0 || cube == 0xff {
                    continue;
                }

                let Some(dest) = destination.as_deref_mut() else {
                    // fast path: we can statically determine the upper bound on triangles the loop below will output based on cube configuration
                    result += usize::from(tables.count[cube]);
                    continue;
                };

                // deciders are only evaluated for cells with an alternate configuration to reduce overhead
                let alt = if tables.alt[cube] == 1 {
                    usize::from(octant_decider(x, y, z, cube, grid, voxels))
                } else if tables.alt[cube] == 2 && options & REMESH_SOLVE != 0 {
                    usize::from(quadric_decider(x, y, z, cube, grid, voxels, rscale, tables))
                } else {
                    0
                };

                for &tri in tables.triangles[cube][alt].iter().take_while(|&&tri| tri != 0) {
                    // note: we only emit the triangle if we have space for it, but we count it regardless for consistent capacity estimation
                    if result < max_triangle_count {
                        let out = &mut dest[result * 9..result * 9 + 9];

                        emit_vertex(&mut out[0..3], x, y, z, (tri >> 8) & 0xf, grid, voxels, offset);
                        emit_vertex(&mut out[3..6], x, y, z, (tri >> 4) & 0xf, grid, voxels, offset);
                        emit_vertex(&mut out[6..9], x, y, z, tri & 0xf, grid, voxels, offset);
                    }

                    result += 1;
                }
            }
        }
    }

    result
}

/// Experimental: Voxel remesher.
/// Generates a new mesh that approximates or refines the original mesh, attempting to preserve mesh appearance at a given voxel resolution.
/// Returns the number of triangles in the new mesh, with `destination` containing a vertex position for each triangle corner.
///
/// `destination` can be `None`; otherwise it must hold `max_triangle_count * 9` floats. When `destination` is `None` or insufficiently
/// large, the returned value is an upper bound on the number of triangles. `resolution` must be in 4..=256.
pub fn remesh(
    destination: Option<&mut [f32]>,
    max_triangle_count: usize,
    indices: &[u32],
    index_count: usize,
    vertex_positions: &[f32],
    vertex_count: usize,
    vertex_positions_stride: usize,
    resolution: i32,
    options: u32,
) -> Result<usize> {
    validate::triangles(index_count)?;
    validate::floats(vertex_positions, vertex_count, vertex_positions_stride, 12)?;
    ensure(
        (4..=256).contains(&resolution),
        Error::InvalidArgument("resolution must be in 4..=256"),
    )?;
    let indices = validate::indices(indices, index_count, vertex_count)?;
    let destination = match destination {
        Some(d) => Some(validate::output(d, max_triangle_count * 9)?),
        None => {
            ensure(
                max_triangle_count == 0,
                Error::InvalidArgument("max_triangle_count must be 0 without destination"),
            )?;
            None
        }
    };

    let vertex_stride_float = vertex_positions_stride / 4;
    let res = resolution as usize;

    // measure voxel grid to compute position => voxel mapping
    let (scale, offset) = measure_grid(vertex_positions, vertex_count, vertex_stride_float, resolution);

    // rasterize triangles into the voxel grid: in the first pass, this simply tags occupied voxels
    let mut grid = Grid {
        data: vec![0u8; res * res * res],
        rowmap: vec![0u32; res * res],
        resolution: res,
    };

    voxelize(
        &mut grid,
        None,
        indices,
        vertex_positions,
        vertex_stride_float,
        scale,
        &offset,
        options,
    );

    // compute row offsets for occupied voxels; also stores ~0u for empty rows to accelerate further processing
    let voxel_count = rowpack(&mut grid);

    // allocate additional voxel data for each occupied voxel; this can be filled in the second pass to compute positions
    // note that we only do this if we need to compute output triangles; counting runs skip it for performance
    let mut voxels = if destination.is_some() {
        vec![Voxel::default(); voxel_count]
    } else {
        Vec::new()
    };

    // fill in empty voxels that are not reachable from the grid boundary; the inside empty voxels are marked with 0xff
    // note that these voxels do *not* have the associated Voxel data as they were never voxelized
    if options & REMESH_SHELL == 0 {
        solidify(&mut grid);
    }

    if destination.is_some() {
        // accumulate voxel positions: in the second pass, this computes enough data in each voxel to calculate positions
        voxelize(
            &mut grid,
            Some(&mut voxels),
            indices,
            vertex_positions,
            vertex_stride_float,
            scale,
            &offset,
            options,
        );

        // compute final voxel positions; each voxel has a single resulting position that will be emitted during polygonization
        solve(&mut voxels, scale, options);
    }

    // output triangles from the voxel grid; if destination is None, this still counts the number of triangles that would be generated
    Ok(polygonize(destination, max_triangle_count, &grid, &voxels, scale, &offset, options))
}
