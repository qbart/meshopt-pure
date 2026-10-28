//! `demo/main.cpp`: runs the library on OBJ meshes and prints statistics for every algorithm.
//! C++ `assert`s (the demo verifies algorithm correctness with them) are `assert!`s; library errors propagate as `Err`.

use std::sync::OnceLock;
use std::time::Instant;

use gltfpack::cstd::format_f;
use gltfpack::fast_obj::{fast_obj_destroy, fast_obj_read};
use meshoptimizer::*;

use crate::sdefl::{SDEFL_LVL_DEF, Sdefl, sdefl_bound, sdeflate};

pub type Result<T> = meshoptimizer::Result<T>;

/// Seconds since the first call (C++: `clock_gettime(CLOCK_MONOTONIC)`).
pub fn timestamp() -> f64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// `%.{p}f`, including glibc's `nan`/`-nan`.
pub(crate) fn f(v: f64, p: usize) -> String {
    format_f(v, p)
}

/// `struct Vertex { float px, py, pz; float nx, ny, nz; float tx, ty; }`
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Vertex {
    pub px: f32,
    pub py: f32,
    pub pz: f32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub tx: f32,
    pub ty: f32,
}

pub const VERTEX_SIZE: usize = 32;

/// Vertex fields as floats: positions at offset 0, normals at 3, UVs at 6, with a stride of `VERTEX_SIZE` bytes.
pub fn floats(vertices: &[Vertex]) -> Vec<f32> {
    vertices
        .iter()
        .flat_map(|v| [v.px, v.py, v.pz, v.nx, v.ny, v.nz, v.tx, v.ty])
        .collect()
}

pub fn bytes(vertices: &[Vertex]) -> Vec<u8> {
    floats(vertices).iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn from_bytes(data: &[u8]) -> Vec<Vertex> {
    data.as_chunks::<VERTEX_SIZE>()
        .0
        .iter()
        .map(|b| {
            let (c, _) = b.as_chunks::<4>();
            let g = |i: usize| f32::from_le_bytes(c[i]);
            Vertex {
                px: g(0),
                py: g(1),
                pz: g(2),
                nx: g(3),
                ny: g(4),
                nz: g(5),
                tx: g(6),
                ty: g(7),
            }
        })
        .collect()
}

#[derive(Clone, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

/// `meshopt_optimizeVertexFetch` into `destination` (C++ often passes the same array as source and destination).
fn optimize_fetch(destination: &mut Vec<Vertex>, indices: &mut [u32], source: &[Vertex]) -> Result<()> {
    let src = bytes(source);
    let mut dst = vec![0u8; src.len()];
    let count = optimize_vertex_fetch(&mut dst, indices, indices.len(), &src, source.len(), VERTEX_SIZE)?;
    *destination = from_bytes(&dst[..count * VERTEX_SIZE]);
    Ok(())
}

pub fn parse_obj(path: &str, reindex: &mut f64) -> Result<Mesh> {
    let Some(obj) = fast_obj_read(path) else {
        println!("Error loading {path}: file not found");
        return Ok(Mesh::default());
    };

    let mut total_indices = 0;

    for &fv in &obj.face_vertices[..obj.face_count as usize] {
        if fv > 2 {
            total_indices += 3 * (fv as usize - 2);
        }
    }

    let mut vertices = vec![Vertex::default(); total_indices];

    let mut vertex_offset = 0;
    let mut index_offset = 0;

    for &fv in &obj.face_vertices[..obj.face_count as usize] {
        if fv <= 2 {
            continue;
        }

        for j in 0..fv as usize {
            let gi = obj.indices[index_offset + j];
            let (p, n, t) = (gi.p as usize, gi.n as usize, gi.t as usize);

            let v = Vertex {
                px: obj.positions[p * 3],
                py: obj.positions[p * 3 + 1],
                pz: obj.positions[p * 3 + 2],
                nx: obj.normals[n * 3],
                ny: obj.normals[n * 3 + 1],
                nz: obj.normals[n * 3 + 2],
                tx: obj.texcoords[t * 2],
                ty: obj.texcoords[t * 2 + 1],
            };

            // triangulate polygon on the fly; offset-3 is always the first polygon vertex
            if j >= 3 {
                vertices[vertex_offset] = vertices[vertex_offset - 3];
                vertices[vertex_offset + 1] = vertices[vertex_offset - 1];
                vertex_offset += 2;
            }

            vertices[vertex_offset] = v;
            vertex_offset += 1;
        }

        index_offset += fv as usize;
    }

    fast_obj_destroy(obj);

    *reindex = timestamp();

    // empty mesh
    if total_indices == 0 {
        return Ok(Mesh::default());
    }

    let mut remap = vec![0u32; total_indices];

    let data = bytes(&vertices);
    let total_vertices = generate_vertex_remap::<u32>(&mut remap, None, total_indices, &data, total_indices, VERTEX_SIZE)?;

    let mut indices = vec![0u32; total_indices];
    remap_index_buffer::<u32>(&mut indices, None, total_indices, &remap)?;

    let mut out = vec![0u8; total_vertices * VERTEX_SIZE];
    remap_vertex_buffer(&mut out, &data, total_indices, VERTEX_SIZE, &remap)?;

    Ok(Mesh {
        vertices: from_bytes(&out),
        indices,
    })
}

pub fn dump_obj(vertices: &[Vertex], indices: &[u32], recompute_normals: bool) {
    let mut normals = Vec::new();

    if recompute_normals {
        normals = vec![0f32; vertices.len() * 3];

        for tri in indices.as_chunks::<3>().0 {
            let (va, vb, vc) = (vertices[tri[0] as usize], vertices[tri[1] as usize], vertices[tri[2] as usize]);

            let nx = (vb.py - va.py) * (vc.pz - va.pz) - (vb.pz - va.pz) * (vc.py - va.py);
            let ny = (vb.pz - va.pz) * (vc.px - va.px) - (vb.px - va.px) * (vc.pz - va.pz);
            let nz = (vb.px - va.px) * (vc.py - va.py) - (vb.py - va.py) * (vc.px - va.px);

            for &index in tri {
                normals[index as usize * 3] += nx;
                normals[index as usize * 3 + 1] += ny;
                normals[index as usize * 3 + 2] += nz;
            }
        }
    }

    for (i, v) in vertices.iter().enumerate() {
        let (mut nx, mut ny, mut nz) = (v.nx, v.ny, v.nz);

        if recompute_normals {
            nx = normals[i * 3];
            ny = normals[i * 3 + 1];
            nz = normals[i * 3 + 2];

            let l = (nx * nx + ny * ny + nz * nz).sqrt();
            let s = if l == 0.0 { 0.0 } else { 1.0 / l };

            nx *= s;
            ny *= s;
            nz *= s;
        }

        eprintln!("v {} {} {}", f(v.px.into(), 6), f(v.py.into(), 6), f(v.pz.into(), 6));
        eprintln!("vn {} {} {}", f(nx.into(), 6), f(ny.into(), 6), f(nz.into(), 6));
    }

    for tri in indices.as_chunks::<3>().0 {
        let (a, b, c) = (tri[0] + 1, tri[1] + 1, tri[2] + 1);
        eprintln!("f {a}//{a} {b}//{b} {c}//{c}");
    }
}

pub fn dump_obj_section(section: &str, indices: &[u32]) {
    eprintln!("o {section}");

    for tri in indices.as_chunks::<3>().0 {
        let (a, b, c) = (tri[0] + 1, tri[1] + 1, tri[2] + 1);
        eprintln!("f {a}//{a} {b}//{b} {c}//{c}");
    }
}

/// `struct PackedVertex { unsigned short px, py, pz, pw; signed char nx, ny, nz, nw; unsigned short tx, ty; }`
const PACKED_VERTEX_SIZE: usize = 16;

fn pack_mesh(vertices: &[Vertex]) -> Vec<u8> {
    let mut pv = Vec::with_capacity(vertices.len() * PACKED_VERTEX_SIZE);

    for vi in vertices {
        for v in [quantize_half(vi.px), quantize_half(vi.py), quantize_half(vi.pz), 0] {
            pv.extend_from_slice(&v.to_le_bytes());
        }

        for v in [vi.nx, vi.ny, vi.nz] {
            pv.push(quantize_snorm(v, 8) as i8 as u8);
        }
        pv.push(0);

        for v in [quantize_half(vi.tx), quantize_half(vi.ty)] {
            pv.extend_from_slice(&v.to_le_bytes());
        }
    }

    pv
}

/// `struct PackedVertexOct { unsigned short px, py, pz; signed char nu, nv; unsigned short tx, ty; }`
const PACKED_VERTEX_OCT_SIZE: usize = 12;

fn pack_mesh_oct(vertices: &[Vertex]) -> Vec<u8> {
    let mut pv = Vec::with_capacity(vertices.len() * PACKED_VERTEX_OCT_SIZE);

    for vi in vertices {
        for v in [quantize_half(vi.px), quantize_half(vi.py), quantize_half(vi.pz)] {
            pv.extend_from_slice(&v.to_le_bytes());
        }

        let nsum = vi.nx.abs() + vi.ny.abs() + vi.nz.abs();
        let nx = vi.nx / nsum;
        let ny = vi.ny / nsum;
        let nz = vi.nz;

        let nu = if nz >= 0.0 {
            nx
        } else {
            (1.0 - ny.abs()) * if nx >= 0.0 { 1.0 } else { -1.0 }
        };
        let nv = if nz >= 0.0 {
            ny
        } else {
            (1.0 - nx.abs()) * if ny >= 0.0 { 1.0 } else { -1.0 }
        };

        pv.push(quantize_snorm(nu, 8) as i8 as u8);
        pv.push(quantize_snorm(nv, 8) as i8 as u8);

        for v in [quantize_half(vi.tx), quantize_half(vi.ty)] {
            pv.extend_from_slice(&v.to_le_bytes());
        }
    }

    pv
}

/// Packed vertex formats of `packVertex` / `encodeVertex`.
#[derive(Clone, Copy)]
enum Packing {
    Plain,
    Oct,
}

impl Packing {
    fn size(self) -> usize {
        match self {
            Packing::Plain => PACKED_VERTEX_SIZE,
            Packing::Oct => PACKED_VERTEX_OCT_SIZE,
        }
    }

    fn pack(self, vertices: &[Vertex]) -> Vec<u8> {
        match self {
            Packing::Plain => pack_mesh(vertices),
            Packing::Oct => pack_mesh_oct(vertices),
        }
    }
}

fn simplify_report(name: &str, mesh: &Mesh, lod: &Mesh, result_error: f32, start: f64, end: f64) {
    println!(
        "{name:<9}: {} triangles => {} triangles ({}% deviation) in {} msec",
        mesh.indices.len() / 3,
        lod.indices.len() / 3,
        f(f64::from(result_error * 100.0), 2),
        f((end - start) * 1000.0, 2)
    );
}

pub fn simplify_mesh(mesh: &Mesh, threshold: f32, options: u32) -> Result<()> {
    let mut lod = Mesh::default();

    let start = timestamp();

    let target_index_count = (mesh.indices.len() as f32 * threshold) as usize;
    let target_error = 1e-2f32;
    let mut result_error = 0f32;

    let positions = floats(&mesh.vertices);

    // note: simplify needs space for index_count elements in the destination array, not target_index_count
    lod.indices = vec![0; mesh.indices.len()];
    let count = simplify(
        &mut lod.indices,
        &mesh.indices,
        mesh.indices.len(),
        &positions,
        mesh.vertices.len(),
        VERTEX_SIZE,
        target_index_count,
        target_error,
        options,
        Some(&mut result_error),
    )?;
    lod.indices.truncate(count);

    optimize_fetch(&mut lod.vertices, &mut lod.indices, &mesh.vertices)?;

    let end = timestamp();

    simplify_report("Simplify", mesh, &lod, result_error, start, end);
    Ok(())
}

pub fn simplify_attr(mesh: &Mesh, threshold: f32, options: u32) -> Result<()> {
    let mut lod = Mesh::default();

    let start = timestamp();

    let target_index_count = (mesh.indices.len() as f32 * threshold) as usize;
    let target_error = 1e-2f32;
    let mut result_error = 0f32;

    let nrm_weight = 0.5f32;
    let attr_weights = [nrm_weight, nrm_weight, nrm_weight];

    let data = floats(&mesh.vertices);

    // note: simplify needs space for index_count elements in the destination array, not target_index_count
    lod.indices = vec![0; mesh.indices.len()];
    let count = simplify_with_attributes(
        &mut lod.indices,
        &mesh.indices,
        mesh.indices.len(),
        &data,
        mesh.vertices.len(),
        VERTEX_SIZE,
        &data[3..],
        VERTEX_SIZE,
        &attr_weights,
        3,
        None,
        target_index_count,
        target_error,
        options,
        Some(&mut result_error),
    )?;
    lod.indices.truncate(count);

    optimize_fetch(&mut lod.vertices, &mut lod.indices, &mesh.vertices)?;

    let end = timestamp();

    simplify_report("SimplifyAttr", mesh, &lod, result_error, start, end);
    Ok(())
}

pub fn simplify_update(mesh: &Mesh, threshold: f32, options: u32) -> Result<()> {
    let start = timestamp();

    let target_index_count = (mesh.indices.len() as f32 * threshold) as usize;
    let target_error = 1e-2f32;
    let mut result_error = 0f32;

    let nrm_weight = 0.5f32;
    let attr_weights = [nrm_weight, nrm_weight, nrm_weight];

    // start from the original mesh
    let mut lod = mesh.clone();
    let mut data = floats(&lod.vertices);
    let count = simplify_with_update_interleaved(
        &mut lod.indices,
        mesh.indices.len(),
        &mut data,
        mesh.vertices.len(),
        VERTEX_SIZE,
        3,
        VERTEX_SIZE,
        &attr_weights,
        3,
        None,
        target_index_count,
        target_error,
        options,
        Some(&mut result_error),
    )?;
    lod.indices.truncate(count);

    let updated = from_bytes(&data.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    optimize_fetch(&mut lod.vertices, &mut lod.indices, &updated)?;

    for v in &mut lod.vertices {
        // update normals
        let nl = (v.nx * v.nx + v.ny * v.ny + v.nz * v.nz).sqrt();
        if nl > 0.0 {
            v.nx /= nl;
            v.ny /= nl;
            v.nz /= nl;
        }
    }

    let end = timestamp();

    simplify_report("SimplifyUpdt", mesh, &lod, result_error, start, end);
    Ok(())
}

pub fn simplify_sloppy_mesh(mesh: &Mesh, threshold: f32) -> Result<()> {
    let mut lod = Mesh::default();

    let start = timestamp();

    let target_index_count = (mesh.indices.len() as f32 * threshold) as usize;
    let target_error = 1e-1f32;
    let mut result_error = 0f32;

    let positions = floats(&mesh.vertices);

    // note: simplify needs space for index_count elements in the destination array, not target_index_count
    lod.indices = vec![0; mesh.indices.len()];
    let count = simplify_sloppy(
        &mut lod.indices,
        &mesh.indices,
        mesh.indices.len(),
        &positions,
        mesh.vertices.len(),
        VERTEX_SIZE,
        None,
        target_index_count,
        target_error,
        Some(&mut result_error),
    )?;
    lod.indices.truncate(count);

    optimize_fetch(&mut lod.vertices, &mut lod.indices, &mesh.vertices)?;

    let end = timestamp();

    simplify_report("SimplifyS", mesh, &lod, result_error, start, end);
    Ok(())
}

pub fn simplify_points_mesh(mesh: &Mesh, threshold: f32) -> Result<()> {
    let start = timestamp();

    let target_vertex_count = (mesh.vertices.len() as f32 * threshold) as usize;
    if target_vertex_count == 0 {
        return Ok(());
    }

    let positions = floats(&mesh.vertices);

    let mut indices = vec![0u32; target_vertex_count];
    let count = simplify_points(
        &mut indices,
        &positions,
        mesh.vertices.len(),
        VERTEX_SIZE,
        None,
        0,
        0.0,
        target_vertex_count,
    )?;
    indices.truncate(count);

    let end = timestamp();

    println!(
        "{:<9}: {} points => {} points in {} msec",
        "SimplifyP",
        mesh.vertices.len(),
        indices.len(),
        f((end - start) * 1000.0, 2)
    );
    Ok(())
}

pub fn simplify_complete(mesh: &Mesh) -> Result<()> {
    const LOD_COUNT: usize = 5;

    let start = timestamp();

    let positions = floats(&mesh.vertices);

    // generate 4 LOD levels (1-4), with each subsequent LOD using 70% triangles
    // note that each LOD uses the same (shared) vertex buffer
    let mut lods: Vec<Vec<u32>> = vec![Vec::new(); LOD_COUNT];

    lods[0] = mesh.indices.clone();

    for i in 1..LOD_COUNT {
        let threshold = 0.7f32.powf(i as f32);
        let mut target_index_count = (mesh.indices.len() as f32 * threshold) as usize / 3 * 3;
        let target_error = 1e-2f32;

        // we can simplify all the way from base level or from the last result
        // simplifying from the base level sometimes produces better results, but simplifying from last level is faster
        let source = lods[i - 1].clone();

        if source.len() < target_index_count {
            target_index_count = source.len();
        }

        let mut lod = vec![0u32; source.len()];
        let count = simplify(
            &mut lod,
            &source,
            source.len(),
            &positions,
            mesh.vertices.len(),
            VERTEX_SIZE,
            target_index_count,
            target_error,
            0,
            None,
        )?;
        lod.truncate(count);
        lods[i] = lod;
    }

    let middle = timestamp();

    // optimize each individual LOD for vertex cache & overdraw
    for lod in &mut lods {
        let source = lod.clone();
        optimize_vertex_cache(lod, &source, source.len(), mesh.vertices.len())?;
        let source = lod.clone();
        optimize_overdraw(lod, &source, source.len(), &positions, mesh.vertices.len(), VERTEX_SIZE, 1.0)?;
    }

    // concatenate all LODs into one IB
    // note: the order of concatenation is important - since we optimize the entire IB for vertex fetch,
    // putting coarse LODs first makes sure that the vertex range referenced by them is as small as possible
    // some GPUs process the entire range referenced by the index buffer region so doing this optimizes the vertex transform
    // cost for coarse LODs
    // this order also produces much better vertex fetch cache coherency for coarse LODs (since they're essentially optimized first)
    // somewhat surprisingly, the vertex fetch cache coherency for fine LODs doesn't seem to suffer that much.
    let mut lod_index_offsets = [0usize; LOD_COUNT];
    let mut lod_index_counts = [0usize; LOD_COUNT];
    let mut total_index_count = 0;

    for i in (0..LOD_COUNT).rev() {
        lod_index_offsets[i] = total_index_count;
        lod_index_counts[i] = lods[i].len();

        total_index_count += lods[i].len();
    }

    let mut indices = vec![0u32; total_index_count];

    for i in 0..LOD_COUNT {
        indices[lod_index_offsets[i]..lod_index_offsets[i] + lods[i].len()].copy_from_slice(&lods[i]);
    }

    // vertex fetch optimization should go last as it depends on the final index order
    // note that the order of LODs above affects vertex fetch results
    let mut vertices = Vec::new();
    optimize_fetch(&mut vertices, &mut indices, &mesh.vertices)?;

    let end = timestamp();

    println!(
        "{:<9}: {} triangles => {} LOD levels down to {} triangles in {} msec, optimized in {} msec",
        "SimplifyC",
        lod_index_counts[0] / 3,
        LOD_COUNT,
        lod_index_counts[LOD_COUNT - 1] / 3,
        f((middle - start) * 1000.0, 2),
        f((end - middle) * 1000.0, 2)
    );

    // for using LOD data at runtime, in addition to vertices and indices you have to save lod_index_offsets/lod_index_counts.

    {
        let lod0 = &indices[lod_index_offsets[0]..];
        let lod_n = &indices[lod_index_offsets[LOD_COUNT - 1]..];

        let vcs0 = analyze_vertex_cache(lod0, lod_index_counts[0], vertices.len(), 16, 0, 0)?;
        let vfs0 = analyze_vertex_fetch(lod0, lod_index_counts[0], vertices.len(), VERTEX_SIZE)?;
        let vcs_n = analyze_vertex_cache(lod_n, lod_index_counts[LOD_COUNT - 1], vertices.len(), 16, 0, 0)?;
        let vfs_n = analyze_vertex_fetch(lod_n, lod_index_counts[LOD_COUNT - 1], vertices.len(), VERTEX_SIZE)?;

        let pv = pack_mesh_oct(&vertices);
        let size = PACKED_VERTEX_OCT_SIZE;

        let mut vbuf = vec![0u8; encode_vertex_buffer_bound(vertices.len(), size)?];
        let len = vbuf.len();
        let vsize = encode_vertex_buffer(&mut vbuf, len, &pv, vertices.len(), size)?;
        vbuf.truncate(vsize);

        let mut ibuf = vec![0u8; encode_index_buffer_bound(indices.len(), vertices.len())?];
        let len = ibuf.len();
        let isize = encode_index_buffer(&mut ibuf, len, &indices, indices.len())?;
        ibuf.truncate(isize);

        println!(
            "{:<9}  ACMR {}...{} Overfetch {}..{} Codec VB {} bits/vertex IB {} bits/triangle",
            "",
            f(vcs0.acmr.into(), 6),
            f(vcs_n.acmr.into(), 6),
            f(vfs0.overfetch.into(), 6),
            f(vfs_n.overfetch.into(), 6),
            f(vbuf.len() as f64 / vertices.len() as f64 * 8.0, 1),
            f(ibuf.len() as f64 / (indices.len() / 3) as f64 * 8.0, 1)
        );
    }

    Ok(())
}

/// Meshlets of a build call, trimmed like the C++ examples (`meshlet_vertices` / `meshlet_triangles` resized to the last meshlet).
struct Meshlets {
    meshlets: Vec<Meshlet>,
    vertices: Vec<u32>,
    triangles: Vec<u8>,
}

impl Meshlets {
    fn new(index_count: usize, max_meshlets: usize) -> Meshlets {
        Meshlets {
            meshlets: vec![Meshlet::default(); max_meshlets],
            vertices: vec![0; index_count],
            triangles: vec![0; index_count],
        }
    }

    fn trim(&mut self, count: usize) {
        self.meshlets.truncate(count);

        if let Some(last) = self.meshlets.last() {
            // this is an example of how to trim the vertex/triangle arrays when copying data out to GPU storage
            self.vertices.truncate((last.vertex_offset + last.vertex_count) as usize);
            self.triangles.truncate((last.triangle_offset + last.triangle_count * 3) as usize);
        }
    }

    /// Mesh index of corner `k` of meshlet `m`.
    fn index(&self, m: &Meshlet, k: usize) -> u32 {
        self.vertices[m.vertex_offset as usize + self.triangles[m.triangle_offset as usize + k] as usize]
    }
}

pub fn simplify_clusters(mesh: &Mesh, threshold: f32) -> Result<()> {
    let max_vertices = 64;
    let max_triangles = 64;
    let target_group_size = 8;

    let start = timestamp();

    let positions = floats(&mesh.vertices);

    // build clusters (meshlets) out of the mesh
    let max_meshlets = build_meshlets_bound(mesh.indices.len(), max_vertices, max_triangles)?;
    let mut ml = Meshlets::new(mesh.indices.len(), max_meshlets);

    let count = build_meshlets(
        &mut ml.meshlets,
        &mut ml.vertices,
        &mut ml.triangles,
        &mesh.indices,
        mesh.indices.len(),
        &positions,
        mesh.vertices.len(),
        VERTEX_SIZE,
        max_vertices,
        max_triangles,
        0.0,
    )?;
    ml.meshlets.truncate(count);

    let middle = timestamp();

    // generate position remap; we'll use that to partition clusters using position-only adjacency
    let mut remap = vec![0u32; mesh.vertices.len()];
    generate_position_remap(&mut remap, &positions, mesh.vertices.len(), VERTEX_SIZE)?;

    // partition clusters in groups; each group will be simplified separately and the boundaries between groups will be preserved
    let mut cluster_indices = Vec::with_capacity(mesh.indices.len());
    let mut cluster_sizes = vec![0u32; ml.meshlets.len()];

    for (i, m) in ml.meshlets.iter().enumerate() {
        for j in 0..m.triangle_count as usize * 3 {
            // use the first vertex with equivalent position so that cluster adjacency ignores attribute seams
            cluster_indices.push(remap[ml.index(m, j) as usize]);
        }

        cluster_sizes[i] = m.triangle_count * 3;
    }

    let mut partition = vec![0u32; ml.meshlets.len()];
    let partition_count = partition_clusters(
        &mut partition,
        &cluster_indices,
        cluster_indices.len(),
        &cluster_sizes,
        cluster_sizes.len(),
        Some(&positions),
        mesh.vertices.len(),
        VERTEX_SIZE,
        target_group_size,
    )?;

    // convert partitions to linked lists to make it easier to iterate over (vectors of vectors would work too)
    let mut partnext = vec![-1i32; ml.meshlets.len()];
    let mut partlast = vec![-1i32; partition_count];

    for i in 0..ml.meshlets.len() {
        let part = partition[i] as usize;

        if partlast[part] >= 0 {
            partnext[partlast[part] as usize] = i as i32;
        }

        partlast[part] = i as i32;
        partnext[i] = -1;
    }

    let parttime = timestamp();

    let scale = simplify_scale(&positions, mesh.vertices.len(), VERTEX_SIZE)?;

    let mut lod: Vec<u32> = Vec::with_capacity(mesh.indices.len());

    let mut error = 0f32;

    for i in 0..ml.meshlets.len() {
        if partlast[partition[i] as usize] < 0 {
            continue; // part of a group that was already processed
        }

        // mark group as processed
        partlast[partition[i] as usize] = -1;

        let group_offset = lod.len();

        let mut j = i as i32;
        while j >= 0 {
            let m = &ml.meshlets[j as usize];

            for k in 0..m.triangle_count as usize * 3 {
                lod.push(ml.index(m, k));
            }

            j = partnext[j as usize];
        }

        let group_triangles = (lod.len() - group_offset) / 3;

        // simplify the group, preserving the border vertices
        // note: this technically also locks the exterior border; a full mesh analysis (see clusterlod.h / lockBoundary) would work better for some meshes
        let options = SIMPLIFY_LOCK_BORDER | SIMPLIFY_SPARSE | SIMPLIFY_ERROR_ABSOLUTE;

        let group_target_error = 1e-2 * scale;
        let group_target = (group_triangles as f32 * threshold) as usize * 3;
        let mut group_error = 0f32;
        let source = lod[group_offset..].to_vec();
        let group_size = simplify(
            &mut lod[group_offset..],
            &source,
            group_triangles * 3,
            &positions,
            mesh.vertices.len(),
            VERTEX_SIZE,
            group_target,
            group_target_error,
            options,
            Some(&mut group_error),
        )?;

        error = if group_error > error { group_error } else { error };

        // simplified group is available in lod[group_offset..group_offset + group_size]
        lod.truncate(group_offset + group_size);
    }

    let end = timestamp();

    println!(
        "{:<9}: {} triangles => {} triangles ({}% deviation) in {} msec, clusterized in {} msec, partitioned in {} msec ({} clusters in {} groups, {} avg)",
        "SimplifyG",
        mesh.indices.len() / 3,
        lod.len() / 3,
        f(f64::from(error / scale * 100.0), 2),
        f((end - parttime) * 1000.0, 2),
        f((middle - start) * 1000.0, 2),
        f((parttime - middle) * 1000.0, 2),
        ml.meshlets.len(),
        partition_count,
        f(ml.meshlets.len() as f64 / partition_count as f64, 1)
    );
    Ok(())
}

pub fn optimize(mesh: &Mesh, fifo: bool) -> Result<()> {
    let mut copy = mesh.clone();
    // note: we assume that the mesh is already optimally indexed (via parseObj); if that is not the case, you'd need to reindex first

    let start = timestamp();

    let positions = floats(&copy.vertices);
    let vertex_count = copy.vertices.len();
    let index_count = copy.indices.len();

    // vertex cache optimization should go first as it provides starting order for overdraw
    // note: fifo optimization is not recommended as a default, since it produces worse results, but it's faster to run so it can be useful for procedural meshes
    let source = copy.indices.clone();
    if fifo {
        optimize_vertex_cache_fifo(&mut copy.indices, &source, index_count, vertex_count, /* cache_size= */ 16)?;
    } else {
        optimize_vertex_cache(&mut copy.indices, &source, index_count, vertex_count)?;
    }

    // reorder indices for overdraw, balancing overdraw and vertex cache efficiency
    let k_threshold = 1.01f32; // allow up to 1% worse ACMR to get more reordering opportunities for overdraw
    let source = copy.indices.clone();
    optimize_overdraw(
        &mut copy.indices,
        &source,
        index_count,
        &positions,
        vertex_count,
        VERTEX_SIZE,
        k_threshold,
    )?;

    // vertex fetch optimization should go last as it depends on the final index order
    let source = copy.vertices.clone();
    optimize_fetch(&mut copy.vertices, &mut copy.indices, &source)?;

    let end = timestamp();

    let positions = floats(&copy.vertices);
    let vertex_count = copy.vertices.len();

    let vcs = analyze_vertex_cache(&copy.indices, index_count, vertex_count, 16, 0, 0)?;
    let vfs = analyze_vertex_fetch(&copy.indices, index_count, vertex_count, VERTEX_SIZE)?;
    let os = analyze_overdraw(&copy.indices, index_count, &positions, vertex_count, VERTEX_SIZE)?;

    let vcs_nv = analyze_vertex_cache(&copy.indices, index_count, vertex_count, 32, 32, 32)?;
    let vcs_amd = analyze_vertex_cache(&copy.indices, index_count, vertex_count, 14, 64, 128)?;
    let vcs_intel = analyze_vertex_cache(&copy.indices, index_count, vertex_count, 128, 0, 0)?;

    println!(
        "Optimize{}: ACMR {} ATVR {} (NV {} AMD {} Intel {}) overfetch {} overdraw {} in {} msec",
        if fifo { "F" } else { " " },
        f(vcs.acmr.into(), 6),
        f(vcs.atvr.into(), 6),
        f(vcs_nv.atvr.into(), 6),
        f(vcs_amd.atvr.into(), 6),
        f(vcs_intel.atvr.into(), 6),
        f(vfs.overfetch.into(), 6),
        f(os.overdraw.into(), 6),
        f((end - start) * 1000.0, 2)
    );
    Ok(())
}

/// Deflated size of `data` (C++ `compress`, sdefl at `SDEFL_LVL_DEF`).
pub fn compress(data: &[u8]) -> usize {
    let mut cbuf = vec![0u8; sdefl_bound(data.len())];
    let mut s = Sdefl::default();
    sdeflate(&mut s, &mut cbuf, data, SDEFL_LVL_DEF)
}

pub fn encode_index(indices: &[u32], vertex_count: usize, desc: char) -> Result<()> {
    // allocate result outside of the timing loop to exclude memset() from decode timing
    let mut result = vec![0u32; indices.len()];

    let start = timestamp();

    let mut buffer = vec![0u8; encode_index_buffer_bound(indices.len(), vertex_count)?];
    let len = buffer.len();
    let size = encode_index_buffer(&mut buffer, len, indices, indices.len())?;
    buffer.truncate(size);

    let middle = timestamp();

    let res = decode_index_buffer(&mut result, indices.len(), &buffer, buffer.len());
    assert!(res.is_ok());

    let end = timestamp();

    let csize = compress(&buffer);

    for (r, i) in result.as_chunks::<3>().0.iter().zip(indices.as_chunks::<3>().0) {
        assert!(
            (r[0] == i[0] && r[1] == i[1] && r[2] == i[2])
                || (r[1] == i[0] && r[2] == i[1] && r[0] == i[2])
                || (r[2] == i[0] && r[0] == i[1] && r[1] == i[2])
        );
    }

    let bytes = result.len() as f64 * 4.0;

    println!(
        "IdxCodec{desc}: {} bits/triangle (post-deflate {} bits/triangle); encode {} msec ({} GB/s), decode {} msec ({} GB/s)",
        f((buffer.len() * 8) as f64 / (indices.len() / 3) as f64, 1),
        f((csize * 8) as f64 / (indices.len() / 3) as f64, 1),
        f((middle - start) * 1000.0, 2),
        f((bytes / 1e9) / (middle - start), 3),
        f((end - middle) * 1000.0, 2),
        f((bytes / 1e9) / (end - middle), 2)
    );
    Ok(())
}

pub fn encode_index_mesh(mesh: &Mesh, desc: char) -> Result<()> {
    encode_index(&mesh.indices, mesh.vertices.len(), desc)
}

pub fn encode_index_sequence_data(data: &[u32], vertex_count: usize, desc: char) -> Result<()> {
    // allocate result outside of the timing loop to exclude memset() from decode timing
    let mut result = vec![0u32; data.len()];

    let start = timestamp();

    let mut buffer = vec![0u8; encode_index_sequence_bound(data.len(), vertex_count)];
    let len = buffer.len();
    let size = encode_index_sequence(&mut buffer, len, data, data.len())?;
    buffer.truncate(size);

    let middle = timestamp();

    let res = decode_index_sequence(&mut result, data.len(), &buffer, buffer.len());
    assert!(res.is_ok());

    let end = timestamp();

    let csize = compress(&buffer);

    assert!(data == &result[..]);

    let bytes = result.len() as f64 * 4.0;

    println!(
        "IdxCodec{desc}: {} bits/index (post-deflate {} bits/index); encode {} msec ({} GB/s), decode {} msec ({} GB/s)",
        f((buffer.len() * 8) as f64 / data.len() as f64, 1),
        f((csize * 8) as f64 / data.len() as f64, 1),
        f((middle - start) * 1000.0, 2),
        f((bytes / 1e9) / (middle - start), 3),
        f((end - middle) * 1000.0, 2),
        f((bytes / 1e9) / (end - middle), 2)
    );
    Ok(())
}

fn validate_decode_meshlet<V: IndexType + PartialEq, T: MeshletTriangleType + Default + Into<u32>>(
    data: &[u8],
    vertices: &[u32],
    vertex_count: usize,
    triangles: &[u8],
    triangle_count: usize,
) {
    let mut rv = [V::default(); 256];
    let mut rt = [T::default(); 256 * 3];

    let rc = decode_meshlet(&mut rv, vertex_count, &mut rt, triangle_count, data, data.len());
    assert!(rc.is_ok());

    for j in 0..vertex_count {
        assert!(rv[j] == V::from_u32(vertices[j]));
    }

    for j in 0..triangle_count {
        let (a, b, c) = (
            u32::from(triangles[j * 3]),
            u32::from(triangles[j * 3 + 1]),
            u32::from(triangles[j * 3 + 2]),
        );

        let tri = if T::PER_TRIANGLE == 3 {
            rt[j * 3].into() | (rt[j * 3 + 1].into() << 8) | (rt[j * 3 + 2].into() << 16)
        } else {
            rt[j].into()
        };

        let abc = a | (b << 8) | (c << 16);
        let bca = b | (c << 8) | (a << 16);
        let cba = c | (a << 8) | (b << 16);

        assert!(tri == abc || tri == bca || tri == cba);
    }
}

pub fn encode_meshlets(mesh: &Mesh, max_vertices: usize, max_triangles: usize, reorder: bool, level: i32) -> Result<()> {
    let positions = floats(&mesh.vertices);

    let max_meshlets = build_meshlets_bound(mesh.indices.len(), max_vertices, max_triangles)?;
    let mut ml = Meshlets::new(mesh.indices.len(), max_meshlets);

    let count = build_meshlets(
        &mut ml.meshlets,
        &mut ml.vertices,
        &mut ml.triangles,
        &mesh.indices,
        mesh.indices.len(),
        &positions,
        mesh.vertices.len(),
        VERTEX_SIZE,
        max_vertices,
        max_triangles,
        0.0,
    )?;
    ml.trim(count);

    let mut cbuf = vec![0u8; encode_meshlet_bound(max_vertices, max_triangles)];

    // optimize each meshlet for locality; this is important for performance, and critical for good compression
    for m in &ml.meshlets {
        optimize_meshlet_level(
            &mut ml.vertices[m.vertex_offset as usize..],
            m.vertex_count as usize,
            &mut ml.triangles[m.triangle_offset as usize..],
            m.triangle_count as usize,
            level,
        )?;
    }

    // optimize the order of vertex references within each meshlet and globally; this is valuable for access locality and critical for compression of vertex references
    // note that this reorders the vertex buffer too, so if a traditional index buffer is required it would need to be reconstructed from the meshlet data for optimal locality
    if reorder {
        let mut vertices = Vec::new();
        optimize_fetch(&mut vertices, &mut ml.vertices, &mesh.vertices)?;
    }

    let mut mbst = 0;

    let mut packed = Vec::new();

    for m in &ml.meshlets {
        let mv = &ml.vertices[m.vertex_offset as usize..];
        let mt = &ml.triangles[m.triangle_offset as usize..];
        let (vc, tc) = (m.vertex_count as usize, m.triangle_count as usize);

        let len = cbuf.len();
        let mbs = encode_meshlet(&mut cbuf, len, mv, vc, mt, tc)?;
        assert!(mbs > 0);

        // 24-bit header: 7 bit (vertex_count-1), 7 bit (triangle_count-1), 10 bit size
        // fits up to 128v/128t meshlet with 1024 bytes of encoded data; meshopt_encodeMeshletBound(128,128) < 1000
        assert!(vc - 1 < 128 && tc - 1 < 128 && mbs < 1024);
        let header = ((m.vertex_count - 1) & 0x7f) | (((m.triangle_count - 1) & 0x7f) << 7) | ((mbs as u32 & 0x3ff) << 14);
        packed.extend_from_slice(&header.to_le_bytes()[..3]);
        packed.extend_from_slice(&cbuf[..mbs]);

        validate_decode_meshlet::<u32, u32>(&cbuf[..mbs], mv, vc, mt, tc);
        validate_decode_meshlet::<u32, u8>(&cbuf[..mbs], mv, vc, mt, tc);
        validate_decode_meshlet::<u16, u32>(&cbuf[..mbs], mv, vc, mt, tc);
        validate_decode_meshlet::<u16, u8>(&cbuf[..mbs], mv, vc, mt, tc);

        mbst += mbs;
    }

    let mbc = compress(&packed);
    let meshlet_count = ml.meshlets.len();
    let tris = (mesh.indices.len() / 3) as f64;

    println!(
        "MeshletCodec ({max_vertices}/{max_triangles}): {} meshlets, {} bytes/meshlet; {} bytes, {} bits/triangle",
        meshlet_count,
        mbst / meshlet_count,
        mbst,
        f((mbst * 8) as f64 / tris, 1)
    );
    println!(
        "MeshletCodec ({max_vertices}/{max_triangles}, packed): {} bytes/meshlet, {} bits/triangle; post-deflate: {} bytes/meshlet, {} bits/triangle)",
        packed.len() / meshlet_count,
        f((packed.len() * 8) as f64 / tris, 1),
        mbc / meshlet_count,
        f((mbc * 8) as f64 / tris, 1)
    );

    let mut mbtime = 0f64;

    for _ in 0..10 {
        let mut rv = [0u32; 256];
        let mut rt = [0u32; 256];
        let t0 = timestamp();
        let mut p = 0;
        for _ in 0..meshlet_count {
            let header = u32::from(packed[p]) | (u32::from(packed[p + 1]) << 8) | (u32::from(packed[p + 2]) << 16);
            let vertex_count = (header & 0x7f) as usize + 1;
            let triangle_count = ((header >> 7) & 0x7f) as usize + 1;
            let size = ((header >> 14) & 0x3ff) as usize;
            let _ = decode_meshlet_raw(&mut rv, vertex_count, &mut rt, triangle_count, &packed[p + 3..], size);
            p += 3 + size;
        }
        let t1 = timestamp();

        mbtime = if mbtime == 0.0 || t1 - t0 < mbtime { t1 - t0 } else { mbtime };
    }

    println!(
        "MeshletCodec ({max_vertices}/{max_triangles}, packed): decode time {} msec, {}B tri/sec, {} ns/meshlet",
        f(mbtime * 1000.0, 3),
        f(tris / 1e9 / mbtime, 3),
        f(mbtime * 1e9 / meshlet_count as f64, 1)
    );
    Ok(())
}

fn write_bits(data: &mut Vec<u8>, acc: &mut u32, acc_bits: &mut i32, value: u32, bits: i32) {
    assert!(u64::from(value) < (1u64 << bits));
    assert!(*acc_bits + bits <= 32);

    *acc |= value << *acc_bits;
    *acc_bits += bits;

    while *acc_bits >= 8 {
        data.push((*acc & 0xff) as u8);
        *acc >>= 8;
        *acc_bits -= 8;
    }
}

/// C `roundf`: halfway cases away from zero.
fn roundf(v: f32) -> f32 {
    v.round()
}

/// C `ldexpf(1.f, e)` for the exponent range used here.
fn ldexpf1(e: i32) -> f32 {
    2f32.powi(e)
}

pub fn encode_meshlets_dxr(mesh: &Mesh, max_triangles: usize, min_exp: i32, level: i32) -> Result<()> {
    // split mesh into clusters, optimized for RT
    let min_triangles = max_triangles / 4;
    let max_vertices = 256;
    let fill_weight = 0.25f32;
    let max_meshlets = build_meshlets_bound(mesh.indices.len(), max_vertices, min_triangles)?;

    let positions = floats(&mesh.vertices);
    let data = bytes(&mesh.vertices);

    let mut shadow_indices = vec![0u32; mesh.indices.len()];
    generate_shadow_index_buffer(
        &mut shadow_indices,
        &mesh.indices,
        mesh.indices.len(),
        &data,
        mesh.vertices.len(),
        12,
        VERTEX_SIZE,
    )?;

    let mut ml = Meshlets::new(mesh.indices.len(), max_meshlets);
    let count = build_meshlets_spatial(
        &mut ml.meshlets,
        &mut ml.vertices,
        &mut ml.triangles,
        &shadow_indices,
        mesh.indices.len(),
        &positions,
        mesh.vertices.len(),
        VERTEX_SIZE,
        max_vertices,
        min_triangles,
        max_triangles,
        fill_weight,
    )?;
    ml.trim(count);

    // optimize each meshlet for locality; this is critical for good compression
    for m in &ml.meshlets {
        optimize_meshlet_level(
            &mut ml.vertices[m.vertex_offset as usize..],
            m.vertex_count as usize,
            &mut ml.triangles[m.triangle_offset as usize..],
            m.triangle_count as usize,
            level,
        )?;
    }

    let pos = |v: u32| -> [f32; 3] {
        let v = &mesh.vertices[v as usize];
        [v.px, v.py, v.pz]
    };

    // compute shared exponent per mesh:
    // - must be at most min_exp which should be set to maximum required precision (e.g. -10 for 1mm accuracy given metric units)
    // - must accommodate each cluster to fit into Compressed1 DXR encoding (16-bit offsets from 24-bit anchors)
    // note, for DGF parity min_exp should be set based on mesh AABB; we do not do this as we do not consider this rigorous.
    let mut exponent = min_exp;

    for m in &ml.meshlets {
        let mut minv = [f32::MAX; 3];
        let mut maxv = [-f32::MAX; 3];

        for j in 0..m.vertex_count as usize {
            let p = pos(ml.vertices[m.vertex_offset as usize + j]);

            for k in 0..3 {
                minv[k] = minv[k].min(p[k]);
                maxv[k] = maxv[k].max(p[k]);
            }
        }

        let cexp = compute_position_exponent(&minv, &maxv, min_exp, /* max_bits= */ 16)?;
        exponent = if exponent > cexp { exponent } else { cexp };
    }

    let mut packed = Vec::new();
    let (mut stats_headers, mut stats_positions, mut stats_topology) = (0usize, 0usize, 0usize);

    for m in &ml.meshlets {
        // for each cluster, use meshopt codec for meshlet topology; will need to decode using meshopt_decodeMeshlet at runtime
        let mut topology = [0u8; 1024];
        assert!(encode_meshlet_bound(0, max_triangles) <= topology.len());

        let tops = encode_meshlet(
            &mut topology,
            1024,
            &[],
            0,
            &ml.triangles[m.triangle_offset as usize..],
            m.triangle_count as usize,
        )?;
        assert!(tops > 0);

        // for each cluster, positions are encoded using 24-bit anchor and up-to-16-bit offset per axis
        let scale = ldexpf1(exponent);

        let mut cpos = [[0i32; 3]; 256];
        let mut anchor = [i32::MAX; 3];

        for (j, cp) in cpos[..m.vertex_count as usize].iter_mut().enumerate() {
            let p = pos(ml.vertices[m.vertex_offset as usize + j]);

            for k in 0..3 {
                cp[k] = roundf(p[k] / scale) as i32;
                anchor[k] = if anchor[k] < cp[k] { anchor[k] } else { cp[k] };
            }
        }

        // compute optimal per-cluster per-axis bit count
        let mut bits = [1i32; 3];

        for p in &cpos[..m.vertex_count as usize] {
            for k in 0..3 {
                let v = p[k] - anchor[k];
                assert!((0..1 << 16).contains(&v));

                while v >= (1 << bits[k]) {
                    bits[k] += 1;
                }
            }
        }

        // prepare cluster header: 1 word with cluster sizes, and 3 words for DXR-compatible cluster position header
        // see https://microsoft.github.io/DirectX-Specs/d3d/Raytracing2.html#compressed1-position-encoding
        let header: [u32; 4] = [
            (m.vertex_count - 1) | ((m.triangle_count - 1) << 8) | ((tops as u32) << 16),
            ((anchor[0] as u32) << 8) | (exponent + 127) as u32,
            ((anchor[1] as u32) << 8) | (((bits[1] - 1) as u32) << 4) | (bits[0] - 1) as u32,
            ((anchor[2] as u32) << 8) | (bits[2] - 1) as u32,
        ];

        // write cluster header, cluster positions, and cluster topology
        // note: we write the data in a way that is easiest to consume for the decoding; however, for best post-deflate compression consider reordering it
        // (by grouping headers, positions, and topology into three separate streams)
        // maximum post-deflate compression can be achieved by interleaving the cluster position bits per axis (all X => all Y => all Z), but that requires extra decoding work
        // this layout can be decoded using `meshopt_decodeMeshlet` + `memcpy` when the consumer is DXR2-compatible.
        for word in header {
            packed.extend_from_slice(&word.to_le_bytes());
        }

        let mut acc = 0u32;
        let mut acc_bits = 0i32;

        for p in &cpos[..m.vertex_count as usize] {
            for k in 0..3 {
                write_bits(&mut packed, &mut acc, &mut acc_bits, (p[k] - anchor[k]) as u32, bits[k]);
            }
        }

        if acc_bits != 0 {
            packed.push((acc & 0xff) as u8);
        }

        packed.extend_from_slice(&topology[..tops]);

        stats_headers += 4 * 4;
        stats_positions += (m.vertex_count as usize * (bits[0] + bits[1] + bits[2]) as usize).div_ceil(8);
        stats_topology += tops;
    }

    let mbc = compress(&packed);
    let tris = (mesh.indices.len() / 3) as f64;

    println!(
        "MeshletCodecDXR ({max_triangles}@{exponent}): {} meshlets, {} bytes; {} bytes/triangle ({} headers, {} positions, {} topology); post-deflate: {} bytes/triangle",
        ml.meshlets.len(),
        packed.len(),
        f(packed.len() as f64 / tris, 1),
        f(stats_headers as f64 / tris, 1),
        f(stats_positions as f64 / tris, 1),
        f(stats_topology as f64 / tris, 1),
        f(mbc as f64 / tris, 1)
    );

    let mut mbtime = 0f64;
    let mut mbsize = 0usize;

    for i in 0..10 {
        let mut clp = [0u8; 12 + 256 * 6]; // up to 48 bits per vertex
        let mut clt = [0u8; 256 * 3 + 4]; // +4 bytes for padding

        let t0 = timestamp();
        let mut p = 0;
        for _ in 0..ml.meshlets.len() {
            // decode cluster header
            let triangle_count = usize::from(packed[p + 1]) + 1;
            let vertex_count = usize::from(packed[p]) + 1;
            let topology_size = usize::from(packed[p + 2]) | (usize::from(packed[p + 3]) << 8);

            // decode DXR header to establish position size
            let position_bitsx = usize::from(packed[p + 8] & 15) + 1;
            let position_bitsy = usize::from((packed[p + 8] >> 4) & 15) + 1;
            let position_bitsz = usize::from(packed[p + 12] & 15) + 1;
            let position_size = (vertex_count * (position_bitsx + position_bitsy + position_bitsz)).div_ceil(8);

            // memcpy DXR Compressed1 portion as is
            clp[..12 + position_size].copy_from_slice(&packed[p + 4..p + 16 + position_size]);

            // decode topology as 3 bytes per triangle
            let topology = &packed[p + 16 + position_size..];
            let rc = decode_meshlet::<u32, u8>(&mut [], 0, &mut clt, triangle_count, topology, topology_size);
            assert!(rc.is_ok());

            p += 16 + position_size + topology_size;
            if i == 0 {
                // stats for decode throughput
                mbsize += position_size + triangle_count * 3;
            }
        }
        let t1 = timestamp();

        mbtime = if mbtime == 0.0 || t1 - t0 < mbtime { t1 - t0 } else { mbtime };
    }

    println!(
        "MeshletCodecDXR ({max_triangles}): decode time {} msec, {}B tri/sec, {} GB/sec",
        f(mbtime * 1000.0, 3),
        f(tris / 1e9 / mbtime, 3),
        f(mbsize as f64 / mbtime * 1e-9, 3)
    );
    Ok(())
}

fn pack_vertex(mesh: &Mesh, packing: Packing, pvn: &str) {
    let pv = packing.pack(&mesh.vertices);

    let csize = compress(&pv);

    println!(
        "VtxPack{pvn}  : {} bits/vertex (post-deflate {} bits/vertex)",
        f((pv.len() * 8) as f64 / mesh.vertices.len() as f64, 1),
        f((csize * 8) as f64 / mesh.vertices.len() as f64, 1)
    );
}

fn encode_vertex(mesh: &Mesh, packing: Packing, pvn: &str, level: i32) -> Result<()> {
    let size = packing.size();
    let pv = packing.pack(&mesh.vertices);

    // allocate result outside of the timing loop to exclude memset() from decode timing
    let mut result = vec![0u8; pv.len()];

    let start = timestamp();

    let mut vbuf = vec![0u8; encode_vertex_buffer_bound(mesh.vertices.len(), size)?];
    let len = vbuf.len();
    let vsize = encode_vertex_buffer_level(&mut vbuf, len, &pv, mesh.vertices.len(), size, level, -1)?;
    vbuf.truncate(vsize);

    let middle = timestamp();

    let res = decode_vertex_buffer(&mut result, mesh.vertices.len(), size, &vbuf, vbuf.len());
    assert!(res.is_ok());

    let end = timestamp();

    assert!(pv == result);

    let csize = compress(&vbuf);

    let bytes = result.len() as f64;

    println!(
        "VtxCodec{pvn:>1}: {} bits/vertex (post-deflate {} bits/vertex); encode {} msec ({} GB/s), decode {} msec ({} GB/s)",
        f((vbuf.len() * 8) as f64 / mesh.vertices.len() as f64, 1),
        f((csize * 8) as f64 / mesh.vertices.len() as f64, 1),
        f((middle - start) * 1000.0, 2),
        f((bytes / 1e9) / (middle - start), 3),
        f((end - middle) * 1000.0, 2),
        f((bytes / 1e9) / (end - middle), 2)
    );
    Ok(())
}

pub fn stripify_mesh(mesh: &Mesh, use_restart: bool, desc: char) -> Result<()> {
    let restart_index = if use_restart { !0u32 } else { 0 };

    // note: input mesh is assumed to be optimized for vertex cache and vertex fetch
    let start = timestamp();
    let mut strip = vec![0u32; stripify_bound(mesh.indices.len())?];
    let count = stripify(&mut strip, &mesh.indices, mesh.indices.len(), mesh.vertices.len(), restart_index)?;
    strip.truncate(count);
    let end = timestamp();

    let restarts = strip.iter().filter(|&&v| use_restart && v == restart_index).count();

    // C++ resizes the copy to the unstripified count but then analyzes mesh.indices.size() elements of it
    let mut copy = mesh.clone();
    copy.indices.resize(unstripify_bound(strip.len())?.max(copy.indices.len()), 0);
    let count = unstripify(&mut copy.indices, &strip, strip.len(), restart_index)?;
    assert!(count <= unstripify_bound(strip.len())?);

    let vc = mesh.vertices.len();
    let vcs = analyze_vertex_cache(&copy.indices, mesh.indices.len(), vc, 16, 0, 0)?;
    let vcs_nv = analyze_vertex_cache(&copy.indices, mesh.indices.len(), vc, 32, 32, 32)?;
    let vcs_amd = analyze_vertex_cache(&copy.indices, mesh.indices.len(), vc, 14, 64, 128)?;
    let vcs_intel = analyze_vertex_cache(&copy.indices, mesh.indices.len(), vc, 128, 0, 0)?;

    println!(
        "Stripify{desc}: ACMR {} ATVR {} (NV {} AMD {} Intel {}); {} run avg, {} strip indices ({}%) in {} msec",
        f(vcs.acmr.into(), 6),
        f(vcs.atvr.into(), 6),
        f(vcs_nv.atvr.into(), 6),
        f(vcs_amd.atvr.into(), 6),
        f(vcs_intel.atvr.into(), 6),
        f(
            if use_restart {
                (strip.len() - restarts) as f64 / (restarts + 1) as f64
            } else {
                0.0
            },
            1
        ),
        strip.len(),
        f(strip.len() as f64 / mesh.indices.len() as f64 * 100.0, 1),
        f((end - start) * 1000.0, 2)
    );
    Ok(())
}

pub fn shadow(mesh: &Mesh) -> Result<()> {
    // note: input mesh is assumed to be optimized for vertex cache and vertex fetch

    let data = bytes(&mesh.vertices);

    let start = timestamp();
    // this index buffer can be used for position-only rendering using the same vertex data that the original index buffer uses
    let mut shadow_indices = vec![0u32; mesh.indices.len()];
    generate_shadow_index_buffer(
        &mut shadow_indices,
        &mesh.indices,
        mesh.indices.len(),
        &data,
        mesh.vertices.len(),
        12,
        VERTEX_SIZE,
    )?;
    let end = timestamp();

    // while you can't optimize the vertex data after shadow IB was constructed, you can and should optimize the shadow IB for vertex cache
    // this is valuable even if the original indices array was optimized for vertex cache!
    let source = shadow_indices.clone();
    optimize_vertex_cache(&mut shadow_indices, &source, source.len(), mesh.vertices.len())?;

    let vcs = analyze_vertex_cache(&mesh.indices, mesh.indices.len(), mesh.vertices.len(), 16, 0, 0)?;
    let vcss = analyze_vertex_cache(&shadow_indices, shadow_indices.len(), mesh.vertices.len(), 16, 0, 0)?;

    let mut shadow_flags = vec![0u8; mesh.vertices.len()];
    let mut shadow_vertices = 0usize;

    for &index in &shadow_indices {
        shadow_vertices += usize::from(1 - shadow_flags[index as usize]);
        shadow_flags[index as usize] = 1;
    }

    println!(
        "ShadowIB : ACMR {} ({}x improvement); {} shadow vertices ({}x improvement) in {} msec",
        f(vcss.acmr.into(), 6),
        f(f64::from(vcs.vertices_transformed) / f64::from(vcss.vertices_transformed), 2),
        shadow_vertices,
        f(mesh.vertices.len() as f64 / shadow_vertices as f64, 2),
        f((end - start) * 1000.0, 2)
    );
    Ok(())
}

fn follow(parents: &mut [i32], mut index: i32) -> i32 {
    while index != parents[index as usize] {
        let parent = parents[index as usize];
        parents[index as usize] = parents[parent as usize];
        index = parent;
    }

    index
}

pub fn meshlets(mesh: &Mesh, scan: bool, uniform: bool, flex: bool, spatial: bool, dump: bool) -> Result<()> {
    // NVidia recommends 64/126; we also test uniform configuration with 64/64 which is better for earlier AMD GPUs
    let max_vertices = 64;
    let max_triangles = if uniform { 64 } else { 126 };
    let min_triangles = if spatial {
        16
    } else if uniform {
        24
    } else {
        32
    }; // only used in flex/spatial modes

    // note: should be set to 0 unless cone culling is used at runtime!
    let cone_weight = 0.25f32;
    let split_factor = if flex { 2.0f32 } else { 0.0 };

    let positions = floats(&mesh.vertices);
    let (ic, vc) = (mesh.indices.len(), mesh.vertices.len());

    // note: input mesh is assumed to be optimized for vertex cache and vertex fetch
    let start = timestamp();
    let max_meshlets = build_meshlets_bound(ic, max_vertices, min_triangles)?;
    let mut ml = Meshlets::new(ic, max_meshlets);

    let count = if scan {
        build_meshlets_scan(
            &mut ml.meshlets,
            &mut ml.vertices,
            &mut ml.triangles,
            &mesh.indices,
            ic,
            vc,
            max_vertices,
            max_triangles,
        )?
    } else if flex {
        build_meshlets_flex(
            &mut ml.meshlets,
            &mut ml.vertices,
            &mut ml.triangles,
            &mesh.indices,
            ic,
            &positions,
            vc,
            VERTEX_SIZE,
            max_vertices,
            min_triangles,
            max_triangles,
            cone_weight,
            split_factor,
        )?
    } else if spatial {
        build_meshlets_spatial(
            &mut ml.meshlets,
            &mut ml.vertices,
            &mut ml.triangles,
            &mesh.indices,
            ic,
            &positions,
            vc,
            VERTEX_SIZE,
            max_vertices,
            min_triangles,
            max_triangles,
            0.0,
        )?
    } else {
        // note: equivalent to the call of buildMeshletsFlex() with split_factor = 0 and min_triangles = max_triangles
        build_meshlets(
            &mut ml.meshlets,
            &mut ml.vertices,
            &mut ml.triangles,
            &mesh.indices,
            ic,
            &positions,
            vc,
            VERTEX_SIZE,
            max_vertices,
            max_triangles,
            cone_weight,
        )?
    };
    ml.meshlets.truncate(count);

    if !dump {
        for m in &ml.meshlets {
            optimize_meshlet(
                &mut ml.vertices[m.vertex_offset as usize..],
                &mut ml.triangles[m.triangle_offset as usize..],
                m.triangle_count as usize,
                m.vertex_count as usize,
            )?;
        }
    }

    ml.trim(count);

    let end = timestamp();

    if dump {
        dump_obj(&mesh.vertices, &[], false);
    }

    let mut avg_vertices = 0f64;
    let mut avg_triangles = 0f64;
    let mut avg_boundary = 0f64;
    let mut avg_connected = 0f64;
    let mut not_full = 0usize;

    let mut boundary = vec![0i32; vc];

    for m in &ml.meshlets {
        for j in 0..m.vertex_count as usize {
            boundary[ml.vertices[m.vertex_offset as usize + j] as usize] += 1;
        }
    }

    let mut cluster = Vec::new();

    for (i, m) in ml.meshlets.iter().enumerate() {
        if dump {
            cluster.clear();
            for j in 0..m.triangle_count as usize * 3 {
                cluster.push(ml.index(m, j));
            }

            dump_obj_section(&format!("ml_{i}"), &cluster);
        }

        avg_vertices += f64::from(m.vertex_count);
        avg_triangles += f64::from(m.triangle_count);
        not_full += usize::from(if uniform {
            (m.triangle_count as usize) < max_triangles
        } else {
            (m.vertex_count as usize) < max_vertices
        });

        for j in 0..m.vertex_count as usize {
            if boundary[ml.vertices[m.vertex_offset as usize + j] as usize] > 1 {
                avg_boundary += 1.0;
            }
        }

        // union-find vertices to check if the meshlet is connected
        let mut parents = [0i32; 256];
        for (j, p) in parents[..m.vertex_count as usize].iter_mut().enumerate() {
            *p = j as i32;
        }

        let tri = |j: usize| i32::from(ml.triangles[m.triangle_offset as usize + j]);

        for j in 0..m.triangle_count as usize * 3 {
            let mut v0 = tri(j);
            let mut v1 = tri(if j % 3 == 2 { j - 2 } else { j + 1 });

            v0 = follow(&mut parents, v0);
            v1 = follow(&mut parents, v1);

            parents[v0 as usize] = v1;
        }

        let mut roots = 0;
        for j in 0..m.vertex_count as i32 {
            roots += i32::from(follow(&mut parents, j) == j);
        }

        assert!(roots != 0);
        avg_connected += f64::from(roots);
    }

    let n = ml.meshlets.len() as f64;
    avg_vertices /= n;
    avg_triangles /= n;
    avg_boundary /= n;
    avg_connected /= n;

    let kind = if scan {
        'S'
    } else if flex {
        'F'
    } else if spatial {
        'X'
    } else if uniform {
        'U'
    } else {
        ' '
    };

    println!(
        "Meshlets{kind}: {} meshlets (avg vertices {}, avg triangles {}, avg boundary {}, avg connected {}, not full {}) in {} msec",
        ml.meshlets.len(),
        f(avg_vertices, 1),
        f(avg_triangles, 1),
        f(avg_boundary, 1),
        f(avg_connected, 2),
        not_full,
        f((end - start) * 1000.0, 2)
    );

    let camera = [100f32, 100.0, 100.0];

    let mut rejected = 0usize;
    let mut accepted = 0usize;
    let mut radius_mean = 0f64;
    let mut cone_mean = 0f64;

    let mut radii = vec![0f32; ml.meshlets.len()];
    let mut cones = vec![0f32; ml.meshlets.len()];

    let startc = timestamp();
    for (i, m) in ml.meshlets.iter().enumerate() {
        let bounds = compute_meshlet_bounds(
            &ml.vertices[m.vertex_offset as usize..],
            &ml.triangles[m.triangle_offset as usize..],
            m.triangle_count as usize,
            &positions,
            vc,
            VERTEX_SIZE,
        )?;

        radii[i] = bounds.radius;
        // C++ uses this approximation of pi; the printed angles depend on it
        #[allow(clippy::approx_constant)]
        let deg = 180.0 / 3.1415926;
        cones[i] = 90.0 - bounds.cone_cutoff.acos() * deg;

        radius_mean += f64::from(radii[i]);
        cone_mean += f64::from(cones[i]);

        // trivial accept: we can't ever backface cull this meshlet
        accepted += usize::from(bounds.cone_cutoff >= 1.0);

        // perspective projection: dot(normalize(cone_apex - camera_position), cone_axis) > cone_cutoff
        // alternative formulation for perspective projection that doesn't use apex (and uses cluster bounding sphere instead):
        // dot(normalize(center - camera_position), cone_axis) > cone_cutoff + radius / length(center - camera_position)
        let cview = [
            bounds.center[0] - camera[0],
            bounds.center[1] - camera[1],
            bounds.center[2] - camera[2],
        ];
        let cviewlength = (cview[0] * cview[0] + cview[1] * cview[1] + cview[2] * cview[2]).sqrt();

        rejected += usize::from(
            cview[0] * bounds.cone_axis[0] + cview[1] * bounds.cone_axis[1] + cview[2] * bounds.cone_axis[2]
                >= bounds.cone_cutoff * cviewlength + bounds.radius,
        );
    }
    let endc = timestamp();

    radius_mean /= n;
    cone_mean /= n;

    let mut radius_variance = 0f64;

    for &r in &radii {
        radius_variance += (f64::from(r) - radius_mean) * (f64::from(r) - radius_mean);
    }

    radius_variance /= (ml.meshlets.len().wrapping_sub(1)) as f64;

    let radius_stddev = radius_variance.sqrt();

    let meshlets_std = radii.iter().filter(|&&r| f64::from(r) < radius_mean + radius_stddev).count();

    println!(
        "Bounds   : radius mean {} stddev {}; {}% meshlets under 1σ; cone angle {}°; cone reject {}% trivial accept {}% in {} msec",
        f(radius_mean, 6),
        f(radius_stddev, 6),
        f(meshlets_std as f64 / n * 100.0, 1),
        f(cone_mean, 1),
        f(rejected as f64 / n * 100.0, 1),
        f(accepted as f64 / n * 100.0, 1),
        f((endc - startc) * 1000.0, 2)
    );
    Ok(())
}

pub fn spatial_sort(mesh: &Mesh) -> Result<()> {
    let size = PACKED_VERTEX_OCT_SIZE;
    let pv = pack_mesh_oct(&mesh.vertices);
    let positions = floats(&mesh.vertices);
    let vc = mesh.vertices.len();

    let start = timestamp();

    let mut remap = vec![0u32; vc];
    spatial_sort_remap(&mut remap, &positions, vc, VERTEX_SIZE)?;

    let end = timestamp();

    let mut sorted = vec![0u8; pv.len()];
    remap_vertex_buffer(&mut sorted, &pv, vc, size, &remap)?;

    let mut vbuf = vec![0u8; encode_vertex_buffer_bound(vc, size)?];
    let len = vbuf.len();
    let vsize = encode_vertex_buffer(&mut vbuf, len, &sorted, vc, size)?;
    vbuf.truncate(vsize);

    let csize = compress(&vbuf);

    println!(
        "Spatial  : {} bits/vertex (post-deflate {} bits/vertex); sort {} msec",
        f((vbuf.len() * 8) as f64 / vc as f64, 1),
        f((csize * 8) as f64 / vc as f64, 1),
        f((end - start) * 1000.0, 2)
    );
    Ok(())
}

pub fn spatial_sort_triangles_mesh(mesh: &Mesh) -> Result<()> {
    let size = PACKED_VERTEX_OCT_SIZE;
    let mut copy = mesh.clone();
    let positions = floats(&copy.vertices);
    let (ic, vc) = (mesh.indices.len(), mesh.vertices.len());

    let start = timestamp();

    let source = copy.indices.clone();
    spatial_sort_triangles(&mut copy.indices, &source, ic, &positions, vc, VERTEX_SIZE)?;

    let end = timestamp();

    let source = copy.indices.clone();
    optimize_vertex_cache(&mut copy.indices, &source, ic, vc)?;
    let source = copy.vertices.clone();
    optimize_fetch(&mut copy.vertices, &mut copy.indices, &source)?;

    let pv = pack_mesh_oct(&copy.vertices);

    let mut vbuf = vec![0u8; encode_vertex_buffer_bound(vc, size)?];
    let len = vbuf.len();
    let vsize = encode_vertex_buffer(&mut vbuf, len, &pv, vc, size)?;
    vbuf.truncate(vsize);

    let mut ibuf = vec![0u8; encode_index_buffer_bound(ic, vc)?];
    let len = ibuf.len();
    let isize = encode_index_buffer(&mut ibuf, len, &copy.indices, ic)?;
    ibuf.truncate(isize);

    let csizev = compress(&vbuf);
    let csizei = compress(&ibuf);

    println!(
        "SpatialT : {} bits/vertex (post-deflate {} bits/vertex); {} bits/triangle (post-deflate {} bits/triangle); sort {} msec",
        f((vbuf.len() * 8) as f64 / vc as f64, 1),
        f((csizev * 8) as f64 / vc as f64, 1),
        f((ibuf.len() * 8) as f64 / (ic / 3) as f64, 1),
        f((csizei * 8) as f64 / (ic / 3) as f64, 1),
        f((end - start) * 1000.0, 2)
    );
    Ok(())
}

pub fn spatial_cluster_points_mesh(mesh: &Mesh, cluster_size: usize) -> Result<()> {
    let size = PACKED_VERTEX_OCT_SIZE;
    let positions = floats(&mesh.vertices);
    let vc = mesh.vertices.len();

    let start = timestamp();

    let mut index = vec![0u32; vc];
    spatial_cluster_points(&mut index, &positions, vc, VERTEX_SIZE, cluster_size)?;

    let end = timestamp();

    let pv = pack_mesh_oct(&mesh.vertices);

    let pvo: Vec<u8> = index
        .iter()
        .flat_map(|&i| pv[i as usize * size..(i as usize + 1) * size].iter().copied())
        .collect();

    let mut vbuf = vec![0u8; encode_vertex_buffer_bound(vc, size)?];
    let len = vbuf.len();
    let vsize = encode_vertex_buffer(&mut vbuf, len, &pvo, vc, size)?;
    vbuf.truncate(vsize);

    let csize = compress(&vbuf);

    println!(
        "SpatialCP: {} bits/vertex (post-deflate {} bits/vertex); sort {} msec",
        f((vbuf.len() * 8) as f64 / vc as f64, 1),
        f((csize * 8) as f64 / vc as f64, 1),
        f((end - start) * 1000.0, 2)
    );
    Ok(())
}

pub fn filter_triangles(mesh: &Mesh) -> Result<()> {
    let data = bytes(&mesh.vertices);

    let start = timestamp();

    let mut newindices = vec![0u32; mesh.indices.len()];
    let count = filter_index_buffer(
        &mut newindices,
        &mesh.indices,
        mesh.indices.len(),
        &data,
        mesh.vertices.len(),
        12,
        VERTEX_SIZE,
    )?;
    newindices.truncate(count);

    let end = timestamp();

    println!(
        "FilterIB : {} triangles -> {} ({}% redundant) in {} msec",
        mesh.indices.len() / 3,
        newindices.len() / 3,
        f(
            ((mesh.indices.len() - newindices.len()) / 3) as f64 / (mesh.indices.len() / 3) as f64 * 100.0,
            2
        ),
        f((end - start) * 1000.0, 2)
    );
    Ok(())
}

pub fn tessellation_adjacency(mesh: &Mesh) -> Result<()> {
    let positions = floats(&mesh.vertices);
    let (ic, vc) = (mesh.indices.len(), mesh.vertices.len());

    let start = timestamp();

    // 12 indices per input triangle
    let mut tessib = vec![0u32; ic * 4];
    generate_tessellation_index_buffer(&mut tessib, &mesh.indices, ic, &positions, vc, VERTEX_SIZE)?;

    let middle = timestamp();

    // 6 indices per input triangle
    let mut adjib = vec![0u32; ic * 2];
    generate_adjacency_index_buffer(&mut adjib, &mesh.indices, ic, &positions, vc, VERTEX_SIZE)?;

    let end = timestamp();

    println!("Tesselltn: {} patches in {} msec", ic / 3, f((middle - start) * 1000.0, 2));
    println!("Adjacency: {} patches in {} msec", ic / 3, f((end - middle) * 1000.0, 2));
    Ok(())
}

pub fn provoking(mesh: &Mesh) -> Result<()> {
    let (ic, vc) = (mesh.indices.len(), mesh.vertices.len());

    let start = timestamp();

    // worst case number of vertices: vertex count + triangle count
    let mut pib = vec![0u32; ic];
    let mut reorder = vec![0u32; vc + ic / 3];

    let pcount = generate_provoking_index_buffer(&mut pib, &mut reorder, &mesh.indices, ic, vc)?;
    reorder.truncate(pcount);

    let end = timestamp();

    // validate invariant: pib[i] == i/3 for provoking vertices
    for i in (0..ic).step_by(3) {
        assert!(pib[i] as usize == i / 3);
    }

    // validate invariant: reorder[pib[x]] == ib[x] modulo triangle rotation
    // note: this is technically not promised by the interface (it may reorder triangles!), it just happens to hold right now
    for i in (0..ic).step_by(3) {
        let (a, b, c) = (mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]);
        let (ra, rb, rc) = (reorder[pib[i] as usize], reorder[pib[i + 1] as usize], reorder[pib[i + 2] as usize]);

        assert!((a == ra && b == rb && c == rc) || (a == rb && b == rc && c == ra) || (a == rc && b == ra && c == rb));
    }

    // best case number of vertices: max(vertex count, triangle count), assuming non-redundant indexing (all vertices are used)
    // note: this is a lower bound, and it's not theoretically possible on some meshes;
    // for example, a union of a flat shaded cube (12t 24v) and a smooth shaded icosahedron (20t 12v) will have 36 vertices and 32 triangles
    // however, the best case for that union is 44 vertices (24 cube vertices + 20 icosahedron vertices due to provoking invariant)
    let bestv = vc.max(ic / 3);

    println!(
        "Provoking: {} triangles / {} vertices (+{}% extra) in {} msec",
        ic / 3,
        pcount,
        f(pcount as f64 / bestv as f64 * 100.0 - 100.0, 1),
        f((end - start) * 1000.0, 2)
    );
    Ok(())
}

pub fn reindex_fuzzy(mesh: &Mesh) -> Result<()> {
    let pv = pack_mesh(&mesh.vertices);
    let positions = floats(&mesh.vertices);
    let (ic, vc) = (mesh.indices.len(), mesh.vertices.len());

    let mut remap = vec![0u32; vc];

    let start = timestamp();

    let up = generate_vertex_remap(&mut remap, Some(&mesh.indices), ic, &pv, vc, PACKED_VERTEX_SIZE)?;

    let middle = timestamp();

    let vertices = &mesh.vertices;
    let mut compare = |lhs: u32, rhs: u32| -> bool {
        let lv = &vertices[lhs as usize];
        let rv = &vertices[rhs as usize];

        let ln = lv.nx * lv.nx + lv.ny * lv.ny + lv.nz * lv.nz;
        let rn = rv.nx * rv.nx + rv.ny * rv.ny + rv.nz * rv.nz;

        // 1/1024px UV tolerance, 3 degree normal tolerance
        (lv.tx - rv.tx).abs() < 1e-3
            && (lv.ty - rv.ty).abs() < 1e-3
            && (lv.nx * rv.nx + lv.ny * rv.ny + lv.nz * rv.nz >= 0.9986 * (ln * rn).sqrt())
    };

    let uf = generate_vertex_remap_custom(&mut remap, Some(&mesh.indices), ic, &positions, vc, VERTEX_SIZE, Some(&mut compare))?;

    let end = timestamp();

    println!(
        "ReindexQ : {vc} vertices => {up} unique vertices in {} msec",
        f((middle - start) * 1000.0, 2)
    );
    println!(
        "ReindexF : {vc} vertices => {uf} unique vertices in {} msec",
        f((end - middle) * 1000.0, 2)
    );
    Ok(())
}

pub fn coverage(mesh: &Mesh) -> Result<()> {
    let positions = floats(&mesh.vertices);

    let start = timestamp();

    let cs = analyze_coverage(&mesh.indices, mesh.indices.len(), &positions, mesh.vertices.len(), VERTEX_SIZE)?;

    let end = timestamp();

    println!(
        "Coverage : X {}% Y {}% Z {}% in {} msec",
        f(f64::from(cs.coverage[0] * 100.0), 1),
        f(f64::from(cs.coverage[1] * 100.0), 1),
        f(f64::from(cs.coverage[2] * 100.0), 1),
        f((end - start) * 1000.0, 2)
    );
    Ok(())
}

/// `struct VertexTangent { float px, py, pz; float nx, ny, nz; float tu, tv; float tx, ty, tz, tw; }`
// positions/normals/UVs are carried along like in C++; only tangents are compared
#[allow(dead_code)]
#[derive(Clone, Copy, Default)]
struct VertexTangent {
    p: [f32; 3],
    n: [f32; 3],
    uv: [f32; 2],
    t: [f32; 4],
}

pub fn tangents(mesh: &Mesh) -> Result<()> {
    let data = floats(&mesh.vertices);
    let ic = mesh.indices.len();

    let start = timestamp();

    // note: tangent generation produces a tangent vector and orientation (+-1) per *corner* (3 per triangle), even for indexed inputs
    // you *could* then copy the tangents to the existing vertices, but that will not work correctly in case the input has UV mirroring
    // a simple way to handle this is to work on unindexed data (indices==NULL), generate tangents, and re-index
    // we show a more involved way that splits vertices with divergent tangents on the fly.
    let mut tangents = vec![0f32; ic * 4];

    generate_tangents(
        &mut tangents,
        Some(&mesh.indices),
        ic,
        &data,
        mesh.vertices.len(),
        VERTEX_SIZE,
        &data[3..],
        VERTEX_SIZE,
        &data[6..],
        VERTEX_SIZE,
        0,
    )?;

    // copy positions/normals/UVs (Vertex has no tangent field, so we use VertexTangent)
    let mut newvertices: Vec<VertexTangent> = mesh
        .vertices
        .iter()
        .map(|v| VertexTangent {
            p: [v.px, v.py, v.pz],
            n: [v.nx, v.ny, v.nz],
            uv: [v.tx, v.ty],
            t: [0.0; 4],
        })
        .collect();
    let mut newindices = mesh.indices.clone();

    let tangent = |i: usize| -> [f32; 4] { [tangents[i * 4], tangents[i * 4 + 1], tangents[i * 4 + 2], tangents[i * 4 + 3]] };

    // seed each vertex with one of its corner tangents; the loop below fixes any mismatches
    for (i, &index) in mesh.indices.iter().enumerate() {
        newvertices[index as usize].t = tangent(i);
    }

    let mut splits = vec![!0u32; mesh.vertices.len()];

    for (i, &index) in mesh.indices.iter().enumerate() {
        let t = tangent(i);
        let mut v = index;

        // walk the chain of split copies looking for a vertex whose tangent matches
        while v != !0u32 && newvertices[v as usize].t != t {
            v = splits[v as usize];
        }

        // no match in chain: append a new split copy with the target tangent and chain it
        if v == !0u32 {
            v = newvertices.len() as u32;
            let mut copy = newvertices[index as usize];
            copy.t = t;
            newvertices.push(copy);
            splits.push(splits[index as usize]);
            splits[index as usize] = v;
        }

        newindices[i] = v;
    }

    let end = timestamp();

    println!(
        "Tangents : {} corners, {} splits in {} msec",
        ic,
        newvertices.len() - mesh.vertices.len(),
        f((end - start) * 1000.0, 2)
    );
    Ok(())
}

pub fn normals(mesh: &Mesh, smoothing: f32) -> Result<()> {
    let positions = floats(&mesh.vertices);
    let ic = mesh.indices.len();

    // note: normal generation produces a normal vector per *corner* (3 per triangle), even for indexed inputs
    // you could split the existing vertices on the fly (see tangents()), here we demonstrate an alternative approach with full reindexing
    let mut normals = vec![0f32; ic * 3];

    let start = timestamp();

    let crease = 1.0f32; // 1 rad ~= 60 deg
    generate_normals(
        &mut normals,
        Some(&mesh.indices),
        ic,
        &positions,
        mesh.vertices.len(),
        VERTEX_SIZE,
        crease,
        smoothing,
    )?;

    let middle = timestamp();

    let mut unindexed = vec![Vertex::default(); ic];

    for (i, &index) in mesh.indices.iter().enumerate() {
        unindexed[i] = mesh.vertices[index as usize];
        unindexed[i].nx = normals[i * 3];
        unindexed[i].ny = normals[i * 3 + 1];
        unindexed[i].nz = normals[i * 3 + 2];
    }

    let data = bytes(&unindexed);

    let mut remap = vec![0u32; ic];
    let vertex_count = generate_vertex_remap::<u32>(&mut remap, None, ic, &data, ic, VERTEX_SIZE)?;

    let mut newindices = vec![0u32; ic];
    remap_index_buffer::<u32>(&mut newindices, None, ic, &remap)?;

    let mut newvertices = vec![0u8; vertex_count * VERTEX_SIZE];
    remap_vertex_buffer(&mut newvertices, &data, ic, VERTEX_SIZE, &remap)?;

    let end = timestamp();

    println!(
        "Normals {}: {} corners, {} vertices in {} msec; reindexed in {} msec",
        smoothing.ceil() as i32,
        ic,
        vertex_count,
        f((middle - start) * 1000.0, 2),
        f((end - middle) * 1000.0, 2)
    );
    Ok(())
}

pub fn remesh_mesh(mesh: &Mesh, resolution: i32, options: u32) -> Result<()> {
    let positions = floats(&mesh.vertices);
    let (ic, vc) = (mesh.indices.len(), mesh.vertices.len());

    let start = timestamp();
    let capacity = remesh(None, 0, &mesh.indices, ic, &positions, vc, VERTEX_SIZE, resolution, options)?;
    let middle = timestamp();
    let mut triangles = vec![0f32; capacity * 3 * 3];
    let count = remesh(
        Some(&mut triangles),
        capacity,
        &mesh.indices,
        ic,
        &positions,
        vc,
        VERTEX_SIZE,
        resolution,
        options,
    )?;
    let end = timestamp();

    assert!(count <= capacity);

    println!(
        "Remesh{resolution:>3}: {} triangles => {} triangles in {} msec (count {} msec, compute {} msec)",
        ic / 3,
        count,
        f((end - start) * 1000.0, 2),
        f((middle - start) * 1000.0, 2),
        f((end - middle) * 1000.0, 2)
    );
    Ok(())
}

pub fn load_mesh(path: &str) -> Result<Option<Mesh>> {
    let start = timestamp();
    let mut middle = 0f64;
    let mesh = parse_obj(path, &mut middle)?;
    let end = timestamp();

    if mesh.vertices.is_empty() {
        println!("Mesh {path} is empty, skipping");
        return Ok(None);
    }

    println!(
        "# {path}: {} vertices, {} triangles; read in {} msec; indexed in {} msec",
        mesh.vertices.len(),
        mesh.indices.len() / 3,
        f((middle - start) * 1000.0, 2),
        f((end - middle) * 1000.0, 2)
    );
    Ok(Some(mesh))
}

pub fn process_deinterleaved(path: &str) -> Result<()> {
    // Most algorithms in the library work out of the box with deinterleaved geometry, but some require slightly special treatment;
    // this code runs a simplified version of complete opt. pipeline using deinterleaved geo. There's no compression performed but you
    // can trivially run it by quantizing all elements and running meshopt_encodeVertexBuffer once for each vertex stream.
    let Some(obj) = fast_obj_read(path) else {
        println!("Error loading {path}: file not found");
        return Ok(());
    };

    let mut total_indices = 0;

    for &fv in &obj.face_vertices[..obj.face_count as usize] {
        total_indices += 3 * (fv as usize).saturating_sub(2);
    }

    let mut unindexed_pos = vec![0f32; total_indices * 3];
    let mut unindexed_nrm = vec![0f32; total_indices * 3];
    let mut unindexed_uv = vec![0f32; total_indices * 2];

    let mut vertex_offset = 0;
    let mut index_offset = 0;

    for &fv in &obj.face_vertices[..obj.face_count as usize] {
        if fv <= 2 {
            continue;
        }

        for j in 0..fv as usize {
            let gi = obj.indices[index_offset + j];

            // triangulate polygon on the fly; offset-3 is always the first polygon vertex
            if j >= 3 {
                unindexed_pos.copy_within((vertex_offset - 3) * 3..(vertex_offset - 2) * 3, vertex_offset * 3);
                unindexed_nrm.copy_within((vertex_offset - 3) * 3..(vertex_offset - 2) * 3, vertex_offset * 3);
                unindexed_uv.copy_within((vertex_offset - 3) * 2..(vertex_offset - 2) * 2, vertex_offset * 2);
                unindexed_pos.copy_within((vertex_offset - 1) * 3..vertex_offset * 3, (vertex_offset + 1) * 3);
                unindexed_nrm.copy_within((vertex_offset - 1) * 3..vertex_offset * 3, (vertex_offset + 1) * 3);
                unindexed_uv.copy_within((vertex_offset - 1) * 2..vertex_offset * 2, (vertex_offset + 1) * 2);
                vertex_offset += 2;
            }

            let (p, n, t) = (gi.p as usize, gi.n as usize, gi.t as usize);
            unindexed_pos[vertex_offset * 3..vertex_offset * 3 + 3].copy_from_slice(&obj.positions[p * 3..p * 3 + 3]);
            unindexed_nrm[vertex_offset * 3..vertex_offset * 3 + 3].copy_from_slice(&obj.normals[n * 3..n * 3 + 3]);
            unindexed_uv[vertex_offset * 2..vertex_offset * 2 + 2].copy_from_slice(&obj.texcoords[t * 2..t * 2 + 2]);
            vertex_offset += 1;
        }

        index_offset += fv as usize;
    }

    fast_obj_destroy(obj);

    let start = timestamp();

    let to_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let (pos_bytes, nrm_bytes, uv_bytes) = (to_bytes(&unindexed_pos), to_bytes(&unindexed_nrm), to_bytes(&unindexed_uv));

    let streams = [
        Stream {
            data: &pos_bytes,
            size: 12,
            stride: 12,
        },
        Stream {
            data: &nrm_bytes,
            size: 12,
            stride: 12,
        },
        Stream {
            data: &uv_bytes,
            size: 8,
            stride: 8,
        },
    ];

    let mut remap = vec![0u32; total_indices];

    let total_vertices = generate_vertex_remap_multi::<u32>(&mut remap, None, total_indices, total_indices, &streams, streams.len())?;

    let mut indices = vec![0u32; total_indices];
    remap_index_buffer::<u32>(&mut indices, None, total_indices, &remap)?;

    let mut pos = vec![0u8; total_vertices * 12];
    remap_vertex_buffer(&mut pos, &pos_bytes, total_indices, 12, &remap)?;

    let mut nrm = vec![0u8; total_vertices * 12];
    remap_vertex_buffer(&mut nrm, &nrm_bytes, total_indices, 12, &remap)?;

    let mut uv = vec![0u8; total_vertices * 8];
    remap_vertex_buffer(&mut uv, &uv_bytes, total_indices, 8, &remap)?;

    let reindex = timestamp();

    let source = indices.clone();
    optimize_vertex_cache(&mut indices, &source, total_indices, total_vertices)?;

    optimize_vertex_fetch_remap(&mut remap, &indices, total_indices, total_vertices)?;
    for (stream, size) in [(&mut pos, 12), (&mut nrm, 12), (&mut uv, 8)] {
        let source = stream.clone();
        remap_vertex_buffer(stream, &source, total_vertices, size, &remap)?;
    }

    let optimize = timestamp();

    // note: since shadow index buffer is computed based on regular vertex/index buffer, the stream points at the indexed data - not unindexed_pos
    let shadow_stream = [Stream {
        data: &pos,
        size: 12,
        stride: 12,
    }];

    let mut shadow_indices = vec![0u32; total_indices];
    generate_shadow_index_buffer_multi(&mut shadow_indices, &indices, total_indices, total_vertices, &shadow_stream, 1)?;

    let source = shadow_indices.clone();
    optimize_vertex_cache(&mut shadow_indices, &source, total_indices, total_vertices)?;

    let shadow = timestamp();

    println!(
        "Deintrlvd: {} vertices, reindexed in {} msec, optimized in {} msec, generated & optimized shadow indices in {} msec",
        total_vertices,
        f((reindex - start) * 1000.0, 2),
        f((optimize - reindex) * 1000.0, 2),
        f((shadow - optimize) * 1000.0, 2)
    );
    Ok(())
}

pub fn process(path: &str) -> Result<()> {
    let Some(mesh) = load_mesh(path)? else {
        return Ok(());
    };

    optimize(&mesh, false)?;
    optimize(&mesh, /* fifo= */ true)?;

    let mut copy = mesh.clone();
    let source = copy.indices.clone();
    optimize_vertex_cache(&mut copy.indices, &source, source.len(), copy.vertices.len())?;
    let source = copy.vertices.clone();
    optimize_fetch(&mut copy.vertices, &mut copy.indices, &source)?;

    let mut copystrip = mesh.clone();
    let source = copystrip.indices.clone();
    optimize_vertex_cache_strip(&mut copystrip.indices, &source, source.len(), copystrip.vertices.len())?;
    let source = copystrip.vertices.clone();
    optimize_fetch(&mut copystrip.vertices, &mut copystrip.indices, &source)?;

    stripify_mesh(&copy, false, ' ')?;
    stripify_mesh(&copy, /* use_restart= */ true, 'R')?;
    stripify_mesh(&copystrip, /* use_restart= */ true, 'S')?;

    meshlets(&copy, /* scan= */ true, false, false, false, false)?;
    meshlets(&copy, /* scan= */ false, false, false, false, false)?;
    meshlets(&copy, /* scan= */ false, /* uniform= */ true, false, false, false)?;
    meshlets(
        &copy, /* scan= */ false, /* uniform= */ false, /* flex= */ true, false, false,
    )?;
    meshlets(
        &copy, /* scan= */ false, /* uniform= */ true, /* flex= */ false, /* spatial= */ true, false,
    )?;

    shadow(&copy)?;
    filter_triangles(&copy)?;
    tessellation_adjacency(&copy)?;
    provoking(&copy)?;

    encode_index_mesh(&copy, ' ')?;
    encode_index_mesh(&copystrip, 'S')?;

    let mut strip = vec![0u32; stripify_bound(copystrip.indices.len())?];
    let count = stripify(&mut strip, &copystrip.indices, copystrip.indices.len(), copystrip.vertices.len(), 0)?;
    strip.truncate(count);

    encode_index_sequence_data(&strip, copystrip.vertices.len(), 'D')?;

    pack_vertex(&copy, Packing::Plain, "");
    encode_vertex(&copy, Packing::Plain, "", 2)?;
    encode_vertex(&copy, Packing::Oct, "O", 2)?;

    encode_meshlets(&mesh, 64, 96, true, 3)?;
    encode_meshlets_dxr(&mesh, 128, /* min_exp= */ -14, 3)?;

    simplify_mesh(&mesh, 0.2, 0)?;
    simplify_mesh(&mesh, 0.1, SIMPLIFY_PRUNE)?;
    simplify_attr(&mesh, 0.2, 0)?;
    simplify_attr(&mesh, 0.1, SIMPLIFY_PERMISSIVE)?;
    simplify_update(&mesh, 0.2, 0)?;
    simplify_update(&mesh, 0.1, SIMPLIFY_PERMISSIVE)?;
    simplify_sloppy_mesh(&mesh, 0.2)?;
    simplify_complete(&mesh)?;
    simplify_points_mesh(&mesh, 0.2)?;
    simplify_clusters(&mesh, 0.2)?;

    spatial_sort(&mesh)?;
    spatial_sort_triangles_mesh(&mesh)?;
    spatial_cluster_points_mesh(&mesh, 64)?;

    reindex_fuzzy(&mesh)?;
    coverage(&mesh)?;

    tangents(&mesh)?;
    normals(&mesh, 1.5)?;

    remesh_mesh(&mesh, 128, 0)?;
    remesh_mesh(&mesh, 128, REMESH_SOLVE)?;

    process_deinterleaved(path)
}

pub fn process_dev(path: &str) -> Result<()> {
    let Some(mesh) = load_mesh(path)? else {
        return Ok(());
    };

    remesh_mesh(&mesh, 160, REMESH_SOLVE)
}

pub fn process_nanite(path: &str) -> Result<()> {
    let Some(mesh) = load_mesh(path)? else {
        return Ok(());
    };

    crate::nanite::nanite(&mesh.vertices, &mesh.indices)
}
