//! `tools/codecbench.cpp`: decoder throughput for vertex/index codecs, meshlet codec and vertex filters.
//! Setup (grid generation, encoding, meshlet packing) is split from the timing loops so tests can check it.

use std::time::Instant;

use meshoptimizer::{
    Meshlet, build_meshlets, build_meshlets_bound, decode_filter_color, decode_filter_exp, decode_filter_oct, decode_filter_quat,
    decode_index_buffer, decode_meshlet, decode_vertex_buffer, encode_index_buffer, encode_index_buffer_bound, encode_meshlet,
    encode_meshlet_bound, encode_vertex_buffer, encode_vertex_buffer_bound, optimize_meshlet_level, optimize_vertex_cache,
    optimize_vertex_fetch,
};

pub type Result<T> = std::result::Result<T, String>;

const ATTEMPTS: usize = 50;

/// `struct Vertex { uint16_t data[16]; }`
pub const VERTEX_SIZE: usize = 32;

fn err(what: &str) -> impl Fn(meshoptimizer::Error) -> String + '_ {
    move |e| format!("{what}: {e}")
}

fn timestamp() -> Instant {
    Instant::now()
}

fn seconds(t0: Instant, t1: Instant) -> f64 {
    (t1 - t0).as_secs_f64()
}

pub fn murmur3(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;

    h
}

/// The `(n + 1)^2` vertex grid from `main`: 32-byte vertices with random k-bit words, xy positions, 2 triangles per cell.
pub struct Grid {
    pub vertices: Vec<u8>,
    pub positions: Vec<f32>,
    pub indices: Vec<u32>,
}

pub fn make_grid(n: usize) -> Grid {
    let side = n + 1;
    let mut vertices = Vec::with_capacity(side * side * VERTEX_SIZE);
    let mut positions = vec![0f32; side * side * 3];

    for x in 0..=n {
        for y in 0..=n {
            for k in 0..16u32 {
                let h = murmur3(((x * side + y) * 16) as u32 + k);

                // use random k-bit sequence for each word to test all encoding types
                let v = (h & ((1u32 << (k + 1)) - 1)) as u16;
                vertices.extend_from_slice(&v.to_le_bytes());
            }

            let index = vertices.len() / VERTEX_SIZE - 1;

            positions[index * 3] = x as f32;
            positions[index * 3 + 1] = y as f32;
            positions[index * 3 + 2] = 0.0;
        }
    }

    let mut indices = Vec::with_capacity(n * n * 6);

    for x in 0..n {
        for y in 0..n {
            let at = |dx: usize, dy: usize| ((x + dx) * side + (y + dy)) as u32;

            indices.extend_from_slice(&[at(0, 0), at(1, 0), at(0, 1)]);
            indices.extend_from_slice(&[at(0, 1), at(1, 0), at(1, 1)]);
        }
    }

    Grid {
        vertices,
        positions,
        indices,
    }
}

/// Encoded streams of `benchCodecs` plus the buffers it decodes into.
pub struct Codecs {
    pub vertex_count: usize,
    pub index_count: usize,
    pub vb: Vec<u8>,
    pub ib: Vec<u32>,
    pub vc: Vec<u8>,
    pub ic: Vec<u8>,
}

pub fn prepare_codecs(vertices: &[u8], indices: &[u32]) -> Result<Codecs> {
    let vertex_count = vertices.len() / VERTEX_SIZE;
    let index_count = indices.len();

    let mut vb = vec![0u8; vertices.len()];
    let mut ib = vec![0u32; index_count];

    optimize_vertex_cache(&mut ib, indices, index_count, vertex_count).map_err(err("optimize_vertex_cache"))?;
    optimize_vertex_fetch(&mut vb, &mut ib, index_count, vertices, vertex_count, VERTEX_SIZE).map_err(err("optimize_vertex_fetch"))?;

    let mut vc = vec![0u8; encode_vertex_buffer_bound(vertex_count, VERTEX_SIZE).map_err(err("encode_vertex_buffer_bound"))?];
    let size = vc.len();
    let size = encode_vertex_buffer(&mut vc, size, &vb, vertex_count, VERTEX_SIZE).map_err(err("encode_vertex_buffer"))?;
    vc.truncate(size);

    let mut ic = vec![0u8; encode_index_buffer_bound(index_count, vertex_count).map_err(err("encode_index_buffer_bound"))?];
    let size = ic.len();
    let size = encode_index_buffer(&mut ic, size, &ib, index_count).map_err(err("encode_index_buffer"))?;
    ic.truncate(size);

    Ok(Codecs {
        vertex_count,
        index_count,
        vb,
        ib,
        vc,
        ic,
    })
}

/// `benchCodecs`: best vertex and index decode throughput in GB/s.
pub fn bench_codecs(vertices: &[u8], indices: &[u32], verbose: bool) -> Result<(f64, f64)> {
    if verbose {
        println!(
            "source: vertex data {} bytes, index data {} bytes",
            vertices.len(),
            indices.len() * 4
        );
    }

    let Codecs {
        vertex_count,
        index_count,
        mut vb,
        mut ib,
        vc,
        ic,
    } = prepare_codecs(vertices, indices)?;

    if verbose {
        println!("encode: vertex data {} bytes, index data {} bytes", vc.len(), ic.len());
    }

    let (mut bestvd, mut bestid) = (0f64, 0f64);
    let vbytes = (vertex_count * VERTEX_SIZE) as f64;
    let ibytes = (index_count * 4) as f64;

    for _ in 0..ATTEMPTS {
        let t0 = timestamp();

        decode_vertex_buffer(&mut vb, vertex_count, VERTEX_SIZE, &vc, vc.len()).map_err(err("decode_vertex_buffer"))?;

        let t1 = timestamp();

        decode_index_buffer(&mut ib, index_count, &ic, ic.len()).map_err(err("decode_index_buffer"))?;

        let t2 = timestamp();

        let (dv, di) = (seconds(t0, t1), seconds(t1, t2));

        if verbose {
            println!(
                "decode: vertex {:.2} ms ({:.2} GB/sec), index {:.2} ms ({:.2} GB/sec)",
                dv * 1000.0,
                vbytes / 1e9 / dv,
                di * 1000.0,
                ibytes / 1e9 / di
            );
        }

        bestvd = bestvd.max(vbytes / 1e9 / dv);
        bestid = bestid.max(ibytes / 1e9 / di);
    }

    Ok((bestvd, bestid))
}

/// Best filter decode throughput in GB/s: oct8, oct12, quat12, col8, col12, exp.
pub type FilterScores = [f64; 6];

/// `benchFilters`
pub fn bench_filters(count: usize, verbose: bool) -> Result<FilterScores> {
    // note: the filters are branchless so we just run them on runs of zeroes
    let count4 = (count + 3) & !3;
    let mut d4 = vec![0u8; count4 * 4];
    let mut d8 = vec![0u8; count4 * 8];

    if verbose {
        println!("filters: oct8 data {} bytes, oct12/quat12 data {} bytes", d4.len(), d8.len());
    }

    let mut best = [0f64; 6];

    for _ in 0..ATTEMPTS {
        let t0 = timestamp();

        decode_filter_oct(&mut d4, count4, 4).map_err(err("decode_filter_oct"))?;

        let t1 = timestamp();

        decode_filter_oct(&mut d8, count4, 8).map_err(err("decode_filter_oct"))?;

        let t2 = timestamp();

        decode_filter_quat(&mut d8, count4, 8).map_err(err("decode_filter_quat"))?;

        let t3 = timestamp();

        decode_filter_color(&mut d4, count4, 4).map_err(err("decode_filter_color"))?;

        let t4 = timestamp();

        decode_filter_color(&mut d8, count4, 8).map_err(err("decode_filter_color"))?;

        let t5 = timestamp();

        decode_filter_exp(&mut d8, count4, 8).map_err(err("decode_filter_exp"))?;

        let t6 = timestamp();

        let sizes = [d4.len(), d8.len(), d8.len(), d4.len(), d8.len(), d8.len()];
        let times = [
            seconds(t0, t1),
            seconds(t1, t2),
            seconds(t2, t3),
            seconds(t3, t4),
            seconds(t4, t5),
            seconds(t5, t6),
        ];
        let rates: Vec<f64> = sizes.iter().zip(times).map(|(&size, time)| size as f64 / 1e9 / time).collect();

        if verbose {
            println!(
                "filter: oct8 {:.2} ms ({:.2} GB/sec), oct12 {:.2} ms ({:.2} GB/sec), quat12 {:.2} ms ({:.2} GB/sec), col8 {:.2} ms ({:.2} GB/sec), col12 {:.2} ms ({:.2} GB/sec), exp {:.2} ms ({:.2} GB/sec)",
                times[0] * 1000.0,
                rates[0],
                times[1] * 1000.0,
                rates[1],
                times[2] * 1000.0,
                rates[2],
                times[3] * 1000.0,
                rates[3],
                times[4] * 1000.0,
                rates[4],
                times[5] * 1000.0,
                rates[5]
            );
        }

        for (best, rate) in best.iter_mut().zip(rates) {
            *best = best.max(rate);
        }
    }

    Ok(best)
}

/// Meshlets of `benchMeshlets`, each packed as `[vertex_count, triangle_count, size_lo, size_hi, encoded...]`.
pub struct PackedMeshlets {
    pub meshlet_count: usize,
    pub packed: Vec<u8>,
    /// decoded bytes: 3 per triangle + 4 per vertex reference
    pub totals: usize,
}

pub const MESHLET_MAX_VERTICES: usize = 64;
pub const MESHLET_MAX_TRIANGLES: usize = 96;

pub fn prepare_meshlets(positions: &[f32], indices: &[u32], encoderefs: bool) -> Result<PackedMeshlets> {
    let max_vertices = MESHLET_MAX_VERTICES;
    let max_triangles = MESHLET_MAX_TRIANGLES;
    let vertex_count = positions.len() / 3;

    let max_meshlets = build_meshlets_bound(indices.len(), max_vertices, max_triangles).map_err(err("build_meshlets_bound"))?;
    let mut meshlets = vec![Meshlet::default(); max_meshlets];
    let mut meshlet_vertices = vec![0u32; indices.len()];
    let mut meshlet_triangles = vec![0u8; indices.len()];

    let meshlet_count = build_meshlets(
        &mut meshlets,
        &mut meshlet_vertices,
        &mut meshlet_triangles,
        indices,
        indices.len(),
        positions,
        vertex_count,
        12,
        max_vertices,
        max_triangles,
        0.0,
    )
    .map_err(err("build_meshlets"))?;
    meshlets.truncate(meshlet_count);

    let last = meshlets.last().copied().ok_or("build_meshlets: no meshlets")?;

    meshlet_vertices.truncate((last.vertex_offset + last.vertex_count) as usize);
    meshlet_triangles.truncate((last.triangle_offset + last.triangle_count * 3) as usize);

    let mut cbuf = vec![0u8; encode_meshlet_bound(max_vertices, max_triangles)];

    // optimize each meshlet for locality before encoding
    for m in &meshlets {
        optimize_meshlet_level(
            &mut meshlet_vertices[m.vertex_offset as usize..],
            m.vertex_count as usize,
            &mut meshlet_triangles[m.triangle_offset as usize..],
            m.triangle_count as usize,
            3,
        )
        .map_err(err("optimize_meshlet_level"))?;
    }

    // optimize meshlet_vertices for locality (requires vertex reorder)
    let position_bytes: Vec<u8> = positions.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut positionscopy = vec![0u8; position_bytes.len()];
    let reference_count = meshlet_vertices.len();
    optimize_vertex_fetch(
        &mut positionscopy,
        &mut meshlet_vertices,
        reference_count,
        &position_bytes,
        vertex_count,
        12,
    )
    .map_err(err("optimize_vertex_fetch"))?;

    let mut packed = Vec::new();
    let mut totals = 0;

    for m in &meshlets {
        let mvc = if encoderefs { m.vertex_count as usize } else { 0 };
        let size = cbuf.len();

        let mbs = encode_meshlet(
            &mut cbuf,
            size,
            &meshlet_vertices[m.vertex_offset as usize..],
            mvc,
            &meshlet_triangles[m.triangle_offset as usize..],
            m.triangle_count as usize,
        )
        .map_err(err("encode_meshlet"))?;

        packed.push(mvc as u8);
        packed.push(m.triangle_count as u8);
        packed.push((mbs & 0xff) as u8);
        packed.push(((mbs >> 8) & 0xff) as u8);
        packed.extend_from_slice(&cbuf[..mbs]);

        totals += m.triangle_count as usize * 3 + mvc * 4;
    }

    Ok(PackedMeshlets {
        meshlet_count,
        packed,
        totals,
    })
}

/// `benchMeshlets`: best meshlet decode throughput in GB/s.
pub fn bench_meshlets(positions: &[f32], indices: &[u32], encoderefs: bool, verbose: bool) -> Result<f64> {
    let PackedMeshlets {
        meshlet_count,
        packed,
        totals,
    } = prepare_meshlets(positions, indices, encoderefs)?;

    if verbose {
        println!(
            "meshlets ({}/{}): {} meshlets, packed {} bytes ({:.1} bits/triangle)",
            MESHLET_MAX_VERTICES,
            MESHLET_MAX_TRIANGLES,
            meshlet_count,
            packed.len(),
            (packed.len() * 8) as f64 / (indices.len() / 3) as f64
        );
    }

    let mut rv = [0u32; 256];
    let mut rt = [0u8; 256 * 3];
    let mut bestml = 0f64;

    for _ in 0..ATTEMPTS {
        let t0 = timestamp();

        let mut p = 0;

        for _ in 0..meshlet_count {
            let mbs = packed[p + 2] as usize | (packed[p + 3] as usize) << 8;
            decode_meshlet(&mut rv, packed[p] as usize, &mut rt, packed[p + 1] as usize, &packed[p + 4..], mbs)
                .map_err(err("decode_meshlet"))?;
            p += 4 + mbs;
        }

        if p != packed.len() {
            return Err("decode_meshlet: packed stream not fully consumed".into());
        }

        let t1 = timestamp();
        let dt = seconds(t0, t1);

        if verbose {
            println!("meshlet decode: {:.2} ms ({:.2} GB/sec)", dt * 1000.0, totals as f64 / 1e9 / dt);
        }

        bestml = bestml.max(totals as f64 / 1e9 / dt);
    }

    Ok(bestml)
}

/// One input of the file mode: a re-encoded vertex stream (`v<count>_s<stride>_*`) or an index stream (`i<count>_*`).
pub struct File {
    pub data: Vec<u8>,
    pub stride: usize,
    pub count: usize,
    pub index: bool,
}

/// `%d` of `sscanf`: optional whitespace, sign and at least one digit.
fn scan_int(s: &str) -> Option<(i64, &str)> {
    let s = s.trim_start();
    let digits_at = usize::from(s.starts_with(['+', '-']));
    let end = digits_at + s[digits_at..].find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len() - digits_at);

    if end == digits_at {
        return None;
    }

    Some((s[..end].parse().ok()?, &s[end..]))
}

/// `sscanf(name, "v%d_s%d_", ...)`
fn scan_vertex_name(name: &str) -> Option<(i64, i64)> {
    let (count, rest) = scan_int(name.strip_prefix('v')?)?;
    let (stride, _) = scan_int(rest.strip_prefix("_s")?)?;
    Some((count, stride))
}

/// `sscanf(name, "i%d_", ...)`
fn scan_index_name(name: &str) -> Option<i64> {
    scan_int(name.strip_prefix('i')?).map(|(count, _)| count)
}

pub fn read_file(path: &str) -> Result<File> {
    let input = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;

    let name = path.rsplit('/').next().unwrap_or(path);
    let positive = |v: i64| usize::try_from(v).map_err(|_| format!("{path}: invalid count in file name"));

    if let Some((vcnt, vsz)) = scan_vertex_name(name) {
        let count = positive(vcnt)?;
        let stride = positive(vsz)?;

        let mut decoded = vec![0u8; count * stride];
        let res = decode_vertex_buffer(&mut decoded, count, stride, &input, input.len());
        if res.is_err() && input.len() == decoded.len() {
            // some files are not encoded
            decoded.copy_from_slice(&input);
        }

        let mut data = vec![0u8; encode_vertex_buffer_bound(count, stride).map_err(err(path))?];
        let size = data.len();
        let size = encode_vertex_buffer(&mut data, size, &decoded, count, stride).map_err(err(path))?;
        data.truncate(size);

        Ok(File {
            data,
            stride,
            count,
            index: false,
        })
    } else if let Some(icnt) = scan_index_name(name) {
        let count = positive(icnt)?;

        // note: technically 2-byte indices can be used for larger meshes in some cases, but we simplify
        let stride = if count <= 0xffff { 2 } else { 4 };

        Ok(File {
            data: input,
            stride,
            count,
            index: true,
        })
    } else {
        Err(format!("{path}: unrecognized file name"))
    }
}

/// File mode of `main`: prints the compression ratio and the best total decode throughput per loop.
pub fn bench_files(paths: &[&str], loops: usize) -> Result<()> {
    let files = paths.iter().map(|path| read_file(path)).collect::<Result<Vec<_>>>()?;

    let max_size = files.iter().map(|f| f.count * f.stride).max().unwrap_or(0);
    let total_size: usize = files.iter().map(|f| f.count * f.stride).sum();
    let total_comp: usize = files.iter().map(|f| f.data.len()).sum();

    println!(
        "{} files, ratio {:.2} ({:.2} MB => {:.2} MB)",
        files.len(),
        total_comp as f64 / total_size as f64,
        total_size as f64 / 1024.0 / 1024.0,
        total_comp as f64 / 1024.0 / 1024.0
    );

    let mut buffer = vec![0u8; max_size];
    let mut buffer16 = vec![0u16; max_size / 2];
    let mut buffer32 = vec![0u32; max_size / 4];

    for _ in 0..loops {
        let mut bestvd = 0f64;

        for _ in 0..ATTEMPTS {
            let t0 = timestamp();

            for file in &files {
                let rc = match (file.index, file.stride) {
                    (true, 2) => decode_index_buffer(&mut buffer16, file.count, &file.data, file.data.len()),
                    (true, _) => decode_index_buffer(&mut buffer32, file.count, &file.data, file.data.len()),
                    (false, _) => decode_vertex_buffer(&mut buffer, file.count, file.stride, &file.data, file.data.len()),
                };

                if rc.is_err() {
                    return Err(format!(
                        "Failed to decode {} bytes into {} elements with stride {} (index {})",
                        file.data.len(),
                        file.count,
                        file.stride,
                        i32::from(file.index)
                    ));
                }
            }

            let t1 = timestamp();

            bestvd = bestvd.max(total_size as f64 / 1e9 / seconds(t0, t1));
        }

        println!("Score (GB/s):\t{bestvd:.2}");
    }

    Ok(())
}

/// Synthetic mode of `main` on an `n` x `n` grid (C++: `N = 1000`).
pub fn bench_synthetic(n: usize, loops: usize, verbose: bool) -> Result<()> {
    let grid = make_grid(n);

    println!("Codec:\tvtx\tidx\tmlet\tmletΔ\toct8\toct12\tquat12\tcol8\tcol12\texp");

    for _ in 0..loops {
        let (bestvd, bestid) = bench_codecs(&grid.vertices, &grid.indices, verbose)?;

        let bestml = bench_meshlets(&grid.positions, &grid.indices, true, verbose)?;
        let bestmt = bench_meshlets(&grid.positions, &grid.indices, false, verbose)?;

        let f = bench_filters(8 * n * n, verbose)?;

        println!(
            "GB/s :\t{bestvd:.2}\t{bestid:.2}\t{bestml:.2}\t{bestmt:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}",
            f[0], f[1], f[2], f[3], f[4], f[5]
        );
    }

    Ok(())
}
