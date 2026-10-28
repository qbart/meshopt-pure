//! `u16` index buffers (C++ template wrappers `meshopt_*<T>` via `meshopt_IndexAdapter`) must match the `u32` results.
//! No 1:1 counterpart in `demo/tests.cpp` (only `customAllocator` uses `unsigned short` indices there).

use meshoptimizer::*;

const N: usize = 9;

/// N x N grid of vertices (xyz + uv), slightly perturbed, with triangles in scrambled order and a duplicated corner vertex.
fn grid() -> (Vec<f32>, Vec<u32>) {
    let mut vb = Vec::new();
    for y in 0..N {
        for x in 0..N {
            let h = ((x * 7 + y * 13) % 5) as f32 * 0.05;
            vb.extend_from_slice(&[x as f32, y as f32, h, x as f32 / N as f32, y as f32 / N as f32]);
        }
    }
    // duplicate of vertex 0
    let dup: Vec<f32> = vb[..5].to_vec();
    vb.extend_from_slice(&dup);

    let mut ib = Vec::new();
    for y in 0..N - 1 {
        for x in 0..N - 1 {
            let i = (y * N + x) as u32;
            let n = N as u32;
            let i0 = if i == 0 { (N * N) as u32 } else { i };
            ib.extend_from_slice(&[i0, i + 1, i + n, i + 1, i + n + 1, i + n]);
        }
    }

    // scramble triangle order deterministically
    let tris: Vec<[u32; 3]> = ib.chunks(3).map(|t| [t[0], t[1], t[2]]).collect();
    let mut order: Vec<usize> = (0..tris.len()).collect();
    order.sort_by_key(|&t| (t * 37) % tris.len());
    let ib = order.iter().flat_map(|&t| tris[t]).collect();

    (vb, ib)
}

fn narrow(ib: &[u32]) -> Vec<u16> {
    ib.iter().map(|&i| i as u16).collect()
}

fn widen(ib: &[u16]) -> Vec<u32> {
    ib.iter().map(|&i| u32::from(i)).collect()
}

const STRIDE: usize = 20;

fn vertex_count(vb: &[f32]) -> usize {
    vb.len() * 4 / STRIDE
}

fn bytes(vb: &[f32]) -> Vec<u8> {
    vb.iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[test]
fn index_generator_u16() -> Result<()> {
    let (vb, ib) = grid();
    let ib16 = narrow(&ib);
    let (vc, ic) = (vertex_count(&vb), ib.len());
    let vbytes = bytes(&vb);

    let mut r32 = vec![0u32; vc];
    let mut r16 = vec![0u32; vc];
    let u32n = generate_vertex_remap(&mut r32, Some(&ib), ic, &vbytes, vc, STRIDE)?;
    let u16n = generate_vertex_remap(&mut r16, Some(&ib16), ic, &vbytes, vc, STRIDE)?;
    assert_eq!((u32n, &r32), (u16n, &r16));
    assert_eq!(u32n, vc - 1);

    let streams = [Stream {
        data: &vbytes,
        size: 12,
        stride: STRIDE,
    }];
    let u16n = generate_vertex_remap_multi(&mut r16, Some(&ib16), ic, vc, &streams, 1)?;
    let u32n = generate_vertex_remap_multi(&mut r32, Some(&ib), ic, vc, &streams, 1)?;
    assert_eq!((u32n, &r32), (u16n, &r16));

    let mut d32 = vec![0u32; ic];
    let mut d16 = vec![0u16; ic];
    remap_index_buffer(&mut d32, Some(&ib), ic, &r32)?;
    remap_index_buffer(&mut d16, Some(&ib16), ic, &r16)?;
    assert_eq!(d32, widen(&d16));

    let mut i16 = ib16.clone();
    remap_index_buffer_in_place(&mut i16, ic, &r16)?;
    assert_eq!(d32, widen(&i16));

    let n32 = filter_index_buffer(&mut d32, &ib, ic, &vbytes, vc, 12, STRIDE)?;
    let n16 = filter_index_buffer(&mut d16, &ib16, ic, &vbytes, vc, 12, STRIDE)?;
    assert_eq!((n32, &d32[..n32]), (n16, &widen(&d16)[..n16]));

    let n32 = filter_index_buffer_multi(&mut d32, &ib, ic, vc, &streams, 1)?;
    let n16 = filter_index_buffer_multi(&mut d16, &ib16, ic, vc, &streams, 1)?;
    assert_eq!((n32, &d32[..n32]), (n16, &widen(&d16)[..n16]));

    generate_shadow_index_buffer(&mut d32, &ib, ic, &vbytes, vc, 12, STRIDE)?;
    generate_shadow_index_buffer(&mut d16, &ib16, ic, &vbytes, vc, 12, STRIDE)?;
    assert_eq!(d32, widen(&d16));
    // vertex 0 and its duplicate collapse into one shadow vertex
    assert!(!(d32.contains(&0) && d32.contains(&((N * N) as u32))));

    generate_shadow_index_buffer_multi(&mut d32, &ib, ic, vc, &streams, 1)?;
    generate_shadow_index_buffer_multi(&mut d16, &ib16, ic, vc, &streams, 1)?;
    assert_eq!(d32, widen(&d16));

    let mut a32 = vec![0u32; ic * 2];
    let mut a16 = vec![0u16; ic * 2];
    generate_adjacency_index_buffer(&mut a32, &ib, ic, &vb, vc, STRIDE)?;
    generate_adjacency_index_buffer(&mut a16, &ib16, ic, &vb, vc, STRIDE)?;
    assert_eq!(a32, widen(&a16));

    let mut t32 = vec![0u32; ic * 4];
    let mut t16 = vec![0u16; ic * 4];
    generate_tessellation_index_buffer(&mut t32, &ib, ic, &vb, vc, STRIDE)?;
    generate_tessellation_index_buffer(&mut t16, &ib16, ic, &vb, vc, STRIDE)?;
    assert_eq!(t32, widen(&t16));

    let mut o32 = vec![0u32; vc + ic / 3];
    let mut o16 = vec![0u32; vc + ic / 3];
    let n32 = generate_provoking_index_buffer(&mut d32, &mut o32, &ib, ic, vc)?;
    let n16 = generate_provoking_index_buffer(&mut d16, &mut o16, &ib16, ic, vc)?;
    assert_eq!((n32, &o32, &d32), (n16, &o16, &widen(&d16)));
    Ok(())
}

#[test]
fn optimizers_u16() -> Result<()> {
    let (vb, ib) = grid();
    let ib16 = narrow(&ib);
    let (vc, ic) = (vertex_count(&vb), ib.len());

    let mut d32 = vec![0u32; ic];
    let mut d16 = vec![0u16; ic];

    optimize_vertex_cache(&mut d32, &ib, ic, vc)?;
    optimize_vertex_cache(&mut d16, &ib16, ic, vc)?;
    assert_eq!(d32, widen(&d16));

    let mut i16 = ib16.clone();
    optimize_vertex_cache_in_place(&mut i16, ic, vc)?;
    assert_eq!(d32, widen(&i16));

    optimize_vertex_cache_strip(&mut d32, &ib, ic, vc)?;
    optimize_vertex_cache_strip(&mut d16, &ib16, ic, vc)?;
    assert_eq!(d32, widen(&d16));

    optimize_vertex_cache_fifo(&mut d32, &ib, ic, vc, 16)?;
    optimize_vertex_cache_fifo(&mut d16, &ib16, ic, vc, 16)?;
    assert_eq!(d32, widen(&d16));

    let opt = d32.clone();
    let opt16 = narrow(&opt);
    optimize_overdraw(&mut d32, &opt, ic, &vb, vc, STRIDE, 1.05)?;
    optimize_overdraw(&mut d16, &opt16, ic, &vb, vc, STRIDE, 1.05)?;
    assert_eq!(d32, widen(&d16));

    let mut r32 = vec![0u32; vc];
    let mut r16 = vec![0u32; vc];
    let n32 = optimize_vertex_fetch_remap(&mut r32, &ib, ic, vc)?;
    let n16 = optimize_vertex_fetch_remap(&mut r16, &ib16, ic, vc)?;
    assert_eq!((n32, &r32), (n16, &r16));

    let vbytes = bytes(&vb);
    let (mut v32, mut v16) = (vec![0u8; vbytes.len()], vec![0u8; vbytes.len()]);
    let (mut f32i, mut f16i) = (ib.clone(), ib16.clone());
    let n32 = optimize_vertex_fetch(&mut v32, &mut f32i, ic, &vbytes, vc, STRIDE)?;
    let n16 = optimize_vertex_fetch(&mut v16, &mut f16i, ic, &vbytes, vc, STRIDE)?;
    assert_eq!((n32, &v32, &f32i), (n16, &v16, &widen(&f16i)));

    let mut s32 = vec![0u32; stripify_bound(ic)?];
    let mut s16 = vec![0u16; stripify_bound(ic)?];
    for (restart32, restart16) in [(0u32, 0u16), (!0u32, !0u16)] {
        let n32 = stripify(&mut s32, &opt, ic, vc, restart32)?;
        let n16 = stripify(&mut s16, &opt16, ic, vc, restart16)?;
        let w16 = widen(&s16[..n16]);
        let expected: Vec<u32> = s32[..n32].iter().map(|&i| if i == !0u32 { 0xffff } else { i }).collect();
        assert_eq!(expected, w16);

        let mut u32l = vec![0u32; unstripify_bound(n32)?];
        let mut u16l = vec![0u16; unstripify_bound(n16)?];
        let m32 = unstripify(&mut u32l, &s32, n32, restart32)?;
        let m16 = unstripify(&mut u16l, &s16, n16, restart16)?;
        assert_eq!(&u32l[..m32], &widen(&u16l)[..m16]);
    }
    Ok(())
}

#[test]
fn analyzers_u16() -> Result<()> {
    let (vb, ib) = grid();
    let ib16 = narrow(&ib);
    let (vc, ic) = (vertex_count(&vb), ib.len());

    assert_eq!(analyze_vertex_cache(&ib, ic, vc, 16, 32, 32)?, analyze_vertex_cache(&ib16, ic, vc, 16, 32, 32)?);
    assert_eq!(analyze_vertex_fetch(&ib, ic, vc, STRIDE)?, analyze_vertex_fetch(&ib16, ic, vc, STRIDE)?);
    assert_eq!(analyze_overdraw(&ib, ic, &vb, vc, STRIDE)?, analyze_overdraw(&ib16, ic, &vb, vc, STRIDE)?);
    assert_eq!(analyze_coverage(&ib, ic, &vb, vc, STRIDE)?, analyze_coverage(&ib16, ic, &vb, vc, STRIDE)?);
    Ok(())
}

#[test]
fn u16_narrowing_truncates() -> Result<()> {
    // C++: result[i] = T(data[i]); remapped values above 0xffff wrap like the C cast
    let remap: Vec<u32> = (0..3).map(|i| 0x10000 + i).collect();
    let mut d16 = [0u16; 3];
    remap_index_buffer(&mut d16, Some(&[0u16, 1, 2]), 3, &remap)?;
    assert_eq!(d16, [0, 1, 2]);

    // errors leave entries the function didn't write untouched, like the u32 path
    let mut d16 = [7u16; 3];
    assert_eq!(remap_index_buffer(&mut d16, Some(&[0u16, 5, 2]), 3, &remap), Err(Error::IndexOutOfRange));
    assert_eq!(d16[2], 7);
    Ok(())
}
