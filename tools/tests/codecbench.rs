//! codecbench setup checked against the C++ tool (FNV-1a 64 of the optimized/encoded streams and packed meshlets,
//! dumped by an out-of-repo build of `tools/codecbench.cpp` with `N` overridden).

use meshoptimizer_tools::codecbench::{bench_files, bench_synthetic, make_grid, prepare_codecs, prepare_meshlets, read_file};

fn fnv(data: &[u8]) -> u64 {
    data.iter()
        .fold(0xcbf29ce484222325, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3))
}

fn bytes32(data: &[u32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

struct Expected {
    n: usize,
    vc: u64,
    ic: u64,
    vb: u64,
    ib: u64,
    sizes: (usize, usize),
    // (hash, size, totals) with and without vertex references
    meshlets: [(u64, usize, usize); 2],
    meshlet_count: usize,
}

const EXPECTED: [Expected; 4] = [
    Expected {
        n: 1,
        vc: 0xeec315eb66fb13db,
        ic: 0xc0767df50c15e997,
        vb: 0x09a7cbc1fbe73d4c,
        ib: 0x12101b55a9b77006,
        sizes: (141, 19),
        meshlets: [(0xc04c7b88a31812df, 20, 22), (0x48ff731143f603bb, 20, 6)],
        meshlet_count: 1,
    },
    Expected {
        n: 7,
        vc: 0x0380e1082309e28f,
        ic: 0xd749237933b9a1ab,
        vb: 0xdcefe4307e979532,
        ib: 0x0e05aa1ef313ae71,
        sizes: (1330, 121),
        meshlets: [(0x29fa3099614e914f, 128, 562), (0xa025e6e7451b88c7, 109, 294)],
        meshlet_count: 2,
    },
    Expected {
        n: 64,
        vc: 0x71f7851b6d01cbe1,
        ic: 0xe0f2c04aa8613112,
        vb: 0x7d8991aadb4bcfac,
        ib: 0xd270adbdc314bba1,
        sizes: (85371, 9240),
        meshlets: [(0xe7266aaf41adffe0, 11660, 46580), (0xa23707b60dd81239, 7515, 24576)],
        meshlet_count: 87,
    },
    Expected {
        n: 100,
        vc: 0x661f0b571359d2df,
        ic: 0xa08fa4c5b1fd88fc,
        vb: 0xed97537207c950d0,
        ib: 0x966b2c1165018f54,
        sizes: (205932, 22542),
        meshlets: [(0x9ffe23fc47e6b587, 28978, 113860), (0xfbd3bc19faa3a4aa, 18339, 60000)],
        meshlet_count: 212,
    },
];

#[test]
fn setup_matches_reference() -> Result<(), String> {
    for e in &EXPECTED {
        let grid = make_grid(e.n);

        let codecs = prepare_codecs(&grid.vertices, &grid.indices)?;
        assert_eq!((codecs.vc.len(), codecs.ic.len()), e.sizes, "n={}", e.n);
        assert_eq!(fnv(&codecs.vc), e.vc, "vc n={}", e.n);
        assert_eq!(fnv(&codecs.ic), e.ic, "ic n={}", e.n);
        assert_eq!(fnv(&codecs.vb), e.vb, "vb n={}", e.n);
        assert_eq!(fnv(&bytes32(&codecs.ib)), e.ib, "ib n={}", e.n);

        for (encoderefs, &(hash, size, totals)) in [true, false].into_iter().zip(&e.meshlets) {
            let m = prepare_meshlets(&grid.positions, &grid.indices, encoderefs)?;
            assert_eq!(m.meshlet_count, e.meshlet_count, "n={} refs={encoderefs}", e.n);
            assert_eq!((m.packed.len(), m.totals), (size, totals), "n={} refs={encoderefs}", e.n);
            assert_eq!(fnv(&m.packed), hash, "n={} refs={encoderefs}", e.n);
        }
    }

    Ok(())
}

#[test]
fn synthetic_runs() -> Result<(), String> {
    bench_synthetic(8, 1, true)
}

#[test]
fn file_mode() -> Result<(), String> {
    let dir = meshoptimizer_tools::temp_dir().join(format!("codecbench-{}", meshoptimizer_tools::unique_suffix()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let grid = make_grid(8);
    let codecs = prepare_codecs(&grid.vertices, &grid.indices)?;

    let raw = dir.join("v81_s32_raw.bin");
    let encoded = dir.join("v81_s32_encoded.bin");
    let index = dir.join(format!("i{}_mesh.bin", codecs.index_count));
    let unknown = dir.join("mesh.bin");
    std::fs::write(&raw, &codecs.vb).map_err(|e| e.to_string())?;
    std::fs::write(&encoded, &codecs.vc).map_err(|e| e.to_string())?;
    std::fs::write(&index, &codecs.ic).map_err(|e| e.to_string())?;
    std::fs::write(&unknown, [0u8]).map_err(|e| e.to_string())?;

    let path = |p: &std::path::Path| p.to_string_lossy().into_owned();
    let (raw, encoded, index, unknown) = (path(&raw), path(&encoded), path(&index), path(&unknown));

    // raw and encoded inputs re-encode to the same stream
    let a = read_file(&raw)?;
    let b = read_file(&encoded)?;
    assert_eq!((a.count, a.stride, a.index), (81, 32, false));
    assert_eq!(a.data, codecs.vc);
    assert_eq!(b.data, codecs.vc);

    let i = read_file(&index)?;
    assert_eq!((i.count, i.stride, i.index), (codecs.index_count, 2, true));

    assert!(read_file(&unknown).is_err());

    let result = bench_files(&[&raw, &encoded, &index], 1);
    let _ = std::fs::remove_dir_all(&dir);
    result
}
