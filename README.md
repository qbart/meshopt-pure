# meshopt-pure (Rust port of meshoptimizer)

An **unofficial pure Rust port** of [meshoptimizer](https://github.com/zeux/meshoptimizer): the mesh
optimization library and its companion projects gltfpack and clusterlod. It is not affiliated with or
endorsed by the meshoptimizer authors.

The port follows the C++ API and behaviour closely: every algorithm produces bit-identical output
to the upstream library (checked against the C++ build on real and generated meshes and fuzzed
inputs), and the upstream tests are ported 1:1. Versions follow upstream: 1.3.0 is meshoptimizer
v1.3 (commit `f0a684f`).

## What is included
- `meshopt-pure` (imported as `meshoptimizer`): indexing, vertex cache / overdraw / vertex fetch
  optimization, simplification (including attribute-aware and sloppy), meshlets (clusterizer,
  meshlet utils and codec), partitioning, spatial ordering, stripification, tangent space,
  remeshing, opacity micromaps, vertex / index codecs and filters, quantization, analyzers.
- SIMD decoders (feature `simd`, default): SSE2/SSSE3/SSE4.1 on x86-64, NEON on AArch64, simd128
  on WebAssembly. The CPU is detected at runtime, so no `target-cpu` flags are needed.
- `gltfpack`: library and CLI, byte-identical output to C++ gltfpack, including KTX2 / BasisU
  texture compression (via [ktx-rs](https://crates.io/crates/ktx-rs)).
- `clusterlod`: cluster LOD hierarchies (`demo/clusterlod.h`).
- WebAssembly: everything builds for `wasm32-unknown-unknown` / `wasm32-wasip1`.

Not ported: WebP texture output in gltfpack (`-tw`; it reports "built without WebP support", like
a C++ build without libwebp), the custom allocator API (`meshopt_setAllocator`; use
`#[global_allocator]`), the JS bindings and the internal tuning tools. See
[ROADMAP.md](ROADMAP.md) for the full status.

## Using the library

The package is `meshopt-pure`; the library is imported as `meshoptimizer`:

```toml
[dependencies]
meshopt-pure = "1.3.0"
```

```rust
use meshoptimizer::{
    Result, generate_vertex_remap, optimize_overdraw, optimize_vertex_cache, optimize_vertex_fetch, remap_index_buffer,
    remap_vertex_buffer, simplify,
};

/// Indexes, optimizes and simplifies an unindexed triangle list of `x, y, z` positions.
fn process(positions: &[f32]) -> Result<(Vec<f32>, Vec<u32>, Vec<u32>)> {
    let index_count = positions.len() / 3;
    let vertices: Vec<u8> = positions.iter().flat_map(|v| v.to_le_bytes()).collect();

    // indexing
    let mut remap = vec![0u32; index_count];
    let vertex_count = generate_vertex_remap::<u32>(&mut remap, None, index_count, &vertices, index_count, 12)?;

    let mut indices = vec![0u32; index_count];
    remap_index_buffer::<u32>(&mut indices, None, index_count, &remap)?;
    let mut unique = vec![0u8; vertex_count * 12];
    remap_vertex_buffer(&mut unique, &vertices, index_count, 12, &remap)?;

    // vertex cache, overdraw and vertex fetch
    let source = indices.clone();
    optimize_vertex_cache(&mut indices, &source, index_count, vertex_count)?;

    let floats: Vec<f32> = unique.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let source = indices.clone();
    optimize_overdraw(&mut indices, &source, index_count, &floats, vertex_count, 12, 1.05)?;

    let mut fetched = vec![0u8; unique.len()];
    optimize_vertex_fetch(&mut fetched, &mut indices, index_count, &unique, vertex_count, 12)?;
    let floats: Vec<f32> = fetched.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();

    // a LOD with half of the triangles
    let mut lod = vec![0u32; index_count];
    let mut error = 0.0;
    let lod_count = simplify(&mut lod, &indices, index_count, &floats, vertex_count, 12, index_count / 2, 0.01, 0, Some(&mut error))?;
    lod.truncate(lod_count);

    Ok((floats, indices, lod))
}
```

Functions keep their meshoptimizer names in snake case (`meshopt_optimizeVertexCache` is
`optimize_vertex_cache`) with the same parameters, are generic over `u32` and `u16` indices, and
return `Result<T>` whose error is `meshoptimizer::Error`. The library never panics on bad input
(`unwrap`, `expect` and `panic!` are denied).

## gltfpack

```sh
cargo install gltfpack
gltfpack -i scene.gltf -o scene.glb -cc -tc
gltfpack -h
```

The options and output match C++ gltfpack 1.3. As a library, `gltfpack::run(&args)` takes the same
arguments as the CLI.

## Design notes
- Pure Rust, no C dependencies.
- `unsafe` is denied; the only exception is calling the SIMD decoders after runtime CPU feature
  detection (see [docs/spec/38-simd.md](docs/spec/38-simd.md)).
- Implementation notes per component live in [docs/spec](docs/spec).

## Testing

```sh
cargo test --release   # ported upstream tests, golden outputs, gltfpack and demo cases
make test-aarch64      # NEON under qemu (make cross-setup)
make test-wasm         # simd128 and scalar under wasmtime
make fuzz              # fuzz harnesses (make libfuzzer for coverage-guided runs)
make codecbench        # decoder throughput
```

Test data is versioned in the repository but not part of the published crates.

## License
MIT, like meshoptimizer. The Rust sources are translations of meshoptimizer (Arseny Kapoulkine),
cgltf (Johannes Kuhlmann), fast_obj (Richard Knight), micro_ply (Nick Klingensmith) and sdefl
(Micha Mettke), all MIT licensed. See [LICENSE.md](LICENSE.md) and [NOTICE.md](NOTICE.md); the test
data licenses are listed in NOTICE.md.
