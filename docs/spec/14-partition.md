# 14 — Cluster partitioner (`src/partition.cpp`)

| C | Rust |
|---|---|
| `meshopt_partitionClusters` | `partition_clusters(destination, cluster_indices, total_index_count, cluster_index_counts, cluster_count, vertex_positions: Option<&[f32]>, vertex_count, stride, target_partition_size) -> Result<usize>` |

- `NULL` positions → `None` (connectivity-only partitioning).
- Extra validation (C asserts): counts sum to `total_index_count`, no empty clusters, indices `< vertex_count`.
- Allocator-shared scratch (`order` reused as `merge_order`) becomes separate `Vec`s.
- Tests: `partitionBasic`, `partitionSpatial`, `partitionSpatialMerge` → `tests/partition.rs`.
- Diff-tested bit-identical on meshlet clusters of all test meshes (targets 1/2/4/8/16, with and without positions).
