//! `demo/nanite.cpp`: builds a cluster LOD DAG with clusterlod and prints per-level statistics.
//! For reference, see the original Nanite paper: Brian Karis. Nanite: A Deep Dive. 2021

use clusterlod::{
    ClodBounds, ClodGroup, ClodMesh, ClodNode, clod_build, clod_build_hierarchy, clod_build_hierarchy_bound, clod_default_config,
    clod_default_config_rt,
};
use gltfpack::cstd::{atof, atoi};
use meshoptimizer::generate_position_remap;

use crate::demo::{Result, VERTEX_SIZE, Vertex, dump_obj, dump_obj_section, f, floats};

/// computes approximate (perspective) projection error of a cluster in screen space (0..1; multiply by screen height to get pixels)
/// camera_proj is projection[1][1], or cot(fovy/2); camera_znear is *positive* near plane distance
/// for DAG cut to be valid, boundsError must be monotonic: it must return a larger error for parent cluster
/// for simplicity, we ignore perspective distortion and use rotationally invariant projection size estimation
fn bounds_error(bounds: &ClodBounds, camera_x: f32, camera_y: f32, camera_z: f32, camera_proj: f32, camera_znear: f32) -> f32 {
    let (dx, dy, dz) = (
        bounds.center[0] - camera_x,
        bounds.center[1] - camera_y,
        bounds.center[2] - camera_z,
    );
    let d = (dx * dx + dy * dy + dz * dz).sqrt() - bounds.radius;
    bounds.error / (if d > camera_znear { d } else { camera_znear }) * (camera_proj * 0.5)
}

/// Viewpoint of the cut: camera position, projection and error threshold.
#[derive(Clone, Copy)]
struct View {
    x: f32,
    y: f32,
    z: f32,
    proj: f32,
    znear: f32,
    threshold: f32,
}

impl View {
    fn error(&self, bounds: &ClodBounds) -> f32 {
        bounds_error(bounds, self.x, self.y, self.z, self.proj, self.znear)
    }
}

/// simulate BVH based LOD cut and return triangle count
fn count_triangles(
    nodes: &[ClodNode],
    index: usize,
    groups: &[ClodGroup],
    group_clusters: &[(u32, u32)],
    clusters: &[(i32, u32)],
    view: &View,
) -> usize {
    let node = &nodes[index];

    // prune the subtree once it's too detailed
    if view.error(&node.bounds) <= view.threshold {
        return 0;
    }

    let mut tris = 0;

    if node.group < 0 {
        // internal node: recurse into children
        for i in 0..node.child_count {
            tris += count_triangles(nodes, (node.child_offset + i) as usize, groups, group_clusters, clusters, view);
        }
    } else {
        // leaf node: keep each cluster that is the least detailed one passing the threshold
        let (first, count) = group_clusters[node.group as usize];
        for &(refined, triangles) in &clusters[first as usize..(first + count) as usize] {
            if refined < 0 || view.error(&groups[refined as usize].simplified) <= view.threshold {
                tris += triangles as usize;
            }
        }
    }

    tris
}

#[derive(Default)]
struct Stats {
    groups: usize,
    clusters: usize,
    triangles: usize,
    vertices: usize,

    full_clusters: usize,
    singleton_groups: usize,

    stuck_clusters: usize,
    stuck_triangles: usize,

    radius: f64,

    indices: Vec<Vec<u32>>, // for detailed connectivity analysis
}

pub fn nanite(vertices: &[Vertex], indices: &[u32]) -> Result<()> {
    let dump = std::env::var("DUMP").ok();
    let clrt = std::env::var("CLRT").ok();

    let mut config = clod_default_config(/*max_triangles=*/ 128)?;

    if let Some(clrt) = &clrt {
        config = clod_default_config_rt(config.max_triangles)?;
        config.cluster_fill_weight = atof(clrt.as_bytes()) as f32;
    }

    let attribute_weights = [0.5f32, 0.5, 0.5];

    let data = floats(vertices);

    let mesh = ClodMesh {
        indices,
        index_count: indices.len(),
        vertex_count: vertices.len(),
        vertex_positions: &data,
        vertex_positions_stride: VERTEX_SIZE,
        vertex_attributes: Some(&data[3..]),
        vertex_attributes_stride: VERTEX_SIZE,
        vertex_lock: None,
        attribute_weights: Some(&attribute_weights),
        attribute_count: attribute_weights.len(),
        attribute_protect_mask: (1 << 3) | (1 << 4), // protect UV seams, maps to Vertex::tx/ty
    };

    let mut stats: Vec<Stats> = Vec::new();
    let mut groups: Vec<ClodGroup> = Vec::new();

    let mut group_clusters: Vec<(u32, u32)> = Vec::new(); // per group: (offset, count) into cluster_data
    let mut cluster_data: Vec<(i32, u32)> = Vec::new(); // per cluster: (clodCluster::refined, triangle count)

    let mut cut: Vec<Vec<u32>> = Vec::new();
    let cut_level = dump.as_ref().map_or(-2, |d| atoi(d.as_bytes()));

    // for testing purposes, we can compute a DAG cut from a given viewpoint and dump it as an OBJ
    let (mut maxx, mut maxy, mut maxz) = (0f32, 0f32, 0f32);
    for v in vertices {
        maxx = maxx.max(v.px * 2.0);
        maxy = maxy.max(v.py * 2.0);
        maxz = maxz.max(v.pz * 2.0);
    }

    let threshold = 2e-3f32; // 2 pixels at 1080p
    let fovy = 60f32;
    let znear = 1e-2f32;
    // C++ uses this approximation of pi; the cut results depend on it
    #[allow(clippy::approx_constant)]
    let proj = 1.0 / (fovy * 3.1415926 / 180.0 * 0.5).tan();

    let view = View {
        x: maxx,
        y: maxy,
        z: maxz,
        proj,
        znear,
        threshold,
    };

    // C++ calls meshopt_setAllocatorCache here; the allocator API is not ported (docs/spec/00-conventions.md)

    let max_triangles = config.max_triangles;

    clod_build(config, mesh, |group, clusters, _positions| {
        if stats.len() <= group.depth as usize {
            stats.push(Stats::default());
        }

        let level = &mut stats[group.depth as usize];

        level.groups += 1;
        level.clusters += clusters.len();
        if group.simplified.error == f32::MAX {
            level.stuck_clusters += clusters.len();
        }
        level.singleton_groups += usize::from(clusters.len() == 1);

        group_clusters.push((cluster_data.len() as u32, clusters.len() as u32));

        for cluster in clusters {
            cluster_data.push((cluster.refined, (cluster.index_count / 3) as u32));

            level.triangles += cluster.index_count / 3;
            if group.simplified.error == f32::MAX {
                level.stuck_triangles += cluster.index_count / 3;
            }
            level.vertices += cluster.vertex_count;

            level.full_clusters += usize::from(cluster.index_count == max_triangles * 3);
            level.radius += f64::from(cluster.bounds.radius);

            let cluster_indices = cluster.indices[..cluster.index_count].to_vec();

            // when requesting DAG cut at a given level, we need to render all terminal clusters at lower depth as well
            if cut_level >= 0 && (group.depth == cut_level || (group.depth < cut_level && group.simplified.error == f32::MAX)) {
                cut.push(cluster_indices.clone());
            }

            // when requesting DAG cut from a viewpoint, we need to check if each cluster is the least detailed cluster that passes the error threshold
            if cut_level == -1
                && (cluster.refined < 0 || view.error(&groups[cluster.refined as usize].simplified) <= threshold)
                && view.error(&group.simplified) > threshold
            {
                cut.push(cluster_indices.clone());
            }

            level.indices.push(cluster_indices);
        }

        level.indices.push(Vec::new()); // mark end of group for measureBoundary

        groups.push(group);
        (groups.len() - 1) as i32
    })?;

    // for cluster connectivity analysis and boundary statistics, we need a position-only remap that maps vertices with the same position to the same index
    let mut remap = vec![0u32; vertices.len()];
    generate_position_remap(&mut remap, &data, vertices.len(), VERTEX_SIZE)?;

    let mut used = vec![0i32; vertices.len()];

    let mut lowest_clusters = 0;
    let mut lowest_triangles = 0;

    for (i, level) in stats.iter().enumerate() {
        lowest_clusters += level.stuck_clusters;
        lowest_triangles += level.stuck_triangles;

        let mut connected = 0usize;
        for cluster in &level.indices {
            connected += measure_components(&mut used, cluster, &remap) as usize;
        }

        let boundary = measure_boundary(&mut used, &level.indices, &remap);
        let saho = if clrt.is_some() {
            sah_overhead(&level.indices, vertices)
        } else {
            0.0
        };

        let inv_clusters = 1.0 / level.clusters as f64;

        print!(
            "lod {}: {} clusters ({}% full, {} tri/cl, {} vtx/cl, {} connected, {} boundary, {} partition, {} singletons, {} sah overhead, {} radius), {} triangles",
            i,
            level.clusters,
            f(level.full_clusters as f64 * inv_clusters * 100.0, 1),
            f(level.triangles as f64 * inv_clusters, 1),
            f(level.vertices as f64 * inv_clusters, 1),
            f(connected as f64 * inv_clusters, 2),
            f(boundary as f64 * inv_clusters, 1),
            f(level.clusters as f64 / level.groups as f64, 1),
            level.singleton_groups,
            f(f64::from(saho), 3),
            f(level.radius * inv_clusters, 6),
            level.triangles
        );
        if level.stuck_clusters != 0 && level.clusters > 1 {
            print!("; stuck {} clusters ({} triangles)", level.stuck_clusters, level.stuck_triangles);
        }
        println!();
    }

    println!("lowest lod: {lowest_clusters} clusters, {lowest_triangles} triangles");

    // build a spatial hierarchy over the groups and use it to compute the cut size from a given viewpoint
    if let Some(last) = groups.last() {
        let node_width = 8;
        let levels = last.depth as usize + 1; // groups are generated in order of increasing depth

        let mut nodes = vec![ClodNode::default(); clod_build_hierarchy_bound(groups.len(), node_width, levels)?];
        let count = clod_build_hierarchy(&mut nodes, &groups, groups.len(), node_width, levels)?;
        nodes.truncate(count);

        let mut bvh_tris = 0;
        for i in 0..levels {
            // the hierarchy is a forest with one root per level
            bvh_tris += count_triangles(&nodes, i, &groups, &group_clusters, &cluster_data, &view);
        }
        println!("bvh cut (error {}): {} triangles", f(threshold.into(), 3), bvh_tris);
    }

    if cut_level >= -1 {
        let cut_tris: usize = cut.iter().map(|cluster| cluster.len() / 3).sum();

        if cut_level >= 0 {
            println!("cut (level {cut_level}): {cut_tris} triangles");
        } else {
            println!("cut (error {}): {} triangles", f(threshold.into(), 3), cut_tris);
        }

        dump_obj(vertices, &[], false);

        for cluster in &cut {
            dump_obj_section("cluster", cluster);
        }
    }

    Ok(())
}

// What follows is code that is helpful for collecting metrics, visualizing cuts, etc.
// This code is not used in the actual clustering implementation and can be ignored.
fn follow(parents: &mut [i32], mut index: i32) -> i32 {
    while index != parents[index as usize] {
        let parent = parents[index as usize];
        parents[index as usize] = parents[parent as usize];
        index = parent;
    }

    index
}

fn measure_components(parents: &mut [i32], indices: &[u32], remap: &[u32]) -> i32 {
    assert!(parents.len() == remap.len());

    for &i in indices {
        let v = remap[i as usize];
        parents[v as usize] = v as i32;
    }

    for i in 0..indices.len() {
        let mut v0 = remap[indices[i] as usize] as i32;
        let mut v1 = remap[indices[if i % 3 == 2 { i - 2 } else { i + 1 }] as usize] as i32;

        v0 = follow(parents, v0);
        v1 = follow(parents, v1);

        parents[v0 as usize] = v1;
    }

    for &i in indices {
        let v = remap[i as usize] as i32;
        parents[v as usize] = follow(parents, v);
    }

    let mut roots = 0;
    for &i in indices {
        let v = remap[i as usize] as usize;
        roots += i32::from(parents[v] == v as i32);
        parents[v] = -1; // make sure we only count each root once
    }

    roots
}

fn measure_boundary(used: &mut [i32], clusters: &[Vec<u32>], remap: &[u32]) -> usize {
    used.fill(-1);

    // mark vertices that are used by multiple groups with -2
    let mut group = 0;

    for cluster in clusters {
        group += i32::from(cluster.is_empty());

        for &i in cluster {
            let v = remap[i as usize] as usize;

            used[v] = if used[v] == -1 || used[v] == group { group } else { -2 };
        }
    }

    let mut result = 0;

    for cluster in clusters {
        // count vertices that are used by multiple groups and change marks to -1
        for &i in cluster {
            let v = remap[i as usize] as usize;

            result += usize::from(used[v] == -2);
            used[v] = if used[v] == -2 { -1 } else { used[v] };
        }

        // change marks back from -1 to -2 for the next pass
        for &i in cluster {
            let v = remap[i as usize] as usize;

            used[v] = if used[v] == -1 { -2 } else { used[v] };
        }
    }

    result
}

#[derive(Clone, Copy)]
struct Box3 {
    min: [f32; 3],
    max: [f32; 3],
}

const DUMMY_BOX: Box3 = Box3 {
    min: [f32::MAX; 3],
    max: [-f32::MAX; 3],
};

fn merge_box(b: &mut Box3, other: &Box3) {
    for k in 0..3 {
        b.min[k] = if other.min[k] < b.min[k] { other.min[k] } else { b.min[k] };
        b.max[k] = if other.max[k] > b.max[k] { other.max[k] } else { b.max[k] };
    }
}

fn surface(b: &Box3) -> f32 {
    let (sx, sy, sz) = (b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
    sx * sy + sx * sz + sy * sz
}

fn sah_cost_split(boxes: &[Box3], order: &mut [u32], temp: &mut [u32]) -> f32 {
    const BINS: usize = 15;

    let count = order.len();

    let mut total = boxes[order[0] as usize];
    for &index in &order[1..] {
        merge_box(&mut total, &boxes[index as usize]);
    }

    let mut best_axis = None;
    let mut best_bin = 0;
    let mut best_cost = f32::MAX;

    let bin_of = |b: &Box3, axis: usize, extent: f32| -> usize {
        let p = (b.min[axis] + b.max[axis]) * 0.5;
        let bin = ((p - total.min[axis]) / extent * (BINS - 1) as f32 + 0.5) as i32;
        assert!(bin >= 0 && (bin as usize) < BINS);
        bin as usize
    };

    for axis in 0..3 {
        let mut bins = [DUMMY_BOX; BINS];
        let mut counts = [0u32; BINS];

        let extent = total.max[axis] - total.min[axis];
        if extent <= 0.0 {
            continue;
        }

        for &index in order.iter() {
            let bin = bin_of(&boxes[index as usize], axis, extent);

            merge_box(&mut bins[bin], &boxes[index as usize]);
            counts[bin] += 1;
        }

        let (mut laccum, mut raccum) = (DUMMY_BOX, DUMMY_BOX);
        let (mut lcount, mut rcount) = (0usize, 0usize);
        let mut costs = [0f32; BINS];

        for i in 0..BINS - 1 {
            merge_box(&mut laccum, &bins[i]);
            merge_box(&mut raccum, &bins[BINS - 1 - i]);

            lcount += counts[i] as usize;
            costs[i] += if lcount != 0 { surface(&laccum) * lcount as f32 } else { 0.0 };
            rcount += counts[BINS - 1 - i] as usize;
            costs[BINS - 2 - i] += if rcount != 0 { surface(&raccum) * rcount as f32 } else { 0.0 };
        }

        for (i, &cost) in costs[..BINS - 1].iter().enumerate() {
            if cost < best_cost {
                best_cost = cost;
                best_bin = i;
                best_axis = Some(axis);
            }
        }
    }

    let Some(best_axis) = best_axis else {
        return surface(&total) * count as f32;
    };

    let best_extent = total.max[best_axis] - total.min[best_axis];

    let (mut offset0, mut offset1) = (0, count);

    for &index in order.iter() {
        let bin = bin_of(&boxes[index as usize], best_axis, best_extent);

        if bin <= best_bin {
            temp[offset0] = index;
            offset0 += 1;
        } else {
            offset1 -= 1;
            temp[offset1] = index;
        }
    }

    assert!(offset0 == offset1);

    if offset0 == 0 || offset0 == count {
        return surface(&total) * count as f32;
    }

    let (temp0, temp1) = temp.split_at_mut(offset0);
    let (order0, order1) = order.split_at_mut(offset0);
    surface(&total) + sah_cost_split(boxes, temp0, order0) + sah_cost_split(boxes, temp1, order1)
}

fn sah_cost(boxes: &[Box3]) -> f32 {
    if boxes.is_empty() {
        return 0.0;
    }

    let mut order: Vec<u32> = (0..boxes.len() as u32).collect();
    let mut temp = vec![0u32; boxes.len()];
    sah_cost_split(boxes, &mut order, &mut temp)
}

fn sah_overhead(clusters: &[Vec<u32>], vertices: &[Vertex]) -> f32 {
    let mut all_tris = Vec::new();
    let mut cluster_tris = Vec::new();
    let mut cluster_boxes = Vec::new();

    let mut sahc = 0f32;

    for cluster in clusters {
        if cluster.is_empty() {
            continue;
        }

        cluster_tris.clear();

        let mut cluster_box = DUMMY_BOX;

        for tri in cluster.as_chunks::<3>().0 {
            let mut b = DUMMY_BOX;

            for &v in tri {
                let vertex = &vertices[v as usize];

                let p = Box3 {
                    min: [vertex.px, vertex.py, vertex.pz],
                    max: [vertex.px, vertex.py, vertex.pz],
                };
                merge_box(&mut b, &p);
            }

            merge_box(&mut cluster_box, &b);

            all_tris.push(b);
            cluster_tris.push(b);
        }

        cluster_boxes.push(cluster_box);

        sahc += sah_cost(&cluster_tris);
        sahc -= surface(&cluster_box); // box will be accounted for in tlas
    }

    sahc += sah_cost(&cluster_boxes);

    let saht = sah_cost(&all_tris);

    sahc / saht
}
