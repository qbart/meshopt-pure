//! Node marking, remapping and transform decomposition (`node.cpp`).

use crate::cgltf::Data;
use crate::types::*;

/// Walks the parent chain of `node` (inclusive); bounded so that unvalidated cycles terminate.
fn ancestors(data: &Data, node: usize) -> impl Iterator<Item = usize> + '_ {
    std::iter::successors(Some(node), |&n| data.nodes[n].parent).take(data.nodes.len() + 1)
}

pub fn mark_scenes(data: &Data, nodes: &mut [NodeInfo]) {
    for ni in nodes.iter_mut() {
        ni.scene = -1;
    }

    for (i, scene) in data.scenes.iter().enumerate() {
        for &node in &scene.nodes {
            let ni = &mut nodes[node];

            if ni.scene >= 0 {
                ni.scene = -2; // multiple scenes
            } else {
                ni.scene = i as i32;
            }
        }
    }

    for i in 0..data.nodes.len() {
        let root = ancestors(data, i).last().unwrap_or(i);

        nodes[i].scene = nodes[root].scene;
    }
}

pub fn mark_animated(data: &Data, nodes: &mut [NodeInfo], animations: &[Animation]) {
    for animation in animations {
        for track in &animation.tracks {
            // mark nodes that have animation tracks that change their base transform as animated
            if !track.dummy {
                nodes[track.node].animated_path_mask |= 1u32 << (track.path as u32);
            }
        }
    }

    for i in 0..data.nodes.len() {
        let animated = ancestors(data, i).any(|n| nodes[n].animated_path_mask != 0);
        nodes[i].animated |= animated;
    }
}

pub fn mark_needed_nodes(data: &Data, nodes: &mut [NodeInfo], meshes: &[Mesh], animations: &[Animation], settings: &Settings) {
    // mark all joints as kept
    for skin in &data.skins {
        // for now we keep all joints directly referenced by the skin and the entire ancestry tree; we keep names for joints as well
        for &joint in &skin.joints {
            nodes[joint].keep = true;
        }
    }

    // mark all animated nodes as kept
    for animation in animations {
        for track in &animation.tracks {
            if settings.anim_const || !track.dummy {
                nodes[track.node].keep = true;
            }
        }
    }

    // mark all mesh nodes as kept
    for mesh in meshes {
        for &node in &mesh.nodes {
            nodes[node].keep = true;
        }
    }

    // mark all light/camera nodes as kept
    for (i, node) in data.nodes.iter().enumerate() {
        if node.light.is_some() || node.camera.is_some() {
            nodes[i].keep = true;
        }
    }

    // mark all named nodes as needed (if -kn is specified)
    if settings.keep_nodes {
        for (i, node) in data.nodes.iter().enumerate() {
            if node.name.as_deref().is_some_and(|n| !n.is_empty()) {
                nodes[i].keep = true;
            }
        }
    }
}

pub fn remap_nodes(data: &Data, nodes: &mut [NodeInfo], node_offset: &mut usize) {
    // to keep a node, we currently need to keep the entire ancestry chain
    for i in 0..data.nodes.len() {
        if !nodes[i].keep {
            continue;
        }

        for n in ancestors(data, i) {
            nodes[n].keep = true;
        }
    }

    // generate sequential indices for all nodes; they aren't sorted topologically
    for ni in nodes.iter_mut() {
        if ni.keep {
            ni.remap = *node_offset as i32;

            *node_offset += 1;
        }
    }
}

/// `decomposeTransform`: TRS decomposition of a column-major 4x4 matrix.
pub fn decompose_transform(translation: &mut [f32; 3], rotation: &mut [f32; 4], scale: &mut [f32; 3], transform: &[f32; 16]) {
    let m = |r: usize, c: usize| transform[r * 4 + c];

    // extract translation from last row
    *translation = [m(3, 0), m(3, 1), m(3, 2)];

    // compute determinant to determine handedness
    let det = m(0, 0) * (m(1, 1) * m(2, 2) - m(2, 1) * m(1, 2)) - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
        + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));

    let sign = if det < 0.0 { -1.0 } else { 1.0 };

    // recover scale from axis lengths
    scale[0] = (m(0, 0) * m(0, 0) + m(0, 1) * m(0, 1) + m(0, 2) * m(0, 2)).sqrt() * sign;
    scale[1] = (m(1, 0) * m(1, 0) + m(1, 1) * m(1, 1) + m(1, 2) * m(1, 2)).sqrt() * sign;
    scale[2] = (m(2, 0) * m(2, 0) + m(2, 1) * m(2, 1) + m(2, 2) * m(2, 2)).sqrt() * sign;

    // normalize axes to get a pure rotation matrix
    let rsx = if scale[0] == 0.0 { 0.0 } else { 1.0 / scale[0] };
    let rsy = if scale[1] == 0.0 { 0.0 } else { 1.0 / scale[1] };
    let rsz = if scale[2] == 0.0 { 0.0 } else { 1.0 / scale[2] };

    let (r00, r10, r20) = (m(0, 0) * rsx, m(1, 0) * rsy, m(2, 0) * rsz);
    let (r01, r11, r21) = (m(0, 1) * rsx, m(1, 1) * rsy, m(2, 1) * rsz);
    let (r02, r12, r22) = (m(0, 2) * rsx, m(1, 2) * rsy, m(2, 2) * rsz);

    // "branchless" version of Mike Day's matrix to quaternion conversion
    let qc: usize = if r22 < 0.0 {
        if r00 > r11 { 0 } else { 1 }
    } else if r00 < -r11 {
        2
    } else {
        3
    };
    let qs1 = if qc & 2 != 0 { -1.0 } else { 1.0 };
    let qs2 = if qc & 1 != 0 { -1.0 } else { 1.0 };
    let qs3 = if (qc as i32 - 1) & 2 != 0 { -1.0 } else { 1.0 };

    let qt = 1.0 - qs3 * r00 - qs2 * r11 - qs1 * r22;
    let qs = 0.5 / qt.sqrt();

    rotation[qc] = qs * qt;
    rotation[qc ^ 1] = qs * (r01 + qs1 * r10);
    rotation[qc ^ 2] = qs * (r20 + qs2 * r02);
    rotation[qc ^ 3] = qs * (r12 + qs3 * r21);
}
