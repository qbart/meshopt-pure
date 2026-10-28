//! Animation resampling and constant track detection (`animation.cpp`).

use crate::cgltf::{self, AnimationPathType, Data, InterpolationType};
use crate::cxx::{fmax, fmin};
use crate::types::*;

fn get_scale_delta(l: f32, r: f32) -> f32 {
    let d = (l - r).abs();
    let m = fmax(l.abs(), r.abs());
    if m == 0.0 { 0.0 } else { d / m }
}

fn get_delta(l: &Attr, r: &Attr, type_: AnimationPathType) -> f32 {
    match type_ {
        AnimationPathType::Translation => fmax(fmax((l[0] - r[0]).abs(), (l[1] - r[1]).abs()), (l[2] - r[2]).abs()),
        AnimationPathType::Rotation => 2.0 * fmin(1.0, (l[0] * r[0] + l[1] * r[1] + l[2] * r[2] + l[3] * r[3]).abs()).acos(),
        AnimationPathType::Scale => fmax(
            fmax(get_scale_delta(l[0], r[0]), get_scale_delta(l[1], r[1])),
            get_scale_delta(l[2], r[2]),
        ),
        AnimationPathType::Weights => (l[0] - r[0]).abs(),
        AnimationPathType::Invalid => 0.0,
    }
}

#[allow(clippy::approx_constant)]
fn get_delta_tolerance(type_: AnimationPathType) -> f32 {
    match type_ {
        AnimationPathType::Translation => 0.0001,                   // 0.1mm linear
        AnimationPathType::Rotation => 0.1 * (3.141_592_6 / 180.0), // 0.1 degrees
        AnimationPathType::Scale => 0.001,                          // 0.1% ratio
        AnimationPathType::Weights => 0.001,                        // 0.1% linear
        AnimationPathType::Invalid => 0.0,
    }
}

fn normalize4(v: &mut Attr) {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();

    if len > 0.0 {
        for c in v.iter_mut() {
            *c /= len;
        }
    }
}

fn interpolate_linear(l: &Attr, r: &Attr, t: f32, type_: AnimationPathType) -> Attr {
    if type_ == AnimationPathType::Rotation {
        // Approximating slerp, https://zeux.io/2015/07/23/approximating-slerp/
        // We also handle quaternion double-cover
        let ca = l[0] * r[0] + l[1] * r[1] + l[2] * r[2] + l[3] * r[3];

        let d = ca.abs();
        let a = 1.0904 + d * (-3.2452 + d * (3.55645 - d * 1.43519));
        let b = 0.848013 + d * (-1.06021 + d * 0.215638);
        let k = a * (t - 0.5) * (t - 0.5) + b;
        let ot = t + t * (t - 0.5) * (t - 1.0) * k;

        let t0 = 1.0 - ot;
        let t1 = if ca > 0.0 { ot } else { -ot };

        let mut lerp = [
            l[0] * t0 + r[0] * t1,
            l[1] * t0 + r[1] * t1,
            l[2] * t0 + r[2] * t1,
            l[3] * t0 + r[3] * t1,
        ];

        normalize4(&mut lerp);

        lerp
    } else {
        [
            l[0] * (1.0 - t) + r[0] * t,
            l[1] * (1.0 - t) + r[1] * t,
            l[2] * (1.0 - t) + r[2] * t,
            l[3] * (1.0 - t) + r[3] * t,
        ]
    }
}

fn interpolate_hermite(v0: &Attr, t0: &Attr, v1: &Attr, t1: &Attr, t: f32, dt: f32, type_: AnimationPathType) -> Attr {
    let s0 = 1.0 + t * t * (2.0 * t - 3.0);
    let s1 = t + t * t * (t - 2.0);
    let s2 = 1.0 - s0;
    let s3 = t * t * (t - 1.0);

    let ts1 = dt * s1;
    let ts3 = dt * s3;

    let mut lerp = [0.0; 4];
    for k in 0..4 {
        lerp[k] = s0 * v0[k] + ts1 * t0[k] + s2 * v1[k] + ts3 * t1[k];
    }

    if type_ == AnimationPathType::Rotation {
        normalize4(&mut lerp);
    }

    lerp
}

#[allow(clippy::too_many_arguments)]
fn resample_keyframes(
    data: &mut Vec<Attr>,
    input: &[f32],
    output: &[Attr],
    type_: AnimationPathType,
    interpolation: InterpolationType,
    components: usize,
    frames: i32,
    mint: f32,
    freq: i32,
) {
    // out of range keyframe reads (C++ undefined behavior on malformed tracks) read as zero
    let out = |i: usize| output.get(i).copied().unwrap_or_default();

    let mut cursor = 0usize;

    for i in 0..frames.max(0) {
        let time = mint + i as f32 / freq as f32;

        while cursor + 1 < input.len() {
            let next_time = input[cursor + 1];

            if next_time > time {
                break;
            }

            cursor += 1;
        }

        if cursor + 1 < input.len() {
            let cursor_time = input[cursor];
            let next_time = input[cursor + 1];

            let range = next_time - cursor_time;
            let inv_range = if range == 0.0 { 0.0 } else { 1.0 / (next_time - cursor_time) };
            let t = fmax(0.0, fmin(1.0, (time - cursor_time) * inv_range));

            for j in 0..components {
                match interpolation {
                    InterpolationType::Linear => {
                        let v0 = out(cursor * components + j);
                        let v1 = out((cursor + 1) * components + j);
                        data.push(interpolate_linear(&v0, &v1, t, type_));
                    }
                    InterpolationType::Step => {
                        data.push(out(cursor * components + j));
                    }
                    InterpolationType::CubicSpline => {
                        let v0 = out((cursor * 3 + 1) * components + j);
                        let b0 = out((cursor * 3 + 2) * components + j);
                        let a1 = out((cursor * 3 + 3) * components + j);
                        let v1 = out((cursor * 3 + 4) * components + j);
                        data.push(interpolate_hermite(&v0, &b0, &v1, &a1, t, range, type_));
                    }
                }
            }
        } else {
            let offset = if interpolation == InterpolationType::CubicSpline {
                cursor * 3 + 1
            } else {
                cursor
            };

            for j in 0..components {
                data.push(out(offset * components + j));
            }
        }
    }
}

fn get_max_delta(data: &[Attr], type_: AnimationPathType, value: &[Attr], components: usize) -> f32 {
    let mut result = 0.0f32;

    for chunk in data.chunks(components.max(1)) {
        for (v, d) in value.iter().zip(chunk) {
            let delta = get_delta(v, d, type_);

            result = if result < delta { delta } else { result };
        }
    }

    result
}

fn get_base_transform(result: &mut [Attr], components: usize, type_: AnimationPathType, data: &Data, node: &cgltf::Node) {
    match type_ {
        AnimationPathType::Translation => result[0][..3].copy_from_slice(&node.translation),
        AnimationPathType::Rotation => result[0].copy_from_slice(&node.rotation),
        AnimationPathType::Scale => result[0][..3].copy_from_slice(&node.scale),
        AnimationPathType::Weights => {
            let node_weights = node.weights.as_deref().unwrap_or(&[]);
            let mesh_weights = node.mesh.and_then(|m| data.meshes[m].weights.as_deref()).unwrap_or(&[]);

            let weights = if !node_weights.is_empty() { node_weights } else { mesh_weights };

            for (r, &w) in result.iter_mut().zip(weights).take(components) {
                r[0] = w;
            }
        }
        AnimationPathType::Invalid => {}
    }
}

fn get_world_scale(data: &Data, node: usize) -> f32 {
    let transform = cgltf::node_transform_world(&data.nodes, &data.nodes[node]);

    // 3x3 determinant computes scale^3
    let a0 = transform[5] * transform[10] - transform[6] * transform[9];
    let a1 = transform[4] * transform[10] - transform[6] * transform[8];
    let a2 = transform[4] * transform[9] - transform[5] * transform[8];
    let det = transform[0] * a0 - transform[1] * a1 + transform[2] * a2;

    det.abs().powf(1.0 / 3.0)
}

pub fn process_animation(animation: &mut Animation, data: &Data, settings: &Settings) {
    let mut mint = f32::MAX;
    let mut maxt = 0.0f32;

    for track in &animation.tracks {
        if let (Some(&front), Some(&back)) = (track.time.first(), track.time.last()) {
            mint = fmin(mint, front);
            maxt = fmax(maxt, back);
        }
    }

    mint = fmin(mint, maxt);
    animation.start = mint;

    if settings.anim_freq != 0 {
        // round the number of frames to nearest but favor the "up" direction
        // this means that at 100 Hz resampling, we will try to preserve the last frame <10ms
        // but if the last frame is <2ms we favor just removing this data
        let frames = 1 + crate::cxx::f2i32((maxt - mint) * settings.anim_freq as f32 + 0.8);

        animation.frames = frames;
    }

    let mut base: Vec<Attr> = Vec::new();

    for track in animation.tracks.iter_mut() {
        if settings.anim_freq != 0 {
            let mut result = Vec::new();
            resample_keyframes(
                &mut result,
                &track.time,
                &track.data,
                track.path,
                track.interpolation,
                track.components,
                animation.frames,
                animation.start,
                settings.anim_freq,
            );

            track.time.clear();
            track.data = result;
            if track.interpolation == InterpolationType::CubicSpline {
                track.interpolation = InterpolationType::Linear;
            }
        }

        // getMaxDelta assumes linear/step interpolation for now
        if track.interpolation == InterpolationType::CubicSpline {
            continue;
        }

        let mut tolerance = get_delta_tolerance(track.path);

        // translation tracks use world space tolerance; in the future, we should compute all errors as linear using hierarchy
        if let Some(parent) = data.nodes[track.node].parent
            && track.path == AnimationPathType::Translation
        {
            let scale = get_world_scale(data, parent);
            tolerance /= if scale == 0.0 { 1.0 } else { scale };
        }

        // C++ passes &track.data[0]; tracks shorter than one keyframe (malformed input) compare against zeros
        let mut first: Vec<Attr> = track.data.iter().take(track.components).copied().collect();
        first.resize(track.components, [0.0; 4]);

        let deviation = get_max_delta(&track.data, track.path, &first, track.components);

        if deviation <= tolerance {
            // track is constant (equal to first keyframe), we only need the first keyframe
            track.constant = true;
            track.time.clear();
            track.data.resize(track.components, [0.0; 4]);

            // track.dummy is true iff track redundantly sets up the value to be equal to default node transform
            base.resize(track.components, [0.0; 4]);
            get_base_transform(&mut base, track.components, track.path, data, &data.nodes[track.node]);

            track.dummy = get_max_delta(&track.data, track.path, &base, track.components) <= tolerance;
        }
    }
}
