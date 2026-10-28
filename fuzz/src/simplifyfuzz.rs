//! `tools/simplifyfuzz.cpp`: simplifier variants on random topology over a 100 vertex grid.

use meshoptimizer::{
    SIMPLIFY_LOCK_BORDER, SIMPLIFY_PERMISSIVE, SIMPLIFY_PRUNE, SIMPLIFY_SPARSE, simplify, simplify_with_attributes,
    simplify_with_update_interleaved,
};

use crate::crand::CRand;

/// `float vb[100][4]` filled with `rand() % 10` after `srand(seed)`.
pub(crate) fn make_vertices(seed: u8) -> [f32; 400] {
    let mut rng = CRand::new(seed as u32);
    let mut vb = [0.0f32; 400];

    for v in vb.iter_mut() {
        *v = (rng.rand() % 10) as f32;
    }

    vb
}

/// `unsigned int ib[999]` from the input bytes; returns the index count.
pub(crate) fn make_indices(buffer: &[u8]) -> ([u32; 999], usize) {
    let mut ib = [0u32; 999];
    let indices = if buffer.len() - 1 < 999 { (buffer.len() - 1) / 3 * 3 } else { 999 };

    for i in 0..indices {
        ib[i] = (buffer[1 + i] % 100) as u32;
    }

    (ib, indices)
}

pub fn test_one_input(buffer: &[u8]) {
    if buffer.is_empty() {
        return;
    }

    let mut vb = make_vertices(buffer[0]);
    let (mut ib, indices) = make_indices(buffer);

    let mut res = [0u32; 999];

    let _ = simplify(&mut res, &ib, indices, &vb, 100, 16, 0, 1e-1, 0, None);
    let _ = simplify(&mut res, &ib, indices, &vb, 100, 16, 0, f32::MAX, 0, None);
    let _ = simplify(&mut res, &ib, indices, &vb, 100, 16, 0, f32::MAX, SIMPLIFY_LOCK_BORDER, None);
    let _ = simplify(&mut res, &ib, indices, &vb, 100, 16, 0, f32::MAX, SIMPLIFY_SPARSE, None);
    let _ = simplify(&mut res, &ib, indices, &vb, 100, 16, 0, f32::MAX, SIMPLIFY_PRUNE, None);
    let _ = simplify(&mut res, &ib, indices, &vb, 100, 16, 0, f32::MAX, SIMPLIFY_PERMISSIVE, None);

    let aw = [1.0f32];
    let _ = simplify_with_attributes(
        &mut res,
        &ib,
        indices,
        &vb,
        100,
        16,
        &vb[3..],
        16,
        &aw,
        1,
        None,
        0,
        f32::MAX,
        0,
        None,
    );

    // must be last, as it modifies all inputs in place
    let _ = simplify_with_update_interleaved(&mut ib, indices, &mut vb, 100, 16, 3, 16, &aw, 1, None, 0, f32::MAX, 0, None);
}
