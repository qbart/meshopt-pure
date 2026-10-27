use crate::error::{Error, Result, ensure};
use crate::math::{frexpf, ldexpf};

/// Quantize a float into half-precision (IEEE-754 fp16) floating point value.
/// Generates +-inf for overflow, preserves NaN, flushes denormals to zero, rounds to nearest.
pub fn quantize_half(v: f32) -> u16 {
    let ui = v.to_bits();

    let s = ((ui >> 16) & 0x8000) as i32;
    let em = (ui & 0x7fff_ffff) as i32;

    // bias exponent and round to nearest; 112 is relative exponent bias (127-15)
    let mut h = (em - (112 << 23) + (1 << 12)) >> 13;

    // underflow: flush to zero; 113 encodes exponent -14
    h = if em < (113 << 23) { 0 } else { h };

    // overflow: infinity; 143 encodes exponent 16
    h = if em >= (143 << 23) { 0x7c00 } else { h };

    // NaN; note that we convert all types of NaN to qNaN
    h = if em > (255 << 23) { 0x7e00 } else { h };

    (s | h) as u16
}

/// Quantize a float into a floating point value with a limited number of significant mantissa bits.
/// Preserves infinities/NaN, flushes denormals to zero, rounds to nearest. `n` must be in 0..=23.
pub fn quantize_float(v: f32, n: i32) -> Result<f32> {
    ensure((0..=23).contains(&n), Error::InvalidArgument("N must be in 0..=23"))?;

    let mut ui = v.to_bits();

    let mask = (1u32 << (23 - n)) - 1;
    let round = (1u32 << (23 - n)) >> 1;

    let e = ui & 0x7f80_0000;
    let rui = ui.wrapping_add(round) & !mask;

    // round all numbers except inf/nan; this is important to make sure nan doesn't overflow into -0
    ui = if e == 0x7f80_0000 { ui } else { rui };

    // flush denormals to zero
    ui = if e == 0 { 0 } else { ui };

    Ok(f32::from_bits(ui))
}

/// Reverse quantization of a half-precision (IEEE-754 fp16) floating point value.
/// Preserves Inf/NaN, flushes denormals to zero.
pub fn dequantize_half(h: u16) -> f32 {
    let s = ((h & 0x8000) as u32) << 16;
    let em = (h & 0x7fff) as i32;

    // bias exponent and pad mantissa with 0; 112 is relative exponent bias (127-15)
    let mut r = (em + (112 << 10)) << 13;

    // denormal: flush to zero
    r = if em < (1 << 10) { 0 } else { r };

    // infinity/NaN; note that we preserve NaN payload as a byproduct of unifying inf/nan cases
    // 112 is an exponent bias fixup; since we already applied it once, applying it twice converts 31 to 255
    r += if em >= (31 << 10) { 112 << 23 } else { 0 };

    f32::from_bits(s | r as u32)
}

/// Compute shared exponent suitable for mesh/cluster position quantization.
/// `minv`/`maxv` are the AABB corners; `min_exp >= -126`, `max_bits` in 2..=24.
pub fn compute_position_exponent(minv: &[f32; 3], maxv: &[f32; 3], min_exp: i32, max_bits: i32) -> Result<i32> {
    ensure(min_exp >= -126, Error::InvalidArgument("min_exp must be >= -126"))?;
    ensure((2..=24).contains(&max_bits), Error::InvalidArgument("max_bits must be in 2..=24"))?;

    let mut exp = min_exp;

    // compute max absolute component to ensure that individual endpoints fit on a 24-bit signed grid
    let mut maxc = 0.0f32;

    for k in 0..3 {
        maxc = if maxc < minv[k].abs() { minv[k].abs() } else { maxc };
        maxc = if maxc < maxv[k].abs() { maxv[k].abs() } else { maxc };
    }

    let (maxc_fr, maxc_exp) = frexpf(maxc);

    // maxc is representable as 2^(maxc_exp-24) * 24-bit *unsigned* integer
    // we have to use a 24-bit *signed* grid, so we have to chop off the last bit
    // however, rounding in the corner case (mantissa is 1.111...) may increase the exponent by 1 so we need to offset it
    let maxc_off = 23 - i32::from(maxc_fr >= 1.0 - f32::EPSILON / 2.0);

    exp = if exp < maxc_exp - maxc_off { maxc_exp - maxc_off } else { exp };

    // compute effective range with conservative rounding, to allow for some ambiguity in the caller's rounding direction
    let scale = ldexpf(1.0, -exp);
    let mut range = 0.0f32;

    for k in 0..3 {
        let a = (minv[k] * scale).floor();
        let v = (maxv[k] * scale).ceil();
        range = if range < v - a { v - a } else { range };
    }

    // range must be representable as max_bits unsigned integer
    let (range_fr, range_exp) = frexpf(range);

    exp += if range_exp > max_bits { range_exp - max_bits } else { 0 };

    // correct range if we are at the rounding boundary that would push us to overflow max_bits
    exp += i32::from(range_exp > max_bits && range_fr >= 1.0 - 1.0 / ((1i32 << max_bits) - 1) as f32);

    Ok(exp)
}

/// Quantize a float in [0..1] range into an N-bit fixed point unorm value.
/// Assumes reconstruction function (q / (2^N-1)); maximum reconstruction error: 1/2^(N+1).
pub fn quantize_unorm(v: f32, n: i32) -> i32 {
    let scale = 1i32.wrapping_shl(n as u32).wrapping_sub(1) as f32;

    let v = if v >= 0.0 { v } else { 0.0 };
    let v = if v <= 1.0 { v } else { 1.0 };

    (v * scale + 0.5) as i32
}

/// Quantize a float in [-1..1] range into an N-bit fixed point snorm value.
/// Assumes reconstruction function (q / (2^(N-1)-1)); maximum reconstruction error: 1/2^N.
pub fn quantize_snorm(v: f32, n: i32) -> i32 {
    let scale = 1i32.wrapping_shl(n.wrapping_sub(1) as u32).wrapping_sub(1) as f32;

    let round = if v >= 0.0 { 0.5 } else { -0.5 };

    let v = if v >= -1.0 { v } else { -1.0 };
    let v = if v <= 1.0 { v } else { 1.0 };

    (v * scale + round) as i32
}
