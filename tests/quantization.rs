//! Port of quantization tests from `demo/tests.cpp`.
#![allow(clippy::approx_constant, clippy::excessive_precision)]

use meshoptimizer::*;

fn ldexpf(x: f32, n: i32) -> f32 {
    (x as f64 * 2f64.powi(n)) as f32
}

#[test]
fn quantize_float_test() -> Result<()> {
    let zero = std::hint::black_box(0.0f32);

    assert_eq!(quantize_float(1.2345, 23)?, 1.2345);

    assert_eq!(quantize_float(1.2345, 16)?, 1.2344971);
    assert_eq!(quantize_float(1.2345, 8)?, 1.2343750);
    assert_eq!(quantize_float(1.2345, 4)?, 1.25);
    assert_eq!(quantize_float(1.2345, 1)?, 1.0);

    assert_eq!(quantize_float(1.0, 0)?, 1.0);

    assert_eq!(quantize_float(1.0 / zero, 0)?, 1.0 / zero);
    assert_eq!(quantize_float(-1.0 / zero, 0)?, -1.0 / zero);

    let nanf = quantize_float(zero / zero, 8)?;
    assert!(nanf.is_nan());
    Ok(())
}

#[test]
fn quantize_half_test() {
    let zero = std::hint::black_box(0.0f32);

    // normal
    assert_eq!(quantize_half(1.2345), 0x3cf0);

    // overflow
    assert_eq!(quantize_half(65535.0), 0x7c00);
    assert_eq!(quantize_half(-65535.0), 0xfc00);

    // large
    assert_eq!(quantize_half(65000.0), 0x7bef);
    assert_eq!(quantize_half(-65000.0), 0xfbef);

    // small
    assert_eq!(quantize_half(0.125), 0x3000);
    assert_eq!(quantize_half(-0.125), 0xb000);

    // very small
    assert_eq!(quantize_half(1e-4), 0x068e);
    assert_eq!(quantize_half(-1e-4), 0x868e);

    // underflow
    assert_eq!(quantize_half(1e-5), 0x0000);
    assert_eq!(quantize_half(-1e-5), 0x8000);

    // exponent underflow
    assert_eq!(quantize_half(1e-20), 0x0000);
    assert_eq!(quantize_half(-1e-20), 0x8000);

    // exponent overflow
    assert_eq!(quantize_half(1e20), 0x7c00);
    assert_eq!(quantize_half(-1e20), 0xfc00);

    // inf
    assert_eq!(quantize_half(1.0 / zero), 0x7c00);
    assert_eq!(quantize_half(-1.0 / zero), 0xfc00);

    // nan
    let nanh = quantize_half(zero / zero);
    assert!(nanh == 0x7e00 || nanh == 0xfe00);
}

#[test]
fn dequantize_half_test() {
    let zero = std::hint::black_box(0.0f32);

    // normal
    assert_eq!(dequantize_half(0x3cf0), 1.234375);

    // large
    assert_eq!(dequantize_half(0x7bef), 64992.0);
    assert_eq!(dequantize_half(0xfbef), -64992.0);

    // small
    assert_eq!(dequantize_half(0x3000), 0.125);
    assert_eq!(dequantize_half(0xb000), -0.125);

    // very small
    assert_eq!(dequantize_half(0x068e), 1.00016594e-4);
    assert_eq!(dequantize_half(0x868e), -1.00016594e-4);

    // denormal
    assert_eq!(dequantize_half(0x00ff), 0.0);
    assert_eq!(dequantize_half(0x80ff), 0.0); // actually this is -0.0
    assert_eq!(1.0 / dequantize_half(0x80ff), -1.0 / zero);

    // inf
    assert_eq!(dequantize_half(0x7c00), 1.0 / zero);
    assert_eq!(dequantize_half(0xfc00), -1.0 / zero);

    // nan
    let nanf = dequantize_half(0x7e00);
    assert!(nanf.is_nan());
}

fn validate_position_exponent(minv: &[f32; 3], maxv: &[f32; 3], min_exponent: i32, max_bits: i32) -> Result<i32> {
    let exponent = compute_position_exponent(minv, maxv, min_exponent, max_bits)?;

    let anchor_min = -(1i32 << 23);
    let anchor_max = (1i32 << 23) - 1;
    let range_max = (1u32 << max_bits) - 1;

    let scale = ldexpf(1.0, -exponent);

    for k in 0..3 {
        let lo = minv[k] * scale;
        let hi = maxv[k] * scale;

        // lo must fit in 24-bit signed under both round-to-nearest tie-break choices
        let lo_neg = (lo - 0.5).ceil() as i32; // round half toward -inf
        let lo_pos = (lo + 0.5).floor() as i32; // round half toward +inf
        assert!(anchor_min <= lo_neg);
        assert!(lo_neg <= lo_pos);
        assert!(lo_pos <= anchor_max);

        // hi must fit in 24-bit signed under both round-to-nearest tie-break choices
        let hi_neg = (hi - 0.5).ceil() as i32; // round half toward -inf
        let hi_pos = (hi + 0.5).floor() as i32; // round half toward +inf
        assert!(anchor_min <= hi_neg);
        assert!(hi_neg <= hi_pos);
        assert!(hi_pos <= anchor_max);

        // largest hi - smallest lo must fit in max_bits (covers both round-to-nearest tie-break choices)
        assert!((hi_pos - lo_neg) as u32 <= range_max);
    }

    Ok(exponent)
}

#[test]
fn compute_position_exponent_test() -> Result<()> {
    let minv = [-1.0, 2.0, 10.0];
    let maxv = [1.0, 2.5, 20.0];

    assert_eq!(validate_position_exponent(&minv, &maxv, -16, 16)?, -12);
    assert_eq!(validate_position_exponent(&minv, &maxv, -10, 16)?, -10);

    let minirr = [-3.14159265, -2.7182818, -1.41321356];
    let maxirr = [3.14159265, 2.7182818, 1.41321356];

    assert_eq!(validate_position_exponent(&minirr, &maxirr, -16, 16)?, -13);

    let point = [1000000.0, 0.0, 0.0];
    assert_eq!(validate_position_exponent(&point, &point, -16, 16)?, -3);

    let negpoint = [-1000000.0, 0.0, 0.0];
    assert_eq!(validate_position_exponent(&negpoint, &negpoint, -16, 16)?, -3);

    let asym_min = [0.0, -1000.0, 0.0];
    let asym_max = [1e-3, 1000.0, 1.0];
    assert_eq!(validate_position_exponent(&asym_min, &asym_max, -16, 16)?, -5);

    let zero = [0.0, 0.0, 0.0];
    assert_eq!(validate_position_exponent(&zero, &zero, -16, 16)?, -16);

    let range_min = [0.0, 0.0, 0.0];
    let range_max = [65535.0, 0.0, 0.0];
    assert_eq!(validate_position_exponent(&range_min, &range_max, 0, 16)?, 0);

    let range_over = [65536.0, 0.0, 0.0];
    assert_eq!(validate_position_exponent(&range_min, &range_over, 0, 16)?, 1);

    let range_dgf_min = [0.4, 0.0, 0.0];
    let range_dgf_max = [65535.5625, 0.0, 0.0];
    assert_eq!(validate_position_exponent(&range_dgf_min, &range_dgf_max, 0, 16)?, 1);

    let range_half_min = [-1.0, 0.0, 0.0];
    let range_half_max = [131069.0, 0.0, 0.0];
    assert_eq!(validate_position_exponent(&range_half_min, &range_half_max, 0, 16)?, 2);

    let anchor_max = [8388607.0, 0.0, 0.0];
    assert_eq!(validate_position_exponent(&anchor_max, &anchor_max, 0, 16)?, 0);

    let anchor_over = [8388608.0, 0.0, 0.0];
    assert_eq!(validate_position_exponent(&anchor_over, &anchor_over, 0, 16)?, 1);

    let anchor_min = [-8388608.0, 0.0, 0.0];
    assert_eq!(validate_position_exponent(&anchor_min, &anchor_min, 0, 16)?, 1); // note: symmetric range; 0 would be acceptable

    let posunit = [1.0, 0.0, 0.0];
    assert_eq!(validate_position_exponent(&posunit, &posunit, -23, 16)?, -22);

    let negunit = [-1.0, 0.0, 0.0];
    assert_eq!(validate_position_exponent(&negunit, &negunit, -23, 16)?, -22); // note: symmetric range; -23 would be acceptable

    let ties = [0.5, -0.5, 2.5];
    assert_eq!(validate_position_exponent(&ties, &ties, 0, 16)?, 0);

    let tie_min = [-0.5, 0.0, 0.0];
    let tie_max = [0.5, 0.0, 0.0];
    assert_eq!(validate_position_exponent(&tie_min, &tie_max, 0, 16)?, 0);

    let bits21_max = [2097151.0, 0.0, 0.0];
    assert_eq!(validate_position_exponent(&range_min, &bits21_max, 0, 21)?, 0);

    let anchor_half = [8388607.5, 0.0, 0.0];
    assert_eq!(validate_position_exponent(&anchor_half, &anchor_half, 0, 16)?, 1);
    Ok(())
}
