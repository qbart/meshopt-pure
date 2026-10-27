//! libm helpers not provided by `core` for `f32`.

/// C `frexpf`: splits `x` into a mantissa in `[0.5, 1)` and a power of two exponent.
pub(crate) fn frexpf(x: f32) -> (f32, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }

    let mut bits = x.to_bits();
    let mut bias = 0;

    // denormals: normalize first
    if (bits >> 23) & 0xff == 0 {
        bits = (x * f32::from_bits(0x4c00_0000)).to_bits(); // 2^25
        bias = 25;
    }

    let e = ((bits >> 23) & 0xff) as i32 - 126 - bias;
    let m = f32::from_bits((bits & 0x807f_ffff) | (126 << 23));

    (m, e)
}

/// C `ldexpf`: computes `x * 2^n` with a single final rounding.
pub(crate) fn ldexpf(x: f32, n: i32) -> f32 {
    let mut y = x;
    let mut n = n;

    if n > 127 {
        y *= f32::from_bits(0x7f00_0000); // 2^127
        n -= 127;
        if n > 127 {
            y *= f32::from_bits(0x7f00_0000);
            n -= 127;
            n = n.min(127);
        }
    } else if n < -126 {
        // 2^-126 * 2^24 keeps precision for the final denormal rounding
        y *= f32::from_bits(0x0080_0000) * f32::from_bits(0x4b80_0000);
        n += 126 - 24;
        if n < -126 {
            y *= f32::from_bits(0x0080_0000) * f32::from_bits(0x4b80_0000);
            n += 126 - 24;
            n = n.max(-126);
        }
    }

    y * f32::from_bits(((0x7f + n) as u32) << 23)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frexp_ldexp() {
        assert_eq!(frexpf(8.0), (0.5, 4));
        assert_eq!(frexpf(-3.0), (-0.75, 2));
        assert_eq!(frexpf(0.0), (0.0, 0));
        assert_eq!(frexpf(f32::from_bits(1)), (0.5, -148));
        assert_eq!(ldexpf(1.0, -149), f32::from_bits(1));
        assert_eq!(ldexpf(1.0, 127), f32::from_bits(0x7f00_0000));
        assert_eq!(ldexpf(1.0, 128), f32::INFINITY);
        assert_eq!(ldexpf(0.75, 3), 6.0);
    }
}
