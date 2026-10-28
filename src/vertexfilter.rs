//! Vertex buffer filters; only the scalar code paths of the C++ implementation are ported.
//! Buffers are byte slices holding little-endian 8/16/32-bit components, like the C `void*` buffers.

use crate::error::{Error, Result, ensure};
use crate::quantization::{quantize_snorm, quantize_unorm};

/// Exponent sharing mode for [`encode_filter_exp`] (C: `meshopt_EncodeExpMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeExpMode {
    /// When encoding exponents, use separate values for each component (maximum quality)
    Separate,
    /// When encoding exponents, use shared value for all components of each vector (better compression)
    SharedVector,
    /// When encoding exponents, use shared value for each component of all vectors (best compression)
    SharedComponent,
    /// When encoding exponents, use separate values for each component, but clamp to 0 (good quality if very small values are not important)
    Clamped,
}

fn set16(buffer: &mut [u8], i: usize, v: u16) {
    buffer[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
}

fn set32(buffer: &mut [u8], i: usize, v: u32) {
    buffer[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
}

/// Rounded signed float->int of `v * s`, rounding direction taken from the sign of `v` like the C code.
fn rounded(v: f32, s: f32) -> i32 {
    (v * s + if v >= 0.0 { 0.5 } else { -0.5 }) as i32
}

/// Component `k` of a 4-component element with `S / 4`-byte components (`S` = 4 or 8), zero extended.
/// `S` is a constant, so element accesses compile to fixed offsets without bounds checks.
#[inline(always)]
fn load<const S: usize>(c: &[u8; S], k: usize) -> u32 {
    if S == 4 {
        u32::from(c[k])
    } else {
        u32::from(u16::from_le_bytes([c[k * 2], c[k * 2 + 1]]))
    }
}

/// Sign extended variant of [`load`].
#[inline(always)]
fn load_signed<const S: usize>(c: &[u8; S], k: usize) -> i32 {
    if S == 4 {
        i32::from(c[k] as i8)
    } else {
        i32::from(i16::from_le_bytes([c[k * 2], c[k * 2 + 1]]))
    }
}

/// Stores the low `S / 4` bytes of `v` into component `k`.
#[inline(always)]
fn store<const S: usize>(c: &mut [u8; S], k: usize, v: i32) {
    if S == 4 {
        c[k] = v as u8;
    } else {
        c[k * 2..k * 2 + 2].copy_from_slice(&(v as u16).to_le_bytes());
    }
}

/// `S` is the element stride: 4 (8-bit components) or 8 (16-bit components); components are signed.
fn decode_oct<const S: usize>(data: &mut [u8], count: usize) {
    let max = ((1i32 << (S * 2 - 1)) - 1) as f32;

    for c in data[..count * S].as_chunks_mut::<S>().0 {
        // convert x and y to floats and reconstruct z; this assumes zf encodes 1.f at the same bit count
        let mut x = load_signed(c, 0) as f32;
        let mut y = load_signed(c, 1) as f32;
        let z = load_signed(c, 2) as f32 - x.abs() - y.abs();

        // fixup octahedral coordinates for z<0
        let t = if z >= 0.0 { 0.0 } else { z };

        x += if x >= 0.0 { t } else { -t };
        y += if y >= 0.0 { t } else { -t };

        // compute normal length & scale
        let l = (x * x + y * y + z * z).sqrt();
        let s = max / l;

        // rounded signed float->int
        store(c, 0, rounded(x, s));
        store(c, 1, rounded(y, s));
        store(c, 2, rounded(z, s));
    }
}

fn decode_quat(data: &mut [u8], count: usize) {
    let scale = 32767.0 / 2.0f32.sqrt();

    for c in data[..count * 8].as_chunks_mut::<8>().0 {
        let q3 = load_signed(c, 3);

        // recover scale from the high byte/word of the component
        let sf = q3 | 3;
        let s = sf as f32;

        // convert x/y/z to floating point (unscaled! implied scale of 1/sqrt(2.f) * 1/sf)
        let x = load_signed(c, 0) as f32;
        let y = load_signed(c, 1) as f32;
        let z = load_signed(c, 2) as f32;

        // reconstruct w as a square root (unscaled); we clamp to 0.f to avoid NaN due to precision errors
        let ws = s * s;
        let ww = ws * 2.0 - x * x - y * y - z * z;
        let w = (if ww >= 0.0 { ww } else { 0.0 }).sqrt();

        // compute final scale; note that all computations above are unscaled
        // we need to divide by sf to get out of fixed point, divide by sqrt(2) to renormalize and multiply by 32767 to get to int16 range
        let ss = scale / s;

        // rounded signed float->int
        let xf = rounded(x, ss);
        let yf = rounded(y, ss);
        let zf = rounded(z, ss);
        let wf = (w * ss + 0.5) as i32;

        let qc = (q3 & 3) as usize;

        // output order is dictated by input index
        store(c, (qc + 1) & 3, xf);
        store(c, (qc + 2) & 3, yf);
        store(c, (qc + 3) & 3, zf);
        store(c, qc & 3, wf);
    }
}

fn decode_exp(data: &mut [u8], count: usize) {
    for c in data[..count * 4].as_chunks_mut::<4>().0 {
        let v = u32::from_le_bytes(*c);

        // decode mantissa and exponent
        let m = ((v << 8) as i32) >> 8;
        let e = (v as i32) >> 24;

        // optimized version of ldexp(float(m), e)
        let f = f32::from_bits(((e + 127) as u32) << 23) * m as f32;

        *c = f.to_bits().to_le_bytes();
    }
}

/// `S` is the element stride: 4 (8-bit components) or 8 (16-bit components); co/cg are sign extended.
fn decode_color<const S: usize>(data: &mut [u8], count: usize) {
    let max = ((1i32 << (S * 2)) - 1) as f32;

    for c in data[..count * S].as_chunks_mut::<S>().0 {
        // recover scale from alpha high bit
        let mut as_ = load(c, 3) as i32;
        as_ |= as_ >> 1;
        as_ |= as_ >> 2;
        as_ |= as_ >> 4;
        as_ |= as_ >> 8; // noop for 8-bit

        // convert to RGB in fixed point (co/cg are sign extended)
        let y = load(c, 0) as i32;
        let co = load_signed(c, 1);
        let cg = load_signed(c, 2);

        let r = y + co - cg;
        let g = y + cg;
        let b = y - co - cg;

        // expand alpha by one bit to match other components
        let mut a = load(c, 3) as i32;
        a = ((a << 1) & as_) | (a & 1);

        // compute scaling factor
        let ss = max / as_ as f32;

        // rounded float->int
        store(c, 0, (r as f32 * ss + 0.5) as i32);
        store(c, 1, (g as f32 * ss + 0.5) as i32);
        store(c, 2, (b as f32 * ss + 0.5) as i32);
        store(c, 3, (a as f32 * ss + 0.5) as i32);
    }
}

fn validate_buffer(buffer: &[u8], count: usize, stride: usize) -> Result<()> {
    ensure(buffer.len() >= count * stride, Error::BufferTooSmall("buffer"))
}

/// Decodes octahedral encoding of a unit vector with K-bit signed X/Y as an input; Z must store 1.0f.
/// Each component is stored as an 8-bit or 16-bit normalized integer; stride must be equal to 4 or 8. W is preserved as is.
pub fn decode_filter_oct(buffer: &mut [u8], count: usize, stride: usize) -> Result<()> {
    ensure(stride == 4 || stride == 8, Error::InvalidArgument("stride must be 4 or 8"))?;
    validate_buffer(buffer, count, stride)?;

    if stride == 4 {
        decode_oct::<4>(buffer, count);
    } else {
        decode_oct::<8>(buffer, count);
    }
    Ok(())
}

/// Decodes 3-component quaternion encoding with K-bit component encoding and a 2-bit component index indicating which component to reconstruct.
/// Each component is stored as a 16-bit integer; stride must be equal to 8.
pub fn decode_filter_quat(buffer: &mut [u8], count: usize, stride: usize) -> Result<()> {
    ensure(stride == 8, Error::InvalidArgument("stride must be 8"))?;
    validate_buffer(buffer, count, stride)?;

    decode_quat(buffer, count);
    Ok(())
}

/// Decodes exponential encoding of floating-point data with 8-bit exponent and 24-bit integer mantissa as 2^E*M.
/// Each 32-bit component is decoded in isolation; stride must be divisible by 4.
pub fn decode_filter_exp(buffer: &mut [u8], count: usize, stride: usize) -> Result<()> {
    ensure(
        stride > 0 && stride.is_multiple_of(4),
        Error::InvalidArgument("stride must be a positive multiple of 4"),
    )?;
    validate_buffer(buffer, count, stride)?;

    decode_exp(buffer, count * (stride / 4));
    Ok(())
}

/// Decodes RGBA colors from YCoCg (+A) color encoding where RGB is converted to YCoCg space with K-bit component encoding, and A is stored using K-1 bits.
/// Each component is stored as an 8-bit or 16-bit normalized integer; stride must be equal to 4 or 8.
pub fn decode_filter_color(buffer: &mut [u8], count: usize, stride: usize) -> Result<()> {
    ensure(stride == 4 || stride == 8, Error::InvalidArgument("stride must be 4 or 8"))?;
    validate_buffer(buffer, count, stride)?;

    if stride == 4 {
        decode_color::<4>(buffer, count);
    } else {
        decode_color::<8>(buffer, count);
    }
    Ok(())
}

fn validate_encode(destination: &[u8], count: usize, stride: usize, data: &[f32], data_stride: usize) -> Result<()> {
    validate_buffer(destination, count, stride)?;
    ensure(data.len() >= count * data_stride, Error::BufferTooSmall("data"))
}

/// Encodes unit vectors with K-bit (2 <= K <= 16) signed X/Y as an output.
/// Each component is stored as an 8-bit or 16-bit normalized integer; stride must be equal to 4 (K <= 8) or 8 (K <= 16).
/// Input data must contain 4 floats for every vector (count*4 total).
pub fn encode_filter_oct(destination: &mut [u8], count: usize, stride: usize, bits: i32, data: &[f32]) -> Result<()> {
    ensure(stride == 4 || stride == 8, Error::InvalidArgument("stride must be 4 or 8"))?;
    ensure((2..=16).contains(&bits), Error::InvalidArgument("bits must be in 2..=16"))?;
    ensure(stride == 8 || bits <= 8, Error::InvalidArgument("bits must be <= 8 for stride 4"))?;
    validate_encode(destination, count, stride, data, 4)?;

    let bytebits = (stride * 2) as i32;

    for (i, n) in data.as_chunks::<4>().0.iter().take(count).enumerate() {
        // octahedral encoding of a unit vector
        let (mut nx, mut ny, nz, nw) = (n[0], n[1], n[2], n[3]);
        let nl = nx.abs() + ny.abs() + nz.abs();
        let ns = if nl == 0.0 { 0.0 } else { 1.0 / nl };

        nx *= ns;
        ny *= ns;

        let u = if nz >= 0.0 {
            nx
        } else {
            (1.0 - ny.abs()) * if nx >= 0.0 { 1.0 } else { -1.0 }
        };
        let v = if nz >= 0.0 {
            ny
        } else {
            (1.0 - nx.abs()) * if ny >= 0.0 { 1.0 } else { -1.0 }
        };

        let encoded = [
            quantize_snorm(u, bits),
            quantize_snorm(v, bits),
            quantize_snorm(1.0, bits),
            quantize_snorm(nw, bytebits),
        ];

        for (k, &e) in encoded.iter().enumerate() {
            if stride == 4 {
                destination[i * 4 + k] = e as u8;
            } else {
                set16(destination, i * 4 + k, e as u16);
            }
        }
    }

    Ok(())
}

/// Encodes unit quaternions with K-bit (4 <= K <= 16) component encoding.
/// Each component is stored as a 16-bit integer; stride must be equal to 8.
/// Input data must contain 4 floats for every quaternion (count*4 total).
pub fn encode_filter_quat(destination: &mut [u8], count: usize, stride: usize, bits: i32, data: &[f32]) -> Result<()> {
    ensure(stride == 8, Error::InvalidArgument("stride must be 8"))?;
    ensure((4..=16).contains(&bits), Error::InvalidArgument("bits must be in 4..=16"))?;
    validate_encode(destination, count, stride, data, 4)?;

    let scaler = 2.0f32.sqrt();

    for (i, q) in data.as_chunks::<4>().0.iter().take(count).enumerate() {
        // establish maximum quaternion component
        let mut qc = 0;
        qc = if q[1].abs() > q[qc].abs() { 1 } else { qc };
        qc = if q[2].abs() > q[qc].abs() { 2 } else { qc };
        qc = if q[3].abs() > q[qc].abs() { 3 } else { qc };

        // we use double-cover properties to discard the sign
        let sign = if q[qc] < 0.0 { -1.0 } else { 1.0 };

        // note: we always encode a cyclical swizzle to be able to recover the order via rotation
        set16(destination, i * 4, quantize_snorm(q[(qc + 1) & 3] * scaler * sign, bits) as u16);
        set16(destination, i * 4 + 1, quantize_snorm(q[(qc + 2) & 3] * scaler * sign, bits) as u16);
        set16(destination, i * 4 + 2, quantize_snorm(q[(qc + 3) & 3] * scaler * sign, bits) as u16);
        set16(destination, i * 4 + 3, ((quantize_snorm(1.0, bits) & !3) | qc as i32) as u16);
    }

    Ok(())
}

// optimized variant of frexp
fn optlog2(v: f32) -> i32 {
    // +1 accounts for implicit 1. in mantissa; zero and denormalized numbers will end up clamped to min_exp by calling code
    // this is important for shared exponent modes, as a zero component should not inflate the shared exponent
    ((v.to_bits() >> 23) & 0xff) as i32 - 127 + 1
}

// optimized variant of ldexp
fn optexp2(e: i32) -> f32 {
    f32::from_bits((e.wrapping_add(127) as u32) << 23)
}

/// Encodes arbitrary (finite) floating-point data with 8-bit exponent and K-bit integer mantissa (1 <= K <= 24).
/// Exponent can be shared between all components of a given vector as defined by stride or all values of a given component;
/// stride must be divisible by 4 (and <= 256). Input data must contain stride/4 floats for every vector (count*stride/4 total).
/// Use [`encode_filter_exp_in_place`] when the destination is the input data itself.
pub fn encode_filter_exp(destination: &mut [u8], count: usize, stride: usize, bits: i32, data: &[f32], mode: EncodeExpMode) -> Result<()> {
    ensure(
        stride > 0 && stride.is_multiple_of(4) && stride <= 256,
        Error::InvalidArgument("stride must be a multiple of 4 in 4..=256"),
    )?;
    ensure((1..=24).contains(&bits), Error::InvalidArgument("bits must be in 1..=24"))?;
    validate_encode(destination, count, stride, data, stride / 4)?;

    let stride_float = stride / 4;

    let mut component_exp = [0i32; 64];

    const MIN_EXP: i32 = -100;

    // used to track maximum exponent for EncodeExpSharedComponent and zero exponent reuse for EncodeExpSeparate
    component_exp[..stride_float].fill(if mode == EncodeExpMode::Separate { 0 } else { MIN_EXP });

    if mode == EncodeExpMode::SharedComponent {
        for v in data.chunks_exact(stride_float).take(count) {
            // use maximum exponent to encode values; this guarantees that mantissa is [-1, 1]
            for (ce, &x) in component_exp.iter_mut().zip(v) {
                let e = optlog2(x);

                *ce = if *ce < e { e } else { *ce };
            }
        }
    }

    for (i, v) in data.chunks_exact(stride_float).take(count).enumerate() {
        let mut vector_exp = MIN_EXP;

        match mode {
            EncodeExpMode::SharedVector => {
                // use maximum exponent to encode values; this guarantees that mantissa is [-1, 1]
                for &x in v {
                    let e = optlog2(x);

                    vector_exp = if vector_exp < e { e } else { vector_exp };
                }
            }
            EncodeExpMode::Separate => {
                for (ce, &x) in component_exp.iter_mut().zip(v) {
                    let e = optlog2(x);

                    // zero values inherit the last encoded exponent to keep the encoded data more compressible
                    if x != 0.0 {
                        *ce = if MIN_EXP < e { e } else { MIN_EXP };
                    }
                }
            }
            EncodeExpMode::Clamped => {
                for (ce, &x) in component_exp.iter_mut().zip(v) {
                    let e = optlog2(x);

                    *ce = if 0 < e { e } else { 0 };
                }
            }
            EncodeExpMode::SharedComponent => {
                // the code below assumes component_exp is initialized outside of the loop
            }
        }

        for (j, &x) in v.iter().enumerate() {
            let mut exp = if mode == EncodeExpMode::SharedVector {
                vector_exp
            } else {
                component_exp[j]
            };

            // note that we additionally scale the mantissa to make it a K-bit signed integer (K-1 bits for magnitude)
            exp -= bits - 1;

            // compute renormalized rounded mantissa for each component
            let mmask = (1i32 << 24) - 1;
            let mmax = mmask >> 1;

            let mut m = (x * optexp2(-exp) + if x >= 0.0 { 0.5 } else { -0.5 }) as i32;

            // clamp mantissa so that rounding away from zero doesn't overflow
            m = if m > mmax { mmax } else { m };

            set32(destination, i * stride_float + j, (m & mmask) as u32 | ((exp as u32) << 24));
        }
    }

    Ok(())
}

/// In-place variant of [`encode_filter_exp`] (C: `destination == data`); encoded 32-bit words are stored as `f32` bit patterns.
pub fn encode_filter_exp_in_place(data: &mut [f32], count: usize, stride: usize, bits: i32, mode: EncodeExpMode) -> Result<()> {
    let input = data.to_vec();
    let mut encoded = vec![0u8; data.len() * 4];

    encode_filter_exp(&mut encoded, count, stride, bits, &input, mode)?;

    for (d, e) in data.iter_mut().zip(encoded.as_chunks::<4>().0).take(count * (stride / 4)) {
        *d = f32::from_bits(u32::from_le_bytes(*e));
    }

    Ok(())
}

/// Encodes RGBA color data by converting RGB to YCoCg color space with K-bit (2 <= K <= 16) component encoding; A is stored using K-1 bits.
/// Each component is stored as an 8-bit or 16-bit integer; stride must be equal to 4 (K <= 8) or 8 (K <= 16).
/// Input data must contain 4 floats for every color (count*4 total).
pub fn encode_filter_color(destination: &mut [u8], count: usize, stride: usize, bits: i32, data: &[f32]) -> Result<()> {
    ensure(stride == 4 || stride == 8, Error::InvalidArgument("stride must be 4 or 8"))?;
    ensure((2..=16).contains(&bits), Error::InvalidArgument("bits must be in 2..=16"))?;
    ensure(stride == 8 || bits <= 8, Error::InvalidArgument("bits must be <= 8 for stride 4"))?;
    validate_encode(destination, count, stride, data, 4)?;

    for (i, c) in data.as_chunks::<4>().0.iter().take(count).enumerate() {
        let fr = quantize_unorm(c[0], bits);
        let fg = quantize_unorm(c[1], bits);
        let fb = quantize_unorm(c[2], bits);

        // YCoCg-R encoding with truncated Co/Cg ensures that decoding can be done using integers
        let fco = (fr - fb) / 2;
        let tmp = fb + fco;
        let fcg = (fg - tmp) / 2;
        let fy = tmp + fcg;

        // alpha: K-1-bit encoding with high bit set to 1
        let fa = (quantize_unorm(c[3], bits) >> 1) | (1 << (bits - 1));

        for (k, e) in [fy, fco, fcg, fa].into_iter().enumerate() {
            if stride == 4 {
                destination[i * 4 + k] = e as u8;
            } else {
                set16(destination, i * 4 + k, e as u16);
            }
        }
    }

    Ok(())
}
