//! C++ semantics the gltfpack output depends on: `std::min`/`std::max` on floats, x86-64 float to integer
//! conversions, and libstdc++ `std::sort` (introsort, not stable: the order of equivalent elements is observable).

/// `std::min(a, b)`: `(b < a) ? b : a`.
pub fn fmin(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}

/// `std::max(a, b)`: `(a < b) ? b : a`.
pub fn fmax(a: f32, b: f32) -> f32 {
    if a < b { b } else { a }
}

/// `int(f)` (x86 `cvttss2si`): NaN and out of range values produce `INT_MIN`.
pub fn f2i32(f: f32) -> i32 {
    if f.is_nan() || !(-2_147_483_648.0..2_147_483_648.0).contains(&f) {
        i32::MIN
    } else {
        f as i32
    }
}

/// 64-bit `cvttss2si`: NaN and out of range values produce `INT64_MIN`.
pub fn f2i64(f: f32) -> i64 {
    if f.is_nan() || !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&f) {
        i64::MIN
    } else {
        f as i64
    }
}

/// `unsigned(f)` / `uint32_t(f)`: GCC converts through a 64-bit signed conversion and keeps the low bits.
pub fn f2u32(f: f32) -> u32 {
    f2i64(f) as u32
}

/// `uint16_t(f)`: converted through `int`.
pub fn f2u16(f: f32) -> u16 {
    f2i32(f) as u16
}

/// `uint8_t(f)`: converted through `int`.
pub fn f2u8(f: f32) -> u8 {
    f2i32(f) as u8
}

/// `size_t(d)` for a double (GCC x86-64 sequence: values >= 2^63 are offset, NaN gives 2^63); truncated on 32-bit targets.
pub fn d2usize(d: f64) -> usize {
    const TWO63: f64 = 9_223_372_036_854_775_808.0;
    const HIGH: u64 = 1 << 63;
    let r = if d.is_nan() {
        HIGH
    } else if d < TWO63 {
        if d >= -TWO63 { d as i64 as u64 } else { HIGH }
    } else {
        let v = d - TWO63;
        if v < TWO63 { (v as i64 as u64) ^ HIGH } else { HIGH }
    };
    r as usize
}

const S_THRESHOLD: usize = 16;

fn lg(n: usize) -> usize {
    (usize::BITS - 1 - n.leading_zeros()) as usize
}

/// libstdc++ `std::sort(v.begin(), v.end(), comp)` where `comp(a, b)` is the strict weak "less than" ordering.
pub fn sort_by<T: Copy>(v: &mut [T], comp: impl Fn(&T, &T) -> bool) {
    let n = v.len();
    if n > 1 {
        introsort_loop(v, 0, n, 2 * lg(n), &comp);
        final_insertion_sort(v, 0, n, &comp);
    }
}

fn introsort_loop<T: Copy>(v: &mut [T], first: usize, mut last: usize, mut depth_limit: usize, comp: &impl Fn(&T, &T) -> bool) {
    while last - first > S_THRESHOLD {
        if depth_limit == 0 {
            heap_sort(v, first, last, comp);
            return;
        }
        depth_limit -= 1;
        let cut = unguarded_partition_pivot(v, first, last, comp);
        introsort_loop(v, cut, last, depth_limit, comp);
        last = cut;
    }
}

fn move_median_to_first<T: Copy>(v: &mut [T], result: usize, a: usize, b: usize, c: usize, comp: &impl Fn(&T, &T) -> bool) {
    if comp(&v[a], &v[b]) {
        if comp(&v[b], &v[c]) {
            v.swap(result, b);
        } else if comp(&v[a], &v[c]) {
            v.swap(result, c);
        } else {
            v.swap(result, a);
        }
    } else if comp(&v[a], &v[c]) {
        v.swap(result, a);
    } else if comp(&v[b], &v[c]) {
        v.swap(result, c);
    } else {
        v.swap(result, b);
    }
}

fn unguarded_partition<T: Copy>(v: &mut [T], mut first: usize, mut last: usize, pivot: usize, comp: &impl Fn(&T, &T) -> bool) -> usize {
    // the scans are bounded by sentinels for consistent comparators; the slice bounds only protect against
    // inconsistent ones (undefined behavior in C++)
    let end = last;
    loop {
        while first < end && comp(&v[first], &v[pivot]) {
            first += 1;
        }
        last -= 1;
        while last > pivot && comp(&v[pivot], &v[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}

fn unguarded_partition_pivot<T: Copy>(v: &mut [T], first: usize, last: usize, comp: &impl Fn(&T, &T) -> bool) -> usize {
    let mid = first + (last - first) / 2;
    move_median_to_first(v, first, first + 1, mid, last - 1, comp);
    unguarded_partition(v, first + 1, last, first, comp)
}

fn unguarded_linear_insert<T: Copy>(v: &mut [T], mut last: usize, lower: usize, comp: &impl Fn(&T, &T) -> bool) {
    let val = v[last];
    while last > lower && comp(&val, &v[last - 1]) {
        v[last] = v[last - 1];
        last -= 1;
    }
    v[last] = val;
}

fn insertion_sort<T: Copy>(v: &mut [T], first: usize, last: usize, comp: &impl Fn(&T, &T) -> bool) {
    if first == last {
        return;
    }
    for i in first + 1..last {
        if comp(&v[i], &v[first]) {
            let val = v[i];
            v.copy_within(first..i, first + 1);
            v[first] = val;
        } else {
            unguarded_linear_insert(v, i, first, comp);
        }
    }
}

fn final_insertion_sort<T: Copy>(v: &mut [T], first: usize, last: usize, comp: &impl Fn(&T, &T) -> bool) {
    if last - first > S_THRESHOLD {
        insertion_sort(v, first, first + S_THRESHOLD, comp);
        for i in first + S_THRESHOLD..last {
            unguarded_linear_insert(v, i, first, comp);
        }
    } else {
        insertion_sort(v, first, last, comp);
    }
}

fn push_heap<T: Copy>(v: &mut [T], first: usize, mut hole: usize, top: usize, value: T, comp: &impl Fn(&T, &T) -> bool) {
    let mut parent = hole.wrapping_sub(1) / 2;
    while hole > top && comp(&v[first + parent], &value) {
        v[first + hole] = v[first + parent];
        hole = parent;
        parent = hole.wrapping_sub(1) / 2;
    }
    v[first + hole] = value;
}

fn adjust_heap<T: Copy>(v: &mut [T], first: usize, mut hole: usize, len: usize, value: T, comp: &impl Fn(&T, &T) -> bool) {
    let top = hole;
    let mut second = hole;
    while second < (len - 1) / 2 {
        second = 2 * (second + 1);
        if comp(&v[first + second], &v[first + second - 1]) {
            second -= 1;
        }
        v[first + hole] = v[first + second];
        hole = second;
    }
    if len & 1 == 0 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        v[first + hole] = v[first + second - 1];
        hole = second - 1;
    }
    push_heap(v, first, hole, top, value, comp);
}

/// `std::__partial_sort(first, last, last)`: make_heap + sort_heap.
fn heap_sort<T: Copy>(v: &mut [T], first: usize, mut last: usize, comp: &impl Fn(&T, &T) -> bool) {
    let len = last - first;
    if len >= 2 {
        let mut parent = (len - 2) / 2;
        loop {
            let value = v[first + parent];
            adjust_heap(v, first, parent, len, value, comp);
            if parent == 0 {
                break;
            }
            parent -= 1;
        }
    }

    while last - first > 1 {
        last -= 1;
        let value = v[last];
        v[last] = v[first];
        adjust_heap(v, first, 0, last - first, value, comp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sorts() {
        let mut seed = 1u32;
        for n in 0..200 {
            let mut v: Vec<(u32, u32)> = (0..n)
                .map(|i| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 28, i)
                })
                .collect();
            sort_by(&mut v, |a, b| a.0 > b.0);
            assert!(v.windows(2).all(|w| w[0].0 >= w[1].0));
        }
    }

    #[test]
    fn conversions() {
        assert_eq!(f2i32(f32::NAN), i32::MIN);
        assert_eq!(f2i32(3e9), i32::MIN);
        assert_eq!(f2i32(-1.5), -1);
        assert_eq!(f2u32(-1.0), u32::MAX);
        assert_eq!(f2u32(3e9), 3_000_000_000);
        assert_eq!(f2u16(70000.0), 70000u32 as u16);
        assert!(fmin(f32::NAN, 1.0).is_nan());
        assert_eq!(fmin(1.0, f32::NAN), 1.0);
        assert!(fmax(f32::NAN, 1.0).is_nan());
        assert_eq!(fmax(1.0, f32::NAN), 1.0);
        assert_eq!(d2usize(1e19), 10_000_000_000_000_000_000u64 as usize);
    }
}
