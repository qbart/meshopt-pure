//! glibc `srand`/`rand` (TYPE_3 additive feedback generator), so the harnesses build the same vertex buffers as C++.

pub struct CRand {
    // r[k] for the last 34 k, stored at k % 34
    r: [i32; 34],
    k: usize,
}

impl CRand {
    pub fn new(seed: u32) -> Self {
        let mut r = [0i32; 34];
        r[0] = if seed == 0 { 1 } else { seed as i32 };

        for i in 1..31 {
            // r[i] = (16807 * r[i - 1]) % 2147483647 without overflow (Schrage's method)
            let hi = r[i - 1] / 127773;
            let lo = r[i - 1] % 127773;
            let word = 16807i32.wrapping_mul(lo).wrapping_sub(2836i32.wrapping_mul(hi));
            r[i] = if word < 0 { word.wrapping_add(2147483647) } else { word };
        }

        for i in 31..34 {
            r[i] = r[i - 31];
        }

        let mut rng = CRand { r, k: 34 };

        // glibc discards the first 310 outputs
        for _ in 0..310 {
            rng.next_raw();
        }

        rng
    }

    fn next_raw(&mut self) -> i32 {
        // r[k] = r[k - 31] + r[k - 3]
        let v = self.r[(self.k + 3) % 34].wrapping_add(self.r[(self.k + 31) % 34]);
        self.r[self.k % 34] = v;
        self.k += 1;
        v
    }

    /// `rand()`: 0..=RAND_MAX (2^31 - 1)
    pub fn rand(&mut self) -> i32 {
        ((self.next_raw() as u32) >> 1) as i32
    }
}
