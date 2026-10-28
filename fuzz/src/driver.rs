//! Pure Rust mutation fuzzer (no coverage feedback): mutates seed inputs with libFuzzer-style byte mutations.

use meshoptimizer::{encode_index_buffer, encode_index_sequence, encode_meshlet, encode_vertex_buffer_level};

/// splitmix64
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// 0..n (n > 0)
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }

    pub fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next_u64() as u8).collect()
    }
}

const INTERESTING: [u8; 8] = [0, 1, 0x7f, 0x80, 0xfe, 0xff, 0x10, 0xe0];

/// Applies 1..=8 random mutations; `other` is used for splicing.
pub fn mutate(rng: &mut Rng, input: &[u8], other: &[u8], max_len: usize) -> Vec<u8> {
    let mut data = input.to_vec();

    for _ in 0..1 + rng.below(8) {
        match rng.below(10) {
            // flip a bit
            0 if !data.is_empty() => {
                let i = rng.below(data.len());
                data[i] ^= 1 << rng.below(8);
            }
            // random byte
            1 if !data.is_empty() => {
                let i = rng.below(data.len());
                data[i] = rng.next_u64() as u8;
            }
            // interesting byte
            2 if !data.is_empty() => {
                let i = rng.below(data.len());
                data[i] = INTERESTING[rng.below(INTERESTING.len())];
            }
            // add/subtract a small value
            3 if !data.is_empty() => {
                let i = rng.below(data.len());
                data[i] = data[i].wrapping_add(rng.below(33) as u8).wrapping_sub(16);
            }
            // insert random bytes
            4 => {
                let i = rng.below(data.len() + 1);
                let n = 1 + rng.below(8);
                let bytes = rng.bytes(n);
                data.splice(i..i, bytes);
            }
            // erase a range
            5 if !data.is_empty() => {
                let i = rng.below(data.len());
                let n = 1 + rng.below((data.len() - i).min(16));
                data.drain(i..i + n);
            }
            // duplicate a range
            6 if !data.is_empty() => {
                let i = rng.below(data.len());
                let n = 1 + rng.below((data.len() - i).min(32));
                let chunk = data[i..i + n].to_vec();
                let j = rng.below(data.len() + 1);
                data.splice(j..j, chunk);
            }
            // truncate
            7 if !data.is_empty() => {
                data.truncate(rng.below(data.len()));
            }
            // splice a range of the other input
            8 if !other.is_empty() => {
                let i = rng.below(other.len());
                let n = 1 + rng.below((other.len() - i).min(64));
                let j = rng.below(data.len() + 1);
                data.splice(j..j, other[i..i + n].iter().copied());
            }
            // overwrite with a range of the other input
            _ if !other.is_empty() && !data.is_empty() => {
                let i = rng.below(other.len());
                let j = rng.below(data.len());
                let n = 1 + rng.below((other.len() - i).min(data.len() - j).min(64));
                data[j..j + n].copy_from_slice(&other[i..i + n]);
            }
            _ => data.push(rng.next_u64() as u8),
        }
    }

    data.truncate(max_len);
    data
}

fn random_triangles(rng: &mut Rng, count: usize, vertices: usize) -> Vec<u32> {
    (0..count).map(|_| rng.below(vertices) as u32).collect()
}

/// Seed corpus for a target: valid encodings for `codec` (so mutations reach past the headers), structured and random
/// index bytes for `simplify`/`cluster`.
pub fn seeds(name: &str, rng: &mut Rng) -> Vec<Vec<u8>> {
    let mut seeds = vec![Vec::new(), vec![0], vec![0xff; 3]];

    if name == "codec" {
        // the decoders read 66 elements
        for vertices in [8, 66, 1000] {
            let mut indices = random_triangles(rng, 66, vertices);
            if vertices == 8 {
                indices = (0..66).map(|i| (i / 3 + i % 3) as u32).collect();
            }

            let mut buf = vec![0u8; 4096];
            let size = buf.len();
            if let Ok(n) = encode_index_buffer(&mut buf, size, &indices, 66) {
                seeds.push(buf[..n].to_vec());
            }
            if let Ok(n) = encode_index_sequence(&mut buf, size, &indices, 66) {
                seeds.push(buf[..n].to_vec());
            }
        }

        for stride in [4, 16, 24, 32] {
            for (level, version) in [(0, 0), (2, 1), (3, 1)] {
                // smooth data compresses into all block modes
                let vertices: Vec<u8> = (0..66 * stride)
                    .map(|i| ((i / stride) as u8).wrapping_mul((i % stride) as u8))
                    .collect();
                let mut buf = vec![0u8; 66 * stride * 2 + 4096];
                let size = buf.len();
                if let Ok(n) = encode_vertex_buffer_level(&mut buf, size, &vertices, 66, stride, level, version) {
                    seeds.push(buf[..n].to_vec());
                }
            }
        }

        // meshlets: the first two bytes are vertex_count - 1 and triangle_count - 1
        for (vertex_count, triangle_count) in [(4, 2), (64, 124), (256, 256)] {
            let vertices: Vec<u32> = (0..vertex_count).map(|i| (i * 3 + rng.below(4)) as u32).collect();
            let triangles: Vec<u8> = (0..triangle_count * 3).map(|_| rng.below(vertex_count) as u8).collect();
            let mut buf = vec![0u8; 8192];
            let size = buf.len();
            if let Ok(n) = encode_meshlet(&mut buf, size, &vertices, vertex_count, &triangles, triangle_count) {
                let mut seed = vec![(vertex_count - 1) as u8, (triangle_count - 1) as u8];
                seed.extend_from_slice(&buf[..n]);
                seeds.push(seed);
            }
        }

        // roundtrip inputs
        seeds.push(rng.bytes(64));
        seeds.push(rng.bytes(1024));
    } else {
        // a grid over the 100 vertices (index bytes are taken % 100)
        let mut grid = vec![7u8];
        for y in 0..9u8 {
            for x in 0..9u8 {
                let a = y * 10 + x;
                grid.extend_from_slice(&[a, a + 1, a + 11, a, a + 11, a + 10]);
            }
        }
        seeds.push(grid);

        for len in [4, 31, 300, 1000] {
            seeds.push(rng.bytes(len));
        }
    }

    seeds
}

/// Runs `iterations` mutated inputs through `target`; returns the first input that panicked, if any.
pub fn run(name: &str, target: fn(&[u8]), iterations: u64, seed: u64, max_len: usize) -> Option<Vec<u8>> {
    let mut rng = Rng::new(seed);
    let seeds = seeds(name, &mut rng);

    for i in 0..iterations {
        let input = if (i as usize) < seeds.len() {
            seeds[i as usize].clone()
        } else {
            let a = &seeds[rng.below(seeds.len())];
            let b = &seeds[rng.below(seeds.len())];
            mutate(&mut rng, a, b, max_len)
        };

        if std::panic::catch_unwind(|| target(&input)).is_err() {
            return Some(input);
        }
    }

    None
}
