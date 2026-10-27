//! Open addressing hash table helpers shared by several algorithms (C: `hashBuckets`/`hashLookup`).

pub(crate) trait Hasher<T> {
    fn hash(&self, key: &T) -> usize;
    fn equal(&mut self, lhs: &T, rhs: &T) -> bool;
}

pub(crate) fn hash_buckets(count: usize) -> usize {
    let mut buckets = 1;
    while buckets < count + count / 4 {
        buckets *= 2;
    }

    buckets
}

/// Returns the slot holding `key` or the empty slot where it should be inserted.
/// `table.len()` must be a power of two larger than the number of stored keys, so a free slot always exists.
pub(crate) fn hash_lookup<T: PartialEq, H: Hasher<T>>(table: &[T], hash: &mut H, key: &T, empty: &T) -> usize {
    debug_assert!(table.len().is_power_of_two());

    let hashmod = table.len() - 1;
    let mut bucket = hash.hash(key) & hashmod;

    for probe in 0..=hashmod {
        let item = &table[bucket];

        if item == empty || hash.equal(item, key) {
            return bucket;
        }

        // hash collision, quadratic probing
        bucket = (bucket + probe + 1) & hashmod;
    }

    // unreachable: tables are sized via hash_buckets so they are never full
    bucket
}

/// MurmurHash2 update over a byte slice.
pub(crate) fn hash_update(mut h: u32, key: &[u8]) -> u32 {
    const M: u32 = 0x5bd1e995;
    const R: u32 = 24;

    let (words, tail) = key.as_chunks::<4>();

    for w in words {
        let mut k = u32::from_le_bytes(*w);

        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);

        h = h.wrapping_mul(M);
        h ^= k;
    }

    let len = tail.len();
    if len > 0 {
        h ^= u32::from(tail[0]) | (u32::from(tail[len / 2]) << 8) | (u32::from(tail[len - 1]) << 16);
        h = h.wrapping_mul(M);
        h ^= h >> R;
    }

    h
}
