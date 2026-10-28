// This work is based on:
// Francine Evans, Steven Skiena and Amitabh Varshney. Optimizing Triangle Strips for Fast Rendering. 1996
// Hugues Hoppe. Optimization of Mesh Locality for Transparent Vertex Caching. 1999

use crate::error::{Error, Result, ensure};
use crate::index::{self, IndexType};
use crate::validate;

fn find_strip_first(buffer: &[[u32; 3]], valence: &[u8], hint: &[u32; 3]) -> usize {
    let mut index = 0;
    let mut iv = !0u32;

    for (i, tri) in buffer.iter().enumerate() {
        if tri == hint {
            return i;
        }

        let (va, vb, vc) = (valence[tri[0] as usize], valence[tri[1] as usize], valence[tri[2] as usize]);
        let v = u32::from(if va < vb && va < vc {
            va
        } else if vb < vc {
            vb
        } else {
            vc
        });

        if v < iv {
            index = i;
            iv = v;
        }
    }

    index
}

fn find_strip_next(buffer: &[[u32; 3]], e0: u32, e1: u32) -> i32 {
    let e = (u64::from(e0) << 32) | u64::from(e1);

    for (i, &[a, b, c]) in buffer.iter().enumerate() {
        let ab = (u64::from(a) << 32) | u64::from(b);
        let bc = (u64::from(b) << 32) | u64::from(c);
        let ca = (u64::from(c) << 32) | u64::from(a);

        if e == ab {
            return ((i as i32) << 2) | 2;
        } else if e == bc {
            return (i as i32) << 2;
        } else if e == ca {
            return ((i as i32) << 2) | 1;
        }
    }

    -1
}

fn count_shared_vertices(tri: &[u32; 3], valence: &[u8]) -> u32 {
    tri.iter().map(|&v| u32::from(valence[v as usize] > 1)).sum()
}

fn emit(destination: &mut [u32], strip_size: &mut usize, v: u32) {
    destination[*strip_size] = v;
    *strip_size += 1;
}

fn take_triangle(buffer: &mut [[u32; 3]], buffer_size: &mut usize, valence: &mut [u8], i: usize) -> [u32; 3] {
    let tri = buffer[i];

    // ordered removal from the buffer
    buffer.copy_within(i + 1..*buffer_size, i);
    *buffer_size -= 1;

    // update vertex valences for strip start/next heuristic
    for v in tri {
        valence[v as usize] = valence[v as usize].wrapping_sub(1);
    }

    tri
}

/// Converts a previously vertex cache optimized triangle list to triangle strip, stitching strips using restart index or degenerate triangles.
/// Returns the number of indices in the resulting strip; `destination` must hold [`stripify_bound`] entries.
/// `restart_index` should be 0 to use degenerate triangles, or `!0` to use restart indices.
pub fn stripify<T: IndexType>(
    destination: &mut [T],
    indices: &[T],
    index_count: usize,
    vertex_count: usize,
    restart_index: T,
) -> Result<usize> {
    let indices = index::input(indices);
    index::output(destination, |destination| {
        stripify_u32(destination, &indices, index_count, vertex_count, restart_index.to_u32())
    })
}

fn stripify_u32(destination: &mut [u32], indices: &[u32], index_count: usize, vertex_count: usize, restart_index: u32) -> Result<usize> {
    let bound = stripify_bound(index_count)?;
    let indices = validate::indices(indices, index_count, vertex_count)?;
    let destination = validate::output(destination, bound)?;

    const BUFFER_CAPACITY: usize = 12;

    let mut buffer = [[0u32; 3]; BUFFER_CAPACITY];
    let mut buffer_size = 0usize;

    let mut index_offset = 0usize;

    let mut strip = [0u32; 2];
    let mut parity = 0u32;

    let mut strip_size = 0usize;

    // compute vertex valence; this is used to prioritize starting and continuation triangles for strips
    // note: we use 8-bit counters for performance; for outlier vertices the valence is incorrect but that just affects the heuristics
    let mut valence = vec![0u8; vertex_count];

    for &index in indices {
        valence[index as usize] = valence[index as usize].wrapping_add(1);
    }

    let mut next = -1i32;
    let mut skipped = [0u32; 3];

    while buffer_size > 0 || index_offset < index_count {
        // fill triangle buffer
        while buffer_size < BUFFER_CAPACITY && index_offset < index_count {
            buffer[buffer_size].copy_from_slice(&indices[index_offset..index_offset + 3]);

            buffer_size += 1;
            index_offset += 3;
        }

        if next >= 0 {
            let i = (next >> 2) as usize;
            let v = buffer[i][(next & 3) as usize];

            take_triangle(&mut buffer, &mut buffer_size, &mut valence, i);
            let live = &buffer[..buffer_size];

            // find next triangle (note that edge order flips on every iteration)
            // in some cases we need to perform a swap to pick a different outgoing triangle edge
            // for [a b c], the default strip edge is [b c], but we might want to use [a c]
            // note: we skip the search if the older strip vertex has no live triangles left for performance
            let mut cont = if valence[strip[1] as usize] != 0 {
                if parity != 0 {
                    find_strip_next(live, strip[1], v)
                } else {
                    find_strip_next(live, v, strip[1])
                }
            } else {
                -1
            };
            let swap = if valence[strip[0] as usize] != 0 {
                if parity != 0 {
                    find_strip_next(live, v, strip[0])
                } else {
                    find_strip_next(live, strip[0], v)
                }
            } else {
                -1
            };

            // when both edges have a neighbor, pick triangle with fewer neighbors to avoid future dead-ends
            // this is locally sub-optimal (we spend one extra index to swap) but creates a smaller output in the end
            if cont >= 0 && swap >= 0 {
                let use_swap = count_shared_vertices(&live[(swap >> 2) as usize], &valence)
                    < count_shared_vertices(&live[(cont >> 2) as usize], &valence);

                // remember the triangle we didn't pick for a future strip restart
                skipped = live[((if use_swap { cont } else { swap }) >> 2) as usize];
                cont = if use_swap { -1 } else { cont };
            }

            if cont < 0 && swap >= 0 {
                // [a b c] => [a b a c]
                emit(destination, &mut strip_size, strip[0]);
                emit(destination, &mut strip_size, v);

                // next strip has same winding
                // ? a b => b a v
                strip[1] = v;

                next = swap;
            } else {
                // emit the next vertex in the strip
                emit(destination, &mut strip_size, v);

                // next strip has flipped winding
                strip[0] = strip[1];
                strip[1] = v;
                parity ^= 1;

                next = cont;
            }
        } else {
            // if we didn't find anything, we need to find the next new triangle
            // we use a heuristic to maximize the strip length
            let i = find_strip_first(&buffer[..buffer_size], &valence, &skipped);
            let [mut a, mut b, mut c] = take_triangle(&mut buffer, &mut buffer_size, &mut valence, i);
            let live = &buffer[..buffer_size];

            // we need to pre-rotate the triangle so that we will find a match in the existing buffer on the next iteration
            let ea = find_strip_next(live, c, b);
            let eb = find_strip_next(live, a, c);
            let ec = find_strip_next(live, b, a);

            // in some cases we can have several matching edges; since we can pick any edge, we pick the one with the smallest
            // triangle index in the buffer. this reduces the effect of stripification on ACMR and additionally - for unclear
            // reasons - slightly improves the stripification efficiency
            let mut mine = i32::MAX;
            mine = if ea >= 0 && mine > ea { ea } else { mine };
            mine = if eb >= 0 && mine > eb { eb } else { mine };
            mine = if ec >= 0 && mine > ec { ec } else { mine };

            if ea == mine {
                // keep abc
                next = ea;
            } else if eb == mine {
                // abc -> bca
                (a, b, c) = (b, c, a);

                next = eb;
            } else if ec == mine {
                // abc -> cab
                (a, b, c) = (c, a, b);

                next = ec;
            }

            if restart_index != 0 {
                if strip_size != 0 {
                    emit(destination, &mut strip_size, restart_index);
                }

                emit(destination, &mut strip_size, a);
                emit(destination, &mut strip_size, b);
                emit(destination, &mut strip_size, c);

                // new strip always starts with the same edge winding
                strip = [b, c];
                parity = 1;
            } else {
                if strip_size != 0 {
                    // connect last strip using degenerate triangles
                    emit(destination, &mut strip_size, strip[1]);
                    emit(destination, &mut strip_size, a);
                }

                // note that we may need to flip the emitted triangle based on parity
                // we always end up with outgoing edge "cb" in the end
                let e0 = if parity != 0 { c } else { b };
                let e1 = if parity != 0 { b } else { c };

                emit(destination, &mut strip_size, a);
                emit(destination, &mut strip_size, e0);
                emit(destination, &mut strip_size, e1);

                strip = [e0, e1];
                parity ^= 1;
            }
        }
    }

    Ok(strip_size)
}

/// Worst case number of indices produced by [`stripify`].
pub fn stripify_bound(index_count: usize) -> Result<usize> {
    validate::triangles(index_count)?;

    // worst case without restarts is 2 degenerate indices and 3 indices per triangle
    // worst case with restarts is 1 restart index and 3 indices per triangle
    Ok((index_count / 3) * 5)
}

/// Converts a triangle strip to a triangle list; returns the number of indices in the resulting list.
/// `destination` must hold [`unstripify_bound`] entries.
pub fn unstripify<T: IndexType>(destination: &mut [T], indices: &[T], index_count: usize, restart_index: T) -> Result<usize> {
    let indices = index::input(indices);
    index::output(destination, |destination| {
        unstripify_u32(destination, &indices, index_count, restart_index.to_u32())
    })
}

fn unstripify_u32(destination: &mut [u32], indices: &[u32], index_count: usize, restart_index: u32) -> Result<usize> {
    let indices = indices.get(..index_count).ok_or(Error::BufferTooSmall("indices"))?;
    let destination = validate::output(destination, if index_count < 3 { 0 } else { (index_count - 2) * 3 })?;

    let mut offset = 0;
    let mut start = 0;

    for i in 0..index_count {
        if restart_index != 0 && indices[i] == restart_index {
            start = i + 1;
        } else if i - start >= 2 {
            let (mut a, mut b, c) = (indices[i - 2], indices[i - 1], indices[i]);

            // flip winding for odd triangles
            if (i - start) & 1 != 0 {
                (a, b) = (b, a);
            }

            // although we use restart indices, strip swaps still produce degenerate triangles, so skip them
            if a != b && a != c && b != c {
                destination[offset..offset + 3].copy_from_slice(&[a, b, c]);
                offset += 3;
            }
        }
    }

    Ok(offset)
}

/// Worst case number of indices produced by [`unstripify`].
pub fn unstripify_bound(index_count: usize) -> Result<usize> {
    ensure(
        index_count == 0 || index_count >= 3,
        Error::InvalidArgument("index_count must be 0 or >= 3"),
    )?;

    Ok(if index_count == 0 { 0 } else { (index_count - 2) * 3 })
}
