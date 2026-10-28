//! Index element types (C++ template wrappers accept any integer index type `T`).

use std::borrow::Cow;

mod sealed {
    pub trait Sealed: Sized {
        /// Zero-copy view for `u32` buffers (`meshopt_IndexAdapter<T, true>`).
        fn as_u32(s: &[Self]) -> Option<&[u32]>;
        fn as_u32_mut(s: &mut [Self]) -> Option<&mut [u32]>;
    }

    impl Sealed for u16 {
        fn as_u32(_: &[Self]) -> Option<&[u32]> {
            None
        }

        fn as_u32_mut(_: &mut [Self]) -> Option<&mut [u32]> {
            None
        }
    }

    impl Sealed for u32 {
        fn as_u32(s: &[Self]) -> Option<&[u32]> {
            Some(s)
        }

        fn as_u32_mut(s: &mut [Self]) -> Option<&mut [u32]> {
            Some(s)
        }
    }
}

/// Integer type usable as an index buffer element (`u16` or `u32`).
pub trait IndexType: Copy + Default + sealed::Sealed {
    /// Size of the element in bytes (C `index_size`).
    const SIZE: usize;

    /// Truncating conversion, like the C casts.
    fn from_u32(v: u32) -> Self;
    fn to_u32(self) -> u32;
}

impl IndexType for u16 {
    const SIZE: usize = 2;

    fn from_u32(v: u32) -> Self {
        v as u16
    }

    fn to_u32(self) -> u32 {
        u32::from(self)
    }
}

impl IndexType for u32 {
    const SIZE: usize = 4;

    fn from_u32(v: u32) -> Self {
        v
    }

    fn to_u32(self) -> u32 {
        self
    }
}

/// Input side of `meshopt_IndexAdapter`: borrows `u32` buffers, widens other index types into a temporary copy.
pub(crate) fn input<T: IndexType>(indices: &[T]) -> Cow<'_, [u32]> {
    match T::as_u32(indices) {
        Some(data) => Cow::Borrowed(data),
        None => Cow::Owned(indices.iter().map(|&v| v.to_u32()).collect()),
    }
}

/// Optional variant of [`input`] (C: `indices ? in.data : NULL`).
pub(crate) fn input_opt<T: IndexType>(indices: Option<&[T]>) -> Option<Cow<'_, [u32]>> {
    indices.map(input)
}

/// Output side of `meshopt_IndexAdapter`: runs `f` on a `u32` view of `destination` and narrows the result back
/// with truncating casts (C: `result[i] = T(data[i])`). The temporary starts as a copy of `destination`, so
/// entries `f` does not write keep their value, same as when `T` is `u32`.
pub(crate) fn output<T: IndexType, R>(destination: &mut [T], f: impl FnOnce(&mut [u32]) -> R) -> R {
    match T::as_u32_mut(destination) {
        Some(data) => f(data),
        None => {
            let mut data: Vec<u32> = destination.iter().map(|&v| v.to_u32()).collect();
            let result = f(&mut data);

            for (d, &v) in destination.iter_mut().zip(&data) {
                *d = T::from_u32(v);
            }

            result
        }
    }
}
