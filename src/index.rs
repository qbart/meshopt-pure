//! Index element types (C++ template wrappers accept any integer index type `T`).

mod sealed {
    pub trait Sealed {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
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
