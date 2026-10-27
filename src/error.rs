use core::fmt;

/// Library error; every C precondition (`assert`) and decoder status code maps here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// An argument violates a documented precondition.
    InvalidArgument(&'static str),
    /// An input or output slice is shorter than the counts/strides require.
    BufferTooSmall(&'static str),
    /// An index references a vertex outside of `vertex_count`.
    IndexOutOfRange,
    /// Decoder failure with the original negative status code.
    Decode(i32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidArgument(what) => write!(f, "invalid argument: {what}"),
            Error::BufferTooSmall(what) => write!(f, "buffer too small: {what}"),
            Error::IndexOutOfRange => write!(f, "index out of range"),
            Error::Decode(code) => write!(f, "decode error {code}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;

pub(crate) fn ensure(cond: bool, err: Error) -> Result<()> {
    if cond { Ok(()) } else { Err(err) }
}
