//! Error types for the hybrid search layer.

use thiserror::Error;

/// Errors produced by hybrid search operations.
#[derive(Debug, Error)]
pub enum Error {
    /// The provided configuration is invalid.
    #[error("configuration error: {0}")]
    Configuration(String),

    /// An upstream BM25 search operation failed.
    #[error("bm25 error: {0}")]
    Bm25(String),

    /// An upstream vector search operation failed.
    #[error("vector error: {0}")]
    Vector(String),

    /// An embedding operation failed.
    #[error("embedding error: {0}")]
    Embedding(String),

    /// An unexpected error.
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Configuration(msg.into())
    }
    pub fn bm25(msg: impl Into<String>) -> Self {
        Self::Bm25(msg.into())
    }
    pub fn vector(msg: impl Into<String>) -> Self {
        Self::Vector(msg.into())
    }
    pub fn embedding(msg: impl Into<String>) -> Self {
        Self::Embedding(msg.into())
    }
}

/// Convert from BM25 errors automatically.
impl From<crate::bm25::error::Error> for Error {
    fn from(e: crate::bm25::error::Error) -> Self {
        Self::Bm25(e.to_string())
    }
}

/// Convert from vector errors automatically.
impl From<crate::vector::error::Error> for Error {
    fn from(e: crate::vector::error::Error) -> Self {
        Self::Vector(e.to_string())
    }
}

/// Shorthand result type for hybrid operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_error_message() {
        let e = Error::config("weights must sum to 1.0");
        assert_eq!(e.to_string(), "configuration error: weights must sum to 1.0");
    }

    #[test]
    fn bm25_error_message() {
        let e = Error::bm25("index not connected");
        assert_eq!(e.to_string(), "bm25 error: index not connected");
    }

    #[test]
    fn vector_error_message() {
        let e = Error::vector("dimension mismatch");
        assert_eq!(e.to_string(), "vector error: dimension mismatch");
    }

    #[test]
    fn embedding_error_message() {
        let e = Error::embedding("model not loaded");
        assert_eq!(e.to_string(), "embedding error: model not loaded");
    }

    #[test]
    fn from_bm25_error() {
        let bm25_err = crate::bm25::error::Error::NotConnected;
        let e = Error::from(bm25_err);
        assert!(matches!(e, Error::Bm25(_)));
    }

    #[test]
    fn from_vector_error() {
        let vec_err = crate::vector::error::Error::NotConnected;
        let e = Error::from(vec_err);
        assert!(matches!(e, Error::Vector(_)));
    }

    #[test]
    fn result_ok() {
        let r: Result<i32> = Ok(1);
        assert!(matches!(r, Ok(1)));
    }

    #[test]
    fn result_err() {
        let r: Result<i32> = Err(Error::config("bad"));
        assert!(r.is_err());
    }
}
