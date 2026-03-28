//! Error types for the BM25 layer.

use thiserror::Error;

/// Errors produced by BM25 adapters.
#[derive(Debug, Error)]
pub enum Error {
    /// The provided configuration is invalid.
    #[error("configuration error: {0}")]
    Configuration(String),

    /// A connection to the backend could not be established or was lost.
    #[error("connection error: {0}")]
    Connection(String),

    /// The adapter is not connected.
    #[error("bm25 adapter is not connected")]
    NotConnected,

    /// An indexing or search operation failed.
    #[error("operation failed: {0}")]
    Operation(String),

    /// A required external dependency is missing.
    #[error("missing dependency: {0}")]
    MissingDependency(String),

    /// An error from an underlying driver.
    #[error("backend error: {0}")]
    Backend(String),

    /// An unexpected error.
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Configuration(msg.into())
    }
    pub fn connection(msg: impl Into<String>) -> Self {
        Self::Connection(msg.into())
    }
    pub fn operation(msg: impl Into<String>) -> Self {
        Self::Operation(msg.into())
    }
    pub fn backend(msg: impl Into<String>) -> Self {
        Self::Backend(msg.into())
    }
}

/// Shorthand result type for BM25 operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_error_message() {
        let e = Error::config("k1 must be positive");
        assert_eq!(e.to_string(), "configuration error: k1 must be positive");
    }

    #[test]
    fn not_connected_message() {
        assert_eq!(Error::NotConnected.to_string(), "bm25 adapter is not connected");
    }

    #[test]
    fn operation_error_message() {
        let e = Error::operation("index write failed");
        assert_eq!(e.to_string(), "operation failed: index write failed");
    }

    #[test]
    fn backend_error_message() {
        let e = Error::backend("driver error");
        assert_eq!(e.to_string(), "backend error: driver error");
    }

    #[test]
    fn connection_error_message() {
        let e = Error::connection("timeout");
        assert_eq!(e.to_string(), "connection error: timeout");
    }

    #[test]
    fn missing_dependency_message() {
        let e = Error::MissingDependency("pg_bm25 extension".to_string());
        assert_eq!(e.to_string(), "missing dependency: pg_bm25 extension");
    }

    #[test]
    fn result_ok() {
        let r: Result<i32> = Ok(1);
        assert_eq!(r.unwrap(), 1);
    }

    #[test]
    fn result_err() {
        let r: Result<i32> = Err(Error::NotConnected);
        assert!(r.is_err());
    }
}
