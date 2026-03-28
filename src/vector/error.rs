//! Error types for the vector layer.

use thiserror::Error;

/// Errors produced by vector storage adapters.
#[derive(Debug, Error)]
pub enum Error {
    /// The provided configuration is invalid.
    #[error("configuration error: {0}")]
    Configuration(String),

    /// A connection to the backend could not be established or was lost.
    #[error("connection error: {0}")]
    Connection(String),

    /// The adapter is not connected. Call `connect()` first.
    #[error("vector adapter is not connected")]
    NotConnected,

    /// A read or write operation failed.
    #[error("operation failed: {0}")]
    Operation(String),

    /// The requested record or namespace was not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// The namespace does not exist or has not been initialised.
    #[error("namespace not found: {0}")]
    NamespaceNotFound(String),

    /// Vector dimension mismatch between a stored record and a query.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// A required external dependency is missing.
    #[error("missing dependency: {0}")]
    MissingDependency(String),

    /// An error propagated from an underlying driver layer.
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

    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }
}

/// Shorthand result type for vector operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_error_message() {
        let e = Error::config("dimension is required");
        assert_eq!(e.to_string(), "configuration error: dimension is required");
    }

    #[test]
    fn connection_error_message() {
        let e = Error::connection("refused");
        assert_eq!(e.to_string(), "connection error: refused");
    }

    #[test]
    fn not_connected_message() {
        assert_eq!(Error::NotConnected.to_string(), "vector adapter is not connected");
    }

    #[test]
    fn operation_error_message() {
        let e = Error::operation("upsert failed");
        assert_eq!(e.to_string(), "operation failed: upsert failed");
    }

    #[test]
    fn not_found_message() {
        let e = Error::NotFound("rec-1".to_string());
        assert_eq!(e.to_string(), "not found: rec-1");
    }

    #[test]
    fn namespace_not_found_message() {
        let e = Error::NamespaceNotFound("docs".to_string());
        assert_eq!(e.to_string(), "namespace not found: docs");
    }

    #[test]
    fn dimension_mismatch_message() {
        let e = Error::dimension_mismatch(384, 512);
        assert_eq!(e.to_string(), "dimension mismatch: expected 384, got 512");
    }

    #[test]
    fn missing_dependency_message() {
        let e = Error::MissingDependency("pgvector extension".to_string());
        assert_eq!(e.to_string(), "missing dependency: pgvector extension");
    }

    #[test]
    fn backend_error_message() {
        let e = Error::backend("driver error");
        assert_eq!(e.to_string(), "backend error: driver error");
    }

    #[test]
    fn result_ok() {
        let r: Result<i32> = Ok(42);
        assert!(matches!(r, Ok(42)));
    }

    #[test]
    fn result_err() {
        let r: Result<i32> = Err(Error::NotConnected);
        assert!(r.is_err());
    }
}
