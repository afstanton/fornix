//! Error types for the store layer.

use thiserror::Error;

/// Errors produced by storage adapters.
#[derive(Debug, Error)]
pub enum Error {
    /// The provided configuration is invalid.
    #[error("configuration error: {0}")]
    Configuration(String),

    /// A connection to the backend could not be established or was lost.
    #[error("connection error: {0}")]
    Connection(String),

    /// The adapter is not connected. Call `connect()` first.
    #[error("adapter is not connected")]
    NotConnected,

    /// A read, write, or query operation failed.
    #[error("operation failed: {0}")]
    Operation(String),

    /// The requested record was not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// A namespace does not exist or has not been initialised.
    #[error("namespace not found: {0}")]
    NamespaceNotFound(String),

    /// Vector dimension mismatch between stored records and a query.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// A required external dependency (e.g. a Postgres extension) is missing.
    #[error("missing dependency: {0}")]
    MissingDependency(String),

    /// An error propagated from an underlying I/O or driver layer.
    #[error("backend error: {0}")]
    Backend(String),

    /// An unexpected error that does not fit the above categories.
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    /// Convenience constructor for [`Error::Configuration`].
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Configuration(msg.into())
    }

    /// Convenience constructor for [`Error::Connection`].
    pub fn connection(msg: impl Into<String>) -> Self {
        Self::Connection(msg.into())
    }

    /// Convenience constructor for [`Error::Operation`].
    pub fn operation(msg: impl Into<String>) -> Self {
        Self::Operation(msg.into())
    }

    /// Convenience constructor for [`Error::Backend`].
    pub fn backend(msg: impl Into<String>) -> Self {
        Self::Backend(msg.into())
    }
}

/// Shorthand result type for store operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    // --- Display / message formatting ---

    #[test]
    fn configuration_error_message() {
        let e = Error::config("url is required");
        assert_eq!(e.to_string(), "configuration error: url is required");
    }

    #[test]
    fn connection_error_message() {
        let e = Error::connection("timed out after 10s");
        assert_eq!(e.to_string(), "connection error: timed out after 10s");
    }

    #[test]
    fn not_connected_message() {
        let e = Error::NotConnected;
        assert_eq!(e.to_string(), "adapter is not connected");
    }

    #[test]
    fn operation_error_message() {
        let e = Error::operation("insert failed");
        assert_eq!(e.to_string(), "operation failed: insert failed");
    }

    #[test]
    fn not_found_message() {
        let e = Error::NotFound("record-42".to_string());
        assert_eq!(e.to_string(), "not found: record-42");
    }

    #[test]
    fn namespace_not_found_message() {
        let e = Error::NamespaceNotFound("documents".to_string());
        assert_eq!(e.to_string(), "namespace not found: documents");
    }

    #[test]
    fn dimension_mismatch_message() {
        let e = Error::DimensionMismatch { expected: 384, actual: 512 };
        assert_eq!(e.to_string(), "dimension mismatch: expected 384, got 512");
    }

    #[test]
    fn missing_dependency_message() {
        let e = Error::MissingDependency("pgvector extension".to_string());
        assert_eq!(e.to_string(), "missing dependency: pgvector extension");
    }

    #[test]
    fn backend_error_message() {
        let e = Error::backend("driver panicked");
        assert_eq!(e.to_string(), "backend error: driver panicked");
    }

    // --- Convenience constructors accept &str and String ---

    #[test]
    fn config_accepts_str_slice() {
        let e = Error::config("bad value");
        assert!(matches!(e, Error::Configuration(_)));
    }

    #[test]
    fn config_accepts_owned_string() {
        let e = Error::config(String::from("bad value"));
        assert!(matches!(e, Error::Configuration(_)));
    }

    #[test]
    fn connection_constructor() {
        let e = Error::connection("refused");
        assert!(matches!(e, Error::Connection(_)));
    }

    #[test]
    fn operation_constructor() {
        let e = Error::operation("deadlock");
        assert!(matches!(e, Error::Operation(_)));
    }

    #[test]
    fn backend_constructor() {
        let e = Error::backend("io error");
        assert!(matches!(e, Error::Backend(_)));
    }

    // --- Result type alias ---

    #[test]
    fn result_ok_variant() {
        let r: Result<i32> = Ok(42);
        assert_eq!(r.unwrap(), 42);
    }

    #[test]
    fn result_err_variant() {
        let r: Result<i32> = Err(Error::NotConnected);
        assert!(r.is_err());
    }

    // --- Other / From<Box<dyn Error>> ---

    #[test]
    fn other_via_from_boxed_error() {
        let inner: Box<dyn std::error::Error + Send + Sync> =
            Box::new(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe broke"));
        let e = Error::from(inner);
        assert!(matches!(e, Error::Other(_)));
        assert!(e.to_string().contains("pipe broke"));
    }
}
