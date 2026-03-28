//! Error types for the cache layer.

use thiserror::Error;

/// Errors produced by cache adapters.
#[derive(Debug, Error)]
pub enum Error {
    /// The provided configuration is invalid.
    #[error("configuration error: {0}")]
    Configuration(String),

    /// A connection to the cache backend could not be established or was lost.
    #[error("connection error: {0}")]
    Connection(String),

    /// The adapter is not connected. Call `connect()` first.
    #[error("cache adapter is not connected")]
    NotConnected,

    /// A cache read, write, or delete operation failed.
    #[error("operation failed: {0}")]
    Operation(String),

    /// A required external dependency is missing.
    #[error("missing dependency: {0}")]
    MissingDependency(String),

    /// Serialisation of a cache value failed.
    #[error("serialisation error: {0}")]
    Serialisation(String),

    /// An error propagated from an underlying driver layer.
    #[error("backend error: {0}")]
    Backend(String),

    /// An unexpected error that does not fit the above categories.
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

    pub fn serialisation(msg: impl Into<String>) -> Self {
        Self::Serialisation(msg.into())
    }
}

/// Shorthand result type for cache operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_error_message() {
        let e = Error::config("namespace is required");
        assert_eq!(e.to_string(), "configuration error: namespace is required");
    }

    #[test]
    fn connection_error_message() {
        let e = Error::connection("refused");
        assert_eq!(e.to_string(), "connection error: refused");
    }

    #[test]
    fn not_connected_message() {
        let e = Error::NotConnected;
        assert_eq!(e.to_string(), "cache adapter is not connected");
    }

    #[test]
    fn operation_error_message() {
        let e = Error::operation("write failed");
        assert_eq!(e.to_string(), "operation failed: write failed");
    }

    #[test]
    fn serialisation_error_message() {
        let e = Error::serialisation("invalid utf-8");
        assert_eq!(e.to_string(), "serialisation error: invalid utf-8");
    }

    #[test]
    fn backend_error_message() {
        let e = Error::backend("driver crashed");
        assert_eq!(e.to_string(), "backend error: driver crashed");
    }

    #[test]
    fn missing_dependency_message() {
        let e = Error::MissingDependency("redis driver".to_string());
        assert_eq!(e.to_string(), "missing dependency: redis driver");
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

    #[test]
    fn from_boxed_error() {
        let inner: Box<dyn std::error::Error + Send + Sync> =
            Box::new(std::io::Error::new(std::io::ErrorKind::TimedOut, "timed out"));
        let e = Error::from(inner);
        assert!(matches!(e, Error::Other(_)));
        assert!(e.to_string().contains("timed out"));
    }
}
