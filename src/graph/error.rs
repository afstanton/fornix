//! Error types for the graph layer.

use thiserror::Error;

/// Errors produced by graph adapters.
#[derive(Debug, Error)]
pub enum Error {
    /// The provided configuration is invalid.
    #[error("configuration error: {0}")]
    Configuration(String),

    /// A connection to the backend could not be established or was lost.
    #[error("connection error: {0}")]
    Connection(String),

    /// The adapter is not connected.
    #[error("graph adapter is not connected")]
    NotConnected,

    /// A graph operation failed.
    #[error("operation failed: {0}")]
    Operation(String),

    /// An entity or relation was not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// An invalid argument was supplied.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

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
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }
    pub fn invalid_arg(msg: impl Into<String>) -> Self {
        Self::InvalidArgument(msg.into())
    }
    pub fn backend(msg: impl Into<String>) -> Self {
        Self::Backend(msg.into())
    }
}

/// Shorthand result type for graph operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_error_message() {
        let e = Error::config("connection required");
        assert_eq!(e.to_string(), "configuration error: connection required");
    }

    #[test]
    fn not_connected_message() {
        assert_eq!(Error::NotConnected.to_string(), "graph adapter is not connected");
    }

    #[test]
    fn operation_error_message() {
        let e = Error::operation("write failed");
        assert_eq!(e.to_string(), "operation failed: write failed");
    }

    #[test]
    fn not_found_message() {
        let e = Error::not_found("entity 42");
        assert_eq!(e.to_string(), "not found: entity 42");
    }

    #[test]
    fn invalid_arg_message() {
        let e = Error::invalid_arg("depth must be >= 1");
        assert_eq!(e.to_string(), "invalid argument: depth must be >= 1");
    }

    #[test]
    fn result_ok() {
        let r: Result<i32> = Ok(1);
        assert!(matches!(r, Ok(1)));
    }

    #[test]
    fn result_err() {
        let r: Result<i32> = Err(Error::NotConnected);
        assert!(r.is_err());
    }
}
