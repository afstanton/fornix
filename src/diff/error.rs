//! Error types for the diff layer.

use thiserror::Error;

/// Errors produced by diff operations.
#[derive(Debug, Error)]
pub enum Error {
    /// The provided configuration is invalid.
    #[error("configuration error: {0}")]
    Configuration(String),

    /// A diffing operation failed.
    #[error("diff error: {0}")]
    Diff(String),

    /// An unexpected error.
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Configuration(msg.into())
    }
    pub fn diff(msg: impl Into<String>) -> Self {
        Self::Diff(msg.into())
    }
}

/// Shorthand result type for diff operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_error_message() {
        assert_eq!(
            Error::config("snippet_max must be > 0").to_string(),
            "configuration error: snippet_max must be > 0"
        );
    }

    #[test]
    fn diff_error_message() {
        assert_eq!(
            Error::diff("LCS computation failed").to_string(),
            "diff error: LCS computation failed"
        );
    }

    #[test]
    fn result_ok() {
        let r: Result<i32> = Ok(1);
        assert_eq!(r.unwrap(), 1);
    }
}
