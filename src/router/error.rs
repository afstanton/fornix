//! Error types for the router layer.

use thiserror::Error;

/// Errors produced by router operations.
#[derive(Debug, Error)]
pub enum Error {
    /// Configuration is missing required fields.
    #[error("configuration error: {0}")]
    Configuration(String),

    /// No models are available to route to.
    #[error("no available models: {0}")]
    NoModels(String),

    /// The routing strategy could not find a suitable model.
    #[error("routing failed: {0}")]
    RoutingFailed(String),

    /// A model or provider was not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// A regex pattern is invalid.
    #[error("invalid pattern: {0}")]
    InvalidPattern(String),

    /// A Random Forest training or inference error.
    #[error("forest error: {0}")]
    Forest(String),

    /// An unexpected error.
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Configuration(msg.into())
    }
    pub fn no_models(msg: impl Into<String>) -> Self {
        Self::NoModels(msg.into())
    }
    pub fn routing(msg: impl Into<String>) -> Self {
        Self::RoutingFailed(msg.into())
    }
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }
    pub fn forest(msg: impl Into<String>) -> Self {
        Self::Forest(msg.into())
    }
}

/// Shorthand result type for router operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_error_message() {
        assert_eq!(
            Error::config("strategy required").to_string(),
            "configuration error: strategy required"
        );
    }

    #[test]
    fn no_models_message() {
        assert_eq!(
            Error::no_models("empty pool").to_string(),
            "no available models: empty pool"
        );
    }

    #[test]
    fn routing_failed_message() {
        assert_eq!(
            Error::routing("threshold mismatch").to_string(),
            "routing failed: threshold mismatch"
        );
    }

    #[test]
    fn not_found_message() {
        assert_eq!(
            Error::not_found("gpt-5").to_string(),
            "not found: gpt-5"
        );
    }

    #[test]
    fn forest_error_message() {
        assert_eq!(
            Error::forest("empty feature matrix").to_string(),
            "forest error: empty feature matrix"
        );
    }

    #[test]
    fn result_ok() {
        let r: Result<i32> = Ok(1);
        assert_eq!(r.unwrap(), 1);
    }
}
