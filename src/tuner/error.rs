//! Error types for the tuner layer.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("configuration error: {0}")]
    Configuration(String),
    #[error("tuning failed: {0}")]
    Tuning(String),
    #[error("evaluation error: {0}")]
    Evaluation(String),
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    pub fn config(msg: impl Into<String>) -> Self { Self::Configuration(msg.into()) }
    pub fn tuning(msg: impl Into<String>) -> Self { Self::Tuning(msg.into()) }
    pub fn evaluation(msg: impl Into<String>) -> Self { Self::Evaluation(msg.into()) }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn config_error() {
        assert_eq!(Error::config("dataset required").to_string(), "configuration error: dataset required");
    }
    #[test]
    fn tuning_error() {
        assert_eq!(Error::tuning("no candidates").to_string(), "tuning failed: no candidates");
    }
}
