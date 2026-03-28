//! Error types for the RAG layer.

use thiserror::Error;

/// Errors produced by RAG operations.
#[derive(Debug, Error)]
pub enum Error {
    /// The provided configuration is invalid.
    #[error("configuration error: {0}")]
    Configuration(String),

    /// A chunking operation failed.
    #[error("chunker error: {0}")]
    Chunker(String),

    /// A reranking operation failed.
    #[error("reranker error: {0}")]
    Reranker(String),

    /// An output filter failed.
    #[error("output filter error: {0}")]
    OutputFilter(String),

    /// An evaluation operation failed.
    #[error("evaluation error: {0}")]
    Evaluation(String),

    /// An unexpected error.
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Configuration(msg.into())
    }
    pub fn chunker(msg: impl Into<String>) -> Self {
        Self::Chunker(msg.into())
    }
    pub fn reranker(msg: impl Into<String>) -> Self {
        Self::Reranker(msg.into())
    }
    pub fn output_filter(msg: impl Into<String>) -> Self {
        Self::OutputFilter(msg.into())
    }
    pub fn evaluation(msg: impl Into<String>) -> Self {
        Self::Evaluation(msg.into())
    }
}

/// Shorthand result type for RAG operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_error_message() {
        let e = Error::config("max_tokens must be > 0");
        assert_eq!(e.to_string(), "configuration error: max_tokens must be > 0");
    }

    #[test]
    fn chunker_error_message() {
        let e = Error::chunker("text was empty");
        assert_eq!(e.to_string(), "chunker error: text was empty");
    }

    #[test]
    fn reranker_error_message() {
        let e = Error::reranker("model not loaded");
        assert_eq!(e.to_string(), "reranker error: model not loaded");
    }

    #[test]
    fn output_filter_error_message() {
        let e = Error::output_filter("filter panicked");
        assert_eq!(e.to_string(), "output filter error: filter panicked");
    }

    #[test]
    fn evaluation_error_message() {
        let e = Error::evaluation("llm call failed");
        assert_eq!(e.to_string(), "evaluation error: llm call failed");
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
