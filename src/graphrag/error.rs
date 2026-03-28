//! Error types for the GraphRAG layer.

use thiserror::Error;

/// Errors produced by GraphRAG operations.
#[derive(Debug, Error)]
pub enum Error {
    #[error("configuration error: {0}")]
    Configuration(String),
    #[error("extraction error: {0}")]
    Extraction(String),
    #[error("search error: {0}")]
    Search(String),
    #[error("ingestion error: {0}")]
    Ingestion(String),
    #[error("graph error: {0}")]
    Graph(#[from] crate::graph::error::Error),
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    pub fn config(msg: impl Into<String>) -> Self { Self::Configuration(msg.into()) }
    pub fn extraction(msg: impl Into<String>) -> Self { Self::Extraction(msg.into()) }
    pub fn search(msg: impl Into<String>) -> Self { Self::Search(msg.into()) }
    pub fn ingestion(msg: impl Into<String>) -> Self { Self::Ingestion(msg.into()) }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_error() {
        assert_eq!(Error::config("graph required").to_string(), "configuration error: graph required");
    }

    #[test]
    fn extraction_error() {
        assert_eq!(Error::extraction("llm failed").to_string(), "extraction error: llm failed");
    }

    #[test]
    fn search_error() {
        assert_eq!(Error::search("no entity found").to_string(), "search error: no entity found");
    }
}
