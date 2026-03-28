//! Reranker trait and implementations.

use crate::rag::{error::Result, types::Context};

/// The reranker interface: takes a query and a list of contexts, and returns
/// them reordered (and optionally rescored) by relevance.
pub trait Reranker: Send + Sync {
    fn name(&self) -> &'static str;

    /// Rerank `contexts` for `query`, returning at most `top_k` results.
    fn rerank(&self, query: &str, contexts: Vec<Context>, top_k: Option<usize>) -> Result<Vec<Context>>;
}

pub mod null;
pub use null::NullReranker;
