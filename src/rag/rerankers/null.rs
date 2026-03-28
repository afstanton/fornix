//! Null reranker — passes contexts through unchanged.

use crate::rag::{error::Result, rerankers::Reranker, types::Context};

/// A no-op reranker. Returns the input contexts unmodified, respecting `top_k`.
#[derive(Debug, Clone, Default)]
pub struct NullReranker;

impl Reranker for NullReranker {
    fn name(&self) -> &'static str {
        "null"
    }

    fn rerank(&self, _query: &str, contexts: Vec<Context>, top_k: Option<usize>) -> Result<Vec<Context>> {
        Ok(match top_k {
            Some(k) => contexts.into_iter().take(k).collect(),
            None => contexts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::types::Context;

    #[test]
    fn returns_all_contexts_without_top_k() {
        let ctxs = vec![Context::new("a"), Context::new("b"), Context::new("c")];
        let result = NullReranker.rerank("q", ctxs, None).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn respects_top_k() {
        let ctxs = vec![Context::new("a"), Context::new("b"), Context::new("c")];
        let result = NullReranker.rerank("q", ctxs, Some(2)).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn top_k_larger_than_input_returns_all() {
        let ctxs = vec![Context::new("a")];
        let result = NullReranker.rerank("q", ctxs, Some(10)).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn preserves_order() {
        let ctxs = vec![
            Context::new("first").with_score(1.0),
            Context::new("second").with_score(0.5),
        ];
        let result = NullReranker.rerank("q", ctxs, None).unwrap();
        assert_eq!(result[0].content, "first");
        assert_eq!(result[1].content, "second");
    }

    #[test]
    fn name_is_null() {
        assert_eq!(NullReranker.name(), "null");
    }
}
