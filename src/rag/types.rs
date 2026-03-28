//! RAG domain types: Chunk, Context, RagResult.

use std::collections::HashMap;
use std::time::Duration;

use crate::common::metadata::Metadata;
use crate::rag::tokenizer::count_tokens;

/// A chunk of text produced by a chunker.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// The chunk text.
    pub content: String,
    /// 0-based position of this chunk in the source document.
    pub index: usize,
    /// Byte offset of the first character in the source text.
    pub start_offset: usize,
    /// Byte offset past the last character in the source text.
    pub end_offset: usize,
    /// Arbitrary metadata (e.g. `chunk_type`, `overlap_applied`).
    pub metadata: Metadata,
    /// Index of the parent chunk, for parent/child chunking strategies.
    pub parent_id: Option<usize>,
}

impl Chunk {
    /// Convenience constructor.
    pub fn new(
        content: impl Into<String>,
        index: usize,
        start_offset: usize,
        end_offset: usize,
    ) -> Self {
        Self {
            content: content.into(),
            index,
            start_offset,
            end_offset,
            metadata: Metadata::new(),
            parent_id: None,
        }
    }

    /// Approximate token count (whitespace-split).
    pub fn token_count(&self) -> usize {
        count_tokens(&self.content)
    }

    /// Byte length of the chunk content.
    pub fn byte_len(&self) -> usize {
        self.end_offset - self.start_offset
    }
}

/// A single retrieved context item: a chunk paired with its retrieval score
/// and source provenance.
#[derive(Debug, Clone)]
pub struct Context {
    /// The text content of this context.
    pub content: String,
    /// Label identifying where this context came from (e.g. document id).
    pub source: Option<String>,
    /// The final score assigned to this context (after reranking if any).
    pub score: Option<f32>,
    /// The raw retrieval score before reranking.
    pub retrieval_score: Option<f32>,
    /// Arbitrary metadata.
    pub metadata: Metadata,
}

impl Context {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            source: None,
            score: None,
            retrieval_score: None,
            metadata: Metadata::new(),
        }
    }

    pub fn with_score(mut self, score: f32) -> Self {
        self.score = Some(score);
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn with_retrieval_score(mut self, score: f32) -> Self {
        self.retrieval_score = Some(score);
        self
    }

    /// Effective score: reranked score if present, otherwise retrieval score.
    pub fn effective_score(&self) -> Option<f32> {
        self.score.or(self.retrieval_score)
    }
}

/// The result of a RAG retrieval pass.
#[derive(Debug, Clone)]
pub struct RagResult {
    /// The original query.
    pub query: String,
    /// The strategy that produced this result.
    pub strategy: String,
    /// Retrieved context items, in relevance order.
    pub contexts: Vec<Context>,
    /// Arbitrary metadata about this retrieval.
    pub metadata: Metadata,
    /// Source record ids or labels contributing to this result.
    pub provenance: Vec<String>,
    /// Wall-clock time taken for retrieval.
    pub retrieval_time: Option<Duration>,
    /// Approximate total token count across all contexts.
    pub token_count: usize,
}

impl RagResult {
    pub fn new(
        query: impl Into<String>,
        strategy: impl Into<String>,
        contexts: Vec<Context>,
    ) -> Self {
        let token_count: usize = contexts.iter().map(|c| count_tokens(&c.content)).sum();
        Self {
            query: query.into(),
            strategy: strategy.into(),
            contexts,
            metadata: Metadata::new(),
            provenance: Vec::new(),
            retrieval_time: None,
            token_count,
        }
    }

    /// Whether this result contains any contexts.
    pub fn is_empty(&self) -> bool {
        self.contexts.is_empty()
    }

    /// Number of contexts returned.
    pub fn len(&self) -> usize {
        self.contexts.len()
    }
}

/// A filtered result wrapping a [`RagResult`] with a per-filter audit trail.
#[derive(Debug, Clone)]
pub struct FilteredResult {
    pub result: RagResult,
    pub filters_applied: Vec<FilterAuditEntry>,
}

/// Audit record for one output filter pass.
#[derive(Debug, Clone)]
pub struct FilterAuditEntry {
    pub filter: String,
    pub contexts_before: usize,
    pub contexts_after: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Chunk ---

    #[test]
    fn chunk_token_count() {
        let c = Chunk::new("hello world foo", 0, 0, 15);
        assert_eq!(c.token_count(), 3);
    }

    #[test]
    fn chunk_byte_len() {
        let c = Chunk::new("abc", 0, 5, 8);
        assert_eq!(c.byte_len(), 3);
    }

    #[test]
    fn chunk_parent_id_none_by_default() {
        let c = Chunk::new("text", 0, 0, 4);
        assert!(c.parent_id.is_none());
    }

    #[test]
    fn chunk_index_is_stored() {
        let c = Chunk::new("text", 7, 0, 4);
        assert_eq!(c.index, 7);
    }

    // --- Context ---

    #[test]
    fn context_effective_score_prefers_score() {
        let c = Context::new("text").with_score(0.9).with_retrieval_score(0.5);
        assert!((c.effective_score().unwrap() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn context_effective_score_falls_back_to_retrieval() {
        let c = Context::new("text").with_retrieval_score(0.7);
        assert!((c.effective_score().unwrap() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn context_effective_score_none_when_absent() {
        assert!(Context::new("text").effective_score().is_none());
    }

    #[test]
    fn context_source_is_stored() {
        let c = Context::new("text").with_source("doc-1");
        assert_eq!(c.source.as_deref(), Some("doc-1"));
    }

    // --- RagResult ---

    #[test]
    fn rag_result_token_count_sums_contexts() {
        let contexts = vec![
            Context::new("hello world"),
            Context::new("foo bar baz"),
        ];
        let r = RagResult::new("q", "vector", contexts);
        assert_eq!(r.token_count, 5);
    }

    #[test]
    fn rag_result_is_empty_when_no_contexts() {
        let r = RagResult::new("q", "hybrid", Vec::new());
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn rag_result_len_matches_context_count() {
        let ctxs = vec![Context::new("a"), Context::new("b"), Context::new("c")];
        let r = RagResult::new("q", "hybrid", ctxs);
        assert_eq!(r.len(), 3);
    }
}
