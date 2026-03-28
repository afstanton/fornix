//! Retrieval-Augmented Generation: chunking, retrieval strategies,
//! reranking, output filtering, and evaluation.

/// A chunk of source content used as retrieval context.
pub struct RagContext {
    pub id: Option<String>,
    pub content: String,
    pub score: Option<f32>,
    pub retrieval_score: Option<f32>,
    pub source: Option<String>,
}

/// The result of a RAG search pass.
pub struct RagResult {
    pub query: String,
    pub contexts: Vec<RagContext>,
}

/// Interface for chunking source text into retrievable units.
pub trait Chunker: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn chunk(&self, text: &str) -> Result<Vec<String>, Self::Error>;
}

/// Interface for retrieval strategies (vector-only, BM25-only, hybrid,
/// graph, HyDE, parent-child, multi-query, dialectic).
pub trait RagStrategy: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn search(&self, query: &str) -> Result<RagResult, Self::Error>;
    fn search_contexts(&self, query: &str) -> Result<Vec<RagContext>, Self::Error>;
}

/// Interface for result reranking (null, cross-encoder).
pub trait Reranker: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn rerank(
        &self,
        query: &str,
        contexts: Vec<RagContext>,
        top_k: Option<usize>,
    ) -> Result<Vec<RagContext>, Self::Error>;
}

/// Interface for output filters applied after retrieval
/// (PII redaction, entity denylist, citation grounding, hallucination review, etc.).
pub trait OutputFilter: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn name(&self) -> &'static str;
    fn filter(&self, result: RagResult, query: &str) -> Result<RagResult, Self::Error>;
}

/// Interface for RAG evaluation metrics
/// (context precision, recall, faithfulness, answer relevance).
pub trait EvaluationMetric: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn name(&self) -> &'static str;
    fn score(&self, result: &RagResult, reference: &str) -> Result<f32, Self::Error>;
}
