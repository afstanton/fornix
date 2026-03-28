//! Hybrid search: fused vector + BM25 with score normalisation.
//!
//! Fusion strategies: RRF (Reciprocal Rank Fusion), linear weighted.
//! Normalisation: min-max, z-score, none.

/// A fused search result carrying both BM25 and vector sub-scores.
pub struct HybridResult {
    pub id: String,
    pub score: f32,
    pub bm25_score: Option<f32>,
    pub vector_score: Option<f32>,
    pub content: Option<String>,
}

/// Interface for embedding text into a vector.
pub trait EmbeddingProvider: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn embed(&self, text: &str) -> Result<Vec<f32>, Self::Error>;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Self::Error>;
    fn dimensions(&self) -> usize;
}

/// Interface for normalising a list of raw scores into [0, 1].
pub trait ScoreNormalizer: Send + Sync {
    fn normalize(&self, scores: &[f32]) -> Vec<f32>;
}

/// Interface for fusing ranked BM25 and vector result lists into a single ranking.
pub trait FusionStrategy: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn fuse(
        &self,
        bm25_results: &[crate::bm25::Bm25Result],
        vector_results: &[crate::vector::VectorResult],
        bm25_weight: f32,
        vector_weight: f32,
    ) -> Result<Vec<HybridResult>, Self::Error>;
}
