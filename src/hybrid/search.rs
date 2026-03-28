//! Hybrid search orchestrator.
//!
//! `HybridSearch` ties together a BM25 adapter, a vector adapter, a fusion
//! strategy, and confidence scoring into a single search call.

use async_trait::async_trait;

use crate::bm25::{
    adapter::{Bm25Adapter, SearchOptions as Bm25SearchOptions},
    result::Bm25Result,
};
use crate::common::namespace::Namespace;
use crate::hybrid::{
    config::{FusionMethod, HybridConfig},
    confidence::apply_confidence,
    error::{Error, Result},
    fusion::{FusedScore, FusionStrategy, ScoredItem},
    fusion::{linear::Linear, rrf::Rrf},
    result::HybridResult,
};
use crate::vector::{
    adapter::{SearchOptions as VectorSearchOptions, VectorAdapter},
    result::VectorResult,
};

/// Options for a single hybrid search request.
#[derive(Debug, Clone, Default)]
pub struct HybridSearchOptions {
    /// Override the configured number of BM25 candidates.
    pub bm25_candidates: Option<usize>,
    /// Override the configured number of vector candidates.
    pub vector_candidates: Option<usize>,
    /// Override the configured BM25 weight.
    pub bm25_weight: Option<f32>,
    /// Override the configured vector weight.
    pub vector_weight: Option<f32>,
    /// Override the configured fusion method.
    pub fusion: Option<FusionMethod>,
    /// Maximum number of results to return after fusion.
    pub limit: usize,
    /// Whether to compute confidence scores. Default: true.
    pub compute_confidence: bool,
}

impl HybridSearchOptions {
    pub fn new() -> Self {
        Self { limit: 20, compute_confidence: true, ..Default::default() }
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn without_confidence(mut self) -> Self {
        self.compute_confidence = false;
        self
    }
}

/// The hybrid search orchestrator.
///
/// Holds references to a BM25 adapter and a vector adapter and orchestrates
/// the full pipeline: parallel retrieval → fusion → optional confidence scoring.
pub struct HybridSearch<B, V>
where
    B: Bm25Adapter,
    V: VectorAdapter,
{
    bm25: B,
    vector: V,
    config: HybridConfig,
}

impl<B: Bm25Adapter, V: VectorAdapter> HybridSearch<B, V> {
    /// Construct a `HybridSearch` from adapters and configuration.
    pub fn new(bm25: B, vector: V, config: HybridConfig) -> Self {
        Self { bm25, vector, config }
    }

    /// Execute a hybrid search for `query` in `namespace`.
    pub async fn search(
        &self,
        query: &str,
        query_vector: &[f32],
        namespace: Option<&Namespace>,
        options: HybridSearchOptions,
    ) -> Result<Vec<HybridResult>> {
        let bm25_limit = options.bm25_candidates.unwrap_or(self.config.bm25_candidates);
        let vec_limit = options.vector_candidates.unwrap_or(self.config.vector_candidates);
        let bm25_weight = options.bm25_weight.unwrap_or(self.config.bm25_weight);
        let vector_weight = options.vector_weight.unwrap_or(self.config.vector_weight);
        let fusion_method = options.fusion.unwrap_or(self.config.fusion);

        // Fetch from both backends
        let bm25_results = self.fetch_bm25(query, namespace, bm25_limit).await?;
        let vector_results = self.fetch_vector(query_vector, namespace, vec_limit).await?;

        // Convert to ScoredItems (already sorted desc by the adapters)
        let bm25_items: Vec<ScoredItem> = bm25_results
            .iter()
            .map(|r| ScoredItem::new(r.id.clone(), r.score))
            .collect();
        let vector_items: Vec<ScoredItem> = vector_results
            .iter()
            .map(|r| ScoredItem::new(r.id.clone(), r.score()))
            .collect();

        // Fuse
        let fused: std::collections::HashMap<String, FusedScore> = match fusion_method {
            FusionMethod::Rrf => {
                Rrf::new(bm25_weight, vector_weight, self.config.rrf_k)
                    .fuse(&bm25_items, &vector_items)
            }
            FusionMethod::Linear => {
                Linear::new(bm25_weight, vector_weight, self.config.normalisation)
                    .fuse(&bm25_items, &vector_items)
            }
        };

        // Build sorted results
        let mut results: Vec<HybridResult> = fused
            .into_iter()
            .map(|(id, score)| {
                HybridResult::new(id, score.hybrid, score.bm25, score.vector, fusion_method_label(fusion_method))
            })
            .collect();

        results.sort_by(|a, b| {
            b.hybrid_score
                .partial_cmp(&a.hybrid_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(options.limit);

        if options.compute_confidence {
            apply_confidence(&mut results);
        }

        Ok(results)
    }

    async fn fetch_bm25(
        &self,
        query: &str,
        namespace: Option<&Namespace>,
        limit: usize,
    ) -> Result<Vec<Bm25Result>> {
        self.bm25
            .search(query, namespace, Bm25SearchOptions::default().with_limit(limit))
            .await
            .map_err(Error::from)
    }

    async fn fetch_vector(
        &self,
        query_vector: &[f32],
        namespace: Option<&Namespace>,
        limit: usize,
    ) -> Result<Vec<VectorResult>> {
        self.vector
            .nearest_neighbors(
                query_vector,
                namespace,
                VectorSearchOptions::default().with_limit(limit),
            )
            .await
            .map_err(Error::from)
    }
}

fn fusion_method_label(method: FusionMethod) -> &'static str {
    match method {
        FusionMethod::Rrf => "rrf",
        FusionMethod::Linear => "linear",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::adapters::MemoryBm25Adapter;
    use crate::bm25::adapter::IndexDocument;
    use crate::bm25::config::Bm25Config;
    use crate::vector::adapters::MemoryVectorAdapter;
    use crate::vector::VectorConfig;
    use crate::hybrid::config::HybridConfig;

    async fn setup() -> HybridSearch<MemoryBm25Adapter, MemoryVectorAdapter> {
        let bm25 = MemoryBm25Adapter::connect(Bm25Config::default()).await.unwrap();
        let vector = MemoryVectorAdapter::connect(VectorConfig::with_dimension(2)).await.unwrap();
        HybridSearch::new(bm25, vector, HybridConfig::default())
    }

    async fn setup_with_data() -> HybridSearch<MemoryBm25Adapter, MemoryVectorAdapter> {
        let bm25 = MemoryBm25Adapter::connect(Bm25Config::default()).await.unwrap();
        let vector = MemoryVectorAdapter::connect(VectorConfig::with_dimension(2)).await.unwrap();

        bm25.index(IndexDocument::new("rust", "rust systems programming language"), None).await.unwrap();
        bm25.index(IndexDocument::new("python", "python scripting easy language"), None).await.unwrap();
        bm25.index(IndexDocument::new("go", "go concurrency goroutines"), None).await.unwrap();

        vector.upsert("rust",   vec![1.0, 0.0], None, None).await.unwrap();
        vector.upsert("python", vec![0.7, 0.7], None, None).await.unwrap();
        vector.upsert("go",     vec![0.0, 1.0], None, None).await.unwrap();

        HybridSearch::new(bm25, vector, HybridConfig::default())
    }

    fn opts() -> HybridSearchOptions {
        HybridSearchOptions::new()
    }

    // --- basic correctness ---

    #[tokio::test]
    async fn search_empty_index_returns_no_results() {
        let hs = setup().await;
        let results = hs.search("rust", &[1.0, 0.0], None, opts()).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_returns_results_when_data_present() {
        let hs = setup_with_data().await;
        let results = hs.search("rust", &[1.0, 0.0], None, opts()).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn search_result_for_rust_query_favours_rust_document() {
        let hs = setup_with_data().await;
        let results = hs.search("rust", &[1.0, 0.0], None, opts()).await.unwrap();
        // "rust" should be top result given both BM25 and vector signal
        assert_eq!(results[0].id, "rust");
    }

    // --- result structure ---

    #[tokio::test]
    async fn results_are_sorted_descending_by_hybrid_score() {
        let hs = setup_with_data().await;
        let results = hs.search("language", &[0.8, 0.2], None, opts()).await.unwrap();
        for window in results.windows(2) {
            assert!(window[0].hybrid_score >= window[1].hybrid_score);
        }
    }

    #[tokio::test]
    async fn confidence_is_set_when_requested() {
        let hs = setup_with_data().await;
        let results = hs.search("rust", &[1.0, 0.0], None, opts()).await.unwrap();
        assert!(results.iter().all(|r| r.confidence_score.is_some()));
    }

    #[tokio::test]
    async fn confidence_not_set_when_disabled() {
        let hs = setup_with_data().await;
        let results = hs
            .search("rust", &[1.0, 0.0], None, opts().without_confidence())
            .await
            .unwrap();
        assert!(results.iter().all(|r| r.confidence_score.is_none()));
    }

    #[tokio::test]
    async fn limit_is_respected() {
        let hs = setup_with_data().await;
        let results = hs.search("language", &[0.7, 0.3], None, opts().with_limit(1)).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    // --- result in both sources has both scores ---

    #[tokio::test]
    async fn result_in_both_sources_has_both_scores() {
        let hs = setup_with_data().await;
        let results = hs.search("rust", &[1.0, 0.0], None, opts()).await.unwrap();
        let rust = results.iter().find(|r| r.id == "rust");
        if let Some(r) = rust {
            // "rust" appears in both BM25 and vector — both scores should be present
            assert!(r.bm25_score.is_some());
            assert!(r.vector_score.is_some());
        }
    }

    // --- fusion method override ---

    #[tokio::test]
    async fn linear_fusion_returns_results() {
        let hs = setup_with_data().await;
        let mut o = opts();
        o.fusion = Some(FusionMethod::Linear);
        let results = hs.search("rust", &[1.0, 0.0], None, o).await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn results_have_positive_hybrid_score() {
        let hs = setup_with_data().await;
        let results = hs.search("rust", &[1.0, 0.0], None, opts()).await.unwrap();
        assert!(results.iter().all(|r| r.hybrid_score >= 0.0));
    }
}
