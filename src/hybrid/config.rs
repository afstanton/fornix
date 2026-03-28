//! Hybrid search configuration.

use crate::store::{
    config::AdapterConfig,
    error::{Error as StoreError, Result as StoreResult},
};

/// Which fusion strategy to use when combining BM25 and vector results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion — rank-based, insensitive to score scale.
    #[default]
    Rrf,
    /// Linear combination of normalised scores.
    Linear,
}

/// Which score normalisation to apply before linear fusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormalisationMethod {
    /// Scale scores to `[0, 1]` using min-max normalisation.
    #[default]
    MinMax,
    /// Standardise scores to zero mean and unit variance.
    ZScore,
    /// Pass scores through unchanged.
    None,
}

/// Configuration for hybrid search.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Default fusion strategy.
    pub fusion: FusionMethod,

    /// Weight applied to BM25 scores. Must be in `[0.0, 1.0]`.
    pub bm25_weight: f32,

    /// Weight applied to vector scores. Must be in `[0.0, 1.0]`.
    pub vector_weight: f32,

    /// Score normalisation applied before linear fusion.
    /// Ignored when `fusion` is `Rrf`.
    pub normalisation: NormalisationMethod,

    /// The `k` parameter for RRF: `score = 1 / (k + rank)`.
    /// Larger values reduce the influence of top-ranked results.
    /// Default: 60 (standard literature recommendation).
    pub rrf_k: u32,

    /// Maximum number of BM25 candidate results to retrieve.
    pub bm25_candidates: usize,

    /// Maximum number of vector candidate results to retrieve.
    pub vector_candidates: usize,

    /// Minimum vector similarity threshold for vector candidates.
    pub min_similarity: Option<f32>,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            fusion: FusionMethod::Rrf,
            bm25_weight: 0.5,
            vector_weight: 0.5,
            normalisation: NormalisationMethod::MinMax,
            rrf_k: 60,
            bm25_candidates: 200,
            vector_candidates: 200,
            min_similarity: None,
        }
    }
}

impl AdapterConfig for HybridConfig {
    fn adapter_name(&self) -> &'static str {
        "hybrid"
    }

    fn validate(&self) -> StoreResult<()> {
        if !(0.0..=1.0).contains(&self.bm25_weight) {
            return Err(StoreError::config("bm25_weight must be in [0.0, 1.0]"));
        }
        if !(0.0..=1.0).contains(&self.vector_weight) {
            return Err(StoreError::config("vector_weight must be in [0.0, 1.0]"));
        }
        if self.rrf_k == 0 {
            return Err(StoreError::config("rrf_k must be greater than zero"));
        }
        if self.bm25_candidates == 0 {
            return Err(StoreError::config("bm25_candidates must be greater than zero"));
        }
        if self.vector_candidates == 0 {
            return Err(StoreError::config("vector_candidates must be greater than zero"));
        }
        if let Some(min) = self.min_similarity {
            if !(0.0..=1.0).contains(&min) {
                return Err(StoreError::config("min_similarity must be in [0.0, 1.0]"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_fusion_is_rrf() {
        assert_eq!(HybridConfig::default().fusion, FusionMethod::Rrf);
    }

    #[test]
    fn default_weights_are_equal() {
        let c = HybridConfig::default();
        assert!((c.bm25_weight - 0.5).abs() < 1e-6);
        assert!((c.vector_weight - 0.5).abs() < 1e-6);
    }

    #[test]
    fn default_rrf_k_is_60() {
        assert_eq!(HybridConfig::default().rrf_k, 60);
    }

    #[test]
    fn default_candidates_are_200() {
        let c = HybridConfig::default();
        assert_eq!(c.bm25_candidates, 200);
        assert_eq!(c.vector_candidates, 200);
    }

    #[test]
    fn adapter_name_is_hybrid() {
        assert_eq!(HybridConfig::default().adapter_name(), "hybrid");
    }

    #[test]
    fn validate_passes_for_valid_config() {
        assert!(HybridConfig::default().validate().is_ok());
    }

    #[test]
    fn validate_fails_for_bm25_weight_above_one() {
        let c = HybridConfig { bm25_weight: 1.1, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_fails_for_bm25_weight_below_zero() {
        let c = HybridConfig { bm25_weight: -0.1, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_fails_for_vector_weight_above_one() {
        let c = HybridConfig { vector_weight: 1.5, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_fails_for_zero_rrf_k() {
        let c = HybridConfig { rrf_k: 0, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_fails_for_zero_bm25_candidates() {
        let c = HybridConfig { bm25_candidates: 0, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_fails_for_zero_vector_candidates() {
        let c = HybridConfig { vector_candidates: 0, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_fails_for_min_similarity_above_one() {
        let c = HybridConfig { min_similarity: Some(1.5), ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_passes_with_valid_min_similarity() {
        let c = HybridConfig { min_similarity: Some(0.7), ..Default::default() };
        assert!(c.validate().is_ok());
    }

    #[test]
    fn normalisation_method_default_is_min_max() {
        assert_eq!(HybridConfig::default().normalisation, NormalisationMethod::MinMax);
    }
}
