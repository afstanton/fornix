//! Hybrid search result types.

use std::collections::HashMap;

/// Breakdown of how a hybrid score was computed.
#[derive(Debug, Clone, Default)]
pub struct ScoreBreakdown {
    /// The BM25 component score, if available.
    pub bm25: Option<f32>,
    /// The vector component score, if available.
    pub vector: Option<f32>,
    /// The final hybrid score after fusion.
    pub hybrid: f32,
    /// Label identifying the fusion path used.
    pub fusion_path: String,
}

/// Decomposition of the confidence score.
#[derive(Debug, Clone)]
pub struct ConfidenceComponents {
    /// Relative position of this result's score in the full result set.
    pub percentile: f32,
    /// Score gap between this result and the next one, normalised.
    pub margin: f32,
    /// Whether both BM25 and vector agreed on this result (1.0 = yes, 0.5 = no).
    pub agreement: f32,
}

/// A single hybrid search result.
#[derive(Debug, Clone)]
pub struct HybridResult {
    /// The document identifier.
    pub id: String,
    /// BM25 score for this document, if it appeared in BM25 results.
    pub bm25_score: Option<f32>,
    /// Vector similarity score for this document, if it appeared in vector results.
    pub vector_score: Option<f32>,
    /// The combined hybrid score after fusion.
    pub hybrid_score: f32,
    /// Per-component score breakdown for observability.
    pub score_breakdown: ScoreBreakdown,
    /// Confidence score in `[0.0, 1.0]`, computed after fusion.
    pub confidence_score: Option<f32>,
    /// Detailed confidence components, populated alongside `confidence_score`.
    pub confidence_components: Option<ConfidenceComponents>,
    /// Document metadata carried through from the source adapters.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl HybridResult {
    /// Construct a result with the minimum required fields.
    pub fn new(
        id: impl Into<String>,
        hybrid_score: f32,
        bm25_score: Option<f32>,
        vector_score: Option<f32>,
        fusion_path: impl Into<String>,
    ) -> Self {
        let breakdown = ScoreBreakdown {
            bm25: bm25_score,
            vector: vector_score,
            hybrid: hybrid_score,
            fusion_path: fusion_path.into(),
        };
        Self {
            id: id.into(),
            bm25_score,
            vector_score,
            hybrid_score,
            score_breakdown: breakdown,
            confidence_score: None,
            confidence_components: None,
            metadata: HashMap::new(),
        }
    }

    /// Returns `true` if this result appeared in both BM25 and vector results.
    pub fn has_both_sources(&self) -> bool {
        self.bm25_score.is_some() && self.vector_score.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(hybrid: f32, bm25: Option<f32>, vec: Option<f32>) -> HybridResult {
        HybridResult::new("id", hybrid, bm25, vec, "rrf")
    }

    #[test]
    fn new_sets_all_fields() {
        let r = result(0.8, Some(0.5), Some(0.9));
        assert_eq!(r.id, "id");
        assert!((r.hybrid_score - 0.8).abs() < 1e-6);
        assert_eq!(r.bm25_score, Some(0.5));
        assert_eq!(r.vector_score, Some(0.9));
        assert_eq!(r.score_breakdown.fusion_path, "rrf");
    }

    #[test]
    fn has_both_sources_true_when_both_present() {
        assert!(result(0.5, Some(0.3), Some(0.7)).has_both_sources());
    }

    #[test]
    fn has_both_sources_false_when_only_bm25() {
        assert!(!result(0.5, Some(0.3), None).has_both_sources());
    }

    #[test]
    fn has_both_sources_false_when_only_vector() {
        assert!(!result(0.5, None, Some(0.7)).has_both_sources());
    }

    #[test]
    fn confidence_is_none_by_default() {
        assert!(result(0.5, Some(0.3), Some(0.7)).confidence_score.is_none());
    }

    #[test]
    fn breakdown_carries_component_scores() {
        let r = result(0.8, Some(0.6), Some(0.9));
        assert_eq!(r.score_breakdown.bm25, Some(0.6));
        assert_eq!(r.score_breakdown.vector, Some(0.9));
        assert!((r.score_breakdown.hybrid - 0.8).abs() < 1e-6);
    }

    #[test]
    fn metadata_is_empty_by_default() {
        assert!(result(0.5, None, None).metadata.is_empty());
    }
}
