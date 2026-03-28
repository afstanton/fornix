//! Fusion strategy trait and a scored item type.

use std::collections::HashMap;

/// A document identifier paired with a pre-fusion score.
/// The score is used by linear fusion; RRF uses only the ordering.
#[derive(Debug, Clone)]
pub struct ScoredItem {
    /// Document identifier.
    pub id: String,
    /// Raw score from the upstream retriever.
    pub score: f32,
}

impl ScoredItem {
    pub fn new(id: impl Into<String>, score: f32) -> Self {
        Self { id: id.into(), score }
    }
}

/// The scores produced by a fusion pass for a single document.
#[derive(Debug, Clone)]
pub struct FusedScore {
    /// Combined hybrid score.
    pub hybrid: f32,
    /// BM25 contribution to the hybrid score, if present.
    pub bm25: Option<f32>,
    /// Vector contribution to the hybrid score, if present.
    pub vector: Option<f32>,
}

/// Fusion strategy trait.
///
/// Takes pre-ranked BM25 and vector result lists (sorted descending by score)
/// and produces a combined ranking.
pub trait FusionStrategy: Send + Sync {
    /// Fuse ranked BM25 and vector results into a combined score map.
    ///
    /// Both slices must be sorted in descending score order before being
    /// passed in. The strategy may use raw scores, rank positions, or both.
    ///
    /// Returns a map of document id → [`FusedScore`], unordered.
    /// Callers sort the output by `FusedScore::hybrid` descending.
    fn fuse(
        &self,
        bm25_items: &[ScoredItem],
        vector_items: &[ScoredItem],
    ) -> HashMap<String, FusedScore>;
}
