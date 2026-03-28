//! Reciprocal Rank Fusion (RRF).
//!
//! RRF scores a document by summing reciprocal rank contributions from each
//! ranked list it appears in:
//!
//! ```text
//! score(d) = Σ  weight_i / (k + rank_i(d))
//! ```
//!
//! where `rank_i(d)` is the 1-based rank of document `d` in list `i`, `k`
//! is a smoothing constant (default 60), and `weight_i` is the list weight.
//!
//! Key property: only rank positions matter, not raw scores. This makes RRF
//! robust to score scale differences between BM25 and vector systems.

use std::collections::HashMap;

use crate::hybrid::fusion::{FusedScore, FusionStrategy, ScoredItem};

/// RRF fusion strategy.
#[derive(Debug, Clone)]
pub struct Rrf {
    /// BM25 list weight.
    pub bm25_weight: f32,
    /// Vector list weight.
    pub vector_weight: f32,
    /// Smoothing constant. Standard recommendation is 60.
    pub k: u32,
}

impl Rrf {
    pub fn new(bm25_weight: f32, vector_weight: f32, k: u32) -> Self {
        Self { bm25_weight, vector_weight, k }
    }
}

impl Default for Rrf {
    fn default() -> Self {
        Self::new(0.5, 0.5, 60)
    }
}

impl FusionStrategy for Rrf {
    fn fuse(
        &self,
        bm25_items: &[ScoredItem],
        vector_items: &[ScoredItem],
    ) -> HashMap<String, FusedScore> {
        let mut hybrid: HashMap<String, f32> = HashMap::new();
        let mut bm25_scores: HashMap<String, f32> = HashMap::new();
        let mut vector_scores: HashMap<String, f32> = HashMap::new();

        // Apply RRF from BM25 list (rank is 1-based)
        for (rank_zero, item) in bm25_items.iter().enumerate() {
            let rrf = self.bm25_weight / (self.k as f32 + (rank_zero + 1) as f32);
            *hybrid.entry(item.id.clone()).or_insert(0.0) += rrf;
            bm25_scores.insert(item.id.clone(), item.score);
        }

        // Apply RRF from vector list (rank is 1-based)
        for (rank_zero, item) in vector_items.iter().enumerate() {
            let rrf = self.vector_weight / (self.k as f32 + (rank_zero + 1) as f32);
            *hybrid.entry(item.id.clone()).or_insert(0.0) += rrf;
            vector_scores.insert(item.id.clone(), item.score);
        }

        hybrid
            .into_iter()
            .map(|(id, score)| {
                let fused = FusedScore {
                    hybrid: score,
                    bm25: bm25_scores.get(&id).copied(),
                    vector: vector_scores.get(&id).copied(),
                };
                (id, fused)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items(ids: &[&str], scores: &[f32]) -> Vec<ScoredItem> {
        ids.iter().zip(scores.iter()).map(|(id, s)| ScoredItem::new(*id, *s)).collect()
    }

    #[test]
    fn empty_inputs_produce_empty_output() {
        let rrf = Rrf::default();
        assert!(rrf.fuse(&[], &[]).is_empty());
    }

    #[test]
    fn bm25_only_produces_rrf_scores() {
        let rrf = Rrf::new(1.0, 0.0, 60);
        let bm25 = items(&["a", "b"], &[1.0, 0.5]);
        let result = rrf.fuse(&bm25, &[]);
        // a is rank 1: 1 / (60 + 1) ≈ 0.01639
        // b is rank 2: 1 / (60 + 2) ≈ 0.01613
        assert!(result["a"].hybrid > result["b"].hybrid);
    }

    #[test]
    fn vector_only_produces_rrf_scores() {
        let rrf = Rrf::new(0.0, 1.0, 60);
        let vec = items(&["x", "y"], &[0.9, 0.1]);
        let result = rrf.fuse(&[], &vec);
        assert!(result["x"].hybrid > result["y"].hybrid);
    }

    #[test]
    fn document_appearing_in_both_lists_gets_boosted() {
        let rrf = Rrf::default(); // equal weights
        let bm25 = items(&["shared", "bm25_only"], &[1.0, 0.8]);
        let vec = items(&["shared", "vec_only"], &[0.9, 0.7]);
        let result = rrf.fuse(&bm25, &vec);

        // "shared" should have higher score than either single-list entry
        let shared = result["shared"].hybrid;
        let bm25_only = result["bm25_only"].hybrid;
        let vec_only = result["vec_only"].hybrid;
        assert!(shared > bm25_only);
        assert!(shared > vec_only);
    }

    #[test]
    fn rrf_score_only_depends_on_rank_not_raw_score() {
        let rrf = Rrf::default();
        // Same rank positions, very different raw scores
        let bm25_a = items(&["a"], &[1000.0]);
        let bm25_b = items(&["b"], &[0.001]);
        let result_a = rrf.fuse(&bm25_a, &[]);
        let result_b = rrf.fuse(&bm25_b, &[]);
        // Both are rank 1, so hybrid scores should be identical
        assert!((result_a["a"].hybrid - result_b["b"].hybrid).abs() < 1e-10);
    }

    #[test]
    fn bm25_score_carried_through_to_fused_score() {
        let rrf = Rrf::default();
        let bm25 = items(&["a"], &[42.0]);
        let result = rrf.fuse(&bm25, &[]);
        assert_eq!(result["a"].bm25, Some(42.0));
        assert!(result["a"].vector.is_none());
    }

    #[test]
    fn vector_score_carried_through_to_fused_score() {
        let rrf = Rrf::default();
        let vec = items(&["a"], &[0.88]);
        let result = rrf.fuse(&[], &vec);
        assert_eq!(result["a"].vector, Some(0.88));
        assert!(result["a"].bm25.is_none());
    }

    #[test]
    fn higher_k_reduces_score_differences() {
        let low_k = Rrf::new(0.5, 0.5, 1);
        let high_k = Rrf::new(0.5, 0.5, 1000);
        let bm25 = items(&["rank1", "rank2"], &[1.0, 0.5]);

        let result_low = low_k.fuse(&bm25, &[]);
        let result_high = high_k.fuse(&bm25, &[]);

        let diff_low = result_low["rank1"].hybrid - result_low["rank2"].hybrid;
        let diff_high = result_high["rank1"].hybrid - result_high["rank2"].hybrid;

        // Higher k compresses rank differences
        assert!(diff_low > diff_high);
    }

    #[test]
    fn bm25_weight_zero_excludes_bm25_contribution() {
        let rrf = Rrf::new(0.0, 1.0, 60);
        let bm25 = items(&["a"], &[1.0]);
        let result = rrf.fuse(&bm25, &[]);
        // Weight is 0 so no contribution despite being in the list
        assert!((result["a"].hybrid).abs() < 1e-10);
    }

    #[test]
    fn equal_weights_sum_to_symmetric_scores() {
        let rrf = Rrf::new(0.5, 0.5, 60);
        let bm25 = items(&["a"], &[1.0]);
        let vec = items(&["a"], &[1.0]);
        let result = rrf.fuse(&bm25, &vec);
        // Both are rank 1 with weight 0.5: score = 2 * (0.5 / 61) ≈ 0.01639
        let expected = 2.0 * 0.5 / 61.0;
        assert!((result["a"].hybrid - expected).abs() < 1e-10);
    }
}
