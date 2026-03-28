//! Linear weighted fusion.
//!
//! Combines normalised BM25 and vector scores by weighted sum:
//!
//! ```text
//! score(d) = bm25_weight * normalise(bm25_score(d))
//!           + vector_weight * normalise(vector_score(d))
//! ```
//!
//! Documents that appear in only one list are scored with 0.0 for the
//! missing component.

use std::collections::HashMap;

use crate::hybrid::{
    config::NormalisationMethod,
    fusion::{FusedScore, FusionStrategy, ScoredItem},
    normalizer,
};

/// Linear fusion strategy.
#[derive(Debug, Clone)]
pub struct Linear {
    /// BM25 list weight.
    pub bm25_weight: f32,
    /// Vector list weight.
    pub vector_weight: f32,
    /// Score normalisation applied before combining.
    pub normalisation: NormalisationMethod,
}

impl Linear {
    pub fn new(bm25_weight: f32, vector_weight: f32, normalisation: NormalisationMethod) -> Self {
        Self { bm25_weight, vector_weight, normalisation }
    }
}

impl Default for Linear {
    fn default() -> Self {
        Self::new(0.5, 0.5, NormalisationMethod::MinMax)
    }
}

impl FusionStrategy for Linear {
    fn fuse(
        &self,
        bm25_items: &[ScoredItem],
        vector_items: &[ScoredItem],
    ) -> HashMap<String, FusedScore> {
        // Convert to (id, score) pairs for normalisation
        let bm25_pairs: Vec<(String, f32)> = bm25_items
            .iter()
            .map(|item| (item.id.clone(), item.score))
            .collect();
        let vector_pairs: Vec<(String, f32)> = vector_items
            .iter()
            .map(|item| (item.id.clone(), item.score))
            .collect();

        let norm_bm25 = self.normalise(&bm25_pairs);
        let norm_vector = self.normalise(&vector_pairs);

        // Union of all ids
        let all_ids: std::collections::HashSet<String> = bm25_pairs
            .iter()
            .map(|(id, _)| id.clone())
            .chain(vector_pairs.iter().map(|(id, _)| id.clone()))
            .collect();

        all_ids
            .into_iter()
            .map(|id| {
                let b = norm_bm25.get(&id).copied().unwrap_or(0.0);
                let v = norm_vector.get(&id).copied().unwrap_or(0.0);
                let hybrid = self.bm25_weight * b + self.vector_weight * v;
                let fused = FusedScore {
                    hybrid,
                    bm25: bm25_pairs.iter().find(|(bid, _)| bid == &id).map(|(_, s)| *s),
                    vector: vector_pairs.iter().find(|(vid, _)| vid == &id).map(|(_, s)| *s),
                };
                (id, fused)
            })
            .collect()
    }
}

impl Linear {
    fn normalise(&self, pairs: &[(String, f32)]) -> normalizer::NormalisedScores {
        match self.normalisation {
            NormalisationMethod::MinMax => normalizer::min_max(pairs),
            NormalisationMethod::ZScore => normalizer::z_score(pairs),
            NormalisationMethod::None => normalizer::none(pairs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items(ids: &[&str], scores: &[f32]) -> Vec<ScoredItem> {
        ids.iter()
            .zip(scores.iter())
            .map(|(id, s)| ScoredItem::new(*id, *s))
            .collect()
    }

    // --- basic correctness ---

    #[test]
    fn empty_inputs_produce_empty_output() {
        let linear = Linear::default();
        assert!(linear.fuse(&[], &[]).is_empty());
    }

    #[test]
    fn bm25_only_with_no_weight_is_zero() {
        let linear = Linear::new(0.0, 1.0, NormalisationMethod::None);
        let bm25 = items(&["a"], &[1.0]);
        let result = linear.fuse(&bm25, &[]);
        assert!((result["a"].hybrid).abs() < 1e-6);
    }

    #[test]
    fn vector_only_with_no_weight_is_zero() {
        let linear = Linear::new(1.0, 0.0, NormalisationMethod::None);
        let vec = items(&["a"], &[1.0]);
        let result = linear.fuse(&[], &vec);
        assert!((result["a"].hybrid).abs() < 1e-6);
    }

    #[test]
    fn document_in_both_lists_combines_scores() {
        let linear = Linear::new(0.5, 0.5, NormalisationMethod::None);
        let bm25 = items(&["a"], &[0.8]);
        let vec = items(&["a"], &[0.6]);
        let result = linear.fuse(&bm25, &vec);
        // 0.5 * 0.8 + 0.5 * 0.6 = 0.7
        assert!((result["a"].hybrid - 0.7).abs() < 1e-5);
    }

    #[test]
    fn document_only_in_bm25_gets_zero_vector_contribution() {
        let linear = Linear::new(0.5, 0.5, NormalisationMethod::None);
        let bm25 = items(&["a"], &[1.0]);
        let vec = items(&["b"], &[1.0]);
        let result = linear.fuse(&bm25, &vec);
        // "a" has vector = 0: score = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        assert!((result["a"].hybrid - 0.5).abs() < 1e-5);
    }

    // --- normalisation ---

    #[test]
    fn min_max_normalisation_maps_to_unit_range() {
        let linear = Linear::new(1.0, 0.0, NormalisationMethod::MinMax);
        let bm25 = items(&["lo", "hi"], &[0.0, 10.0]);
        let result = linear.fuse(&bm25, &[]);
        // After min-max: lo → 0, hi → 1 × weight 1.0
        assert!((result["lo"].hybrid).abs() < 1e-5);
        assert!((result["hi"].hybrid - 1.0).abs() < 1e-5);
    }

    #[test]
    fn z_score_normalisation_centres_at_zero() {
        let linear = Linear::new(1.0, 0.0, NormalisationMethod::ZScore);
        let bm25 = items(&["a", "b", "c"], &[1.0, 2.0, 3.0]);
        let result = linear.fuse(&bm25, &[]);
        // Mean = 2.0; b (score=2.0) should have hybrid ≈ 0.0 × weight
        assert!(result["b"].hybrid.abs() < 1e-4);
    }

    #[test]
    fn no_normalisation_uses_raw_scores() {
        let linear = Linear::new(1.0, 0.0, NormalisationMethod::None);
        let bm25 = items(&["a"], &[3.5]);
        let result = linear.fuse(&bm25, &[]);
        assert!((result["a"].hybrid - 3.5).abs() < 1e-5);
    }

    // --- score passthrough ---

    #[test]
    fn raw_bm25_score_preserved_in_fused_score() {
        let linear = Linear::new(0.5, 0.5, NormalisationMethod::None);
        let bm25 = items(&["a"], &[7.0]);
        let result = linear.fuse(&bm25, &[]);
        assert_eq!(result["a"].bm25, Some(7.0));
        assert!(result["a"].vector.is_none());
    }

    #[test]
    fn raw_vector_score_preserved_in_fused_score() {
        let linear = Linear::new(0.5, 0.5, NormalisationMethod::None);
        let vec = items(&["a"], &[0.9]);
        let result = linear.fuse(&[], &vec);
        assert_eq!(result["a"].vector, Some(0.9));
        assert!(result["a"].bm25.is_none());
    }

    // --- union behaviour ---

    #[test]
    fn union_of_ids_is_present_in_output() {
        let linear = Linear::default();
        let bm25 = items(&["a", "b"], &[1.0, 0.5]);
        let vec = items(&["b", "c"], &[0.9, 0.3]);
        let result = linear.fuse(&bm25, &vec);
        assert!(result.contains_key("a"));
        assert!(result.contains_key("b"));
        assert!(result.contains_key("c"));
    }
}
