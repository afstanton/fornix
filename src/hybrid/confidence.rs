//! Confidence scoring for hybrid search results.
//!
//! Assigns a confidence score in `[0.0, 1.0]` to each result after fusion,
//! based on three components:
//!
//! - **Percentile** (weight 0.6): relative position of this score within
//!   the full result set — higher-scoring results are more confident
//! - **Margin** (weight 0.3): normalised gap to the next result — a large
//!   gap suggests the result is clearly better than what follows
//! - **Agreement** (weight 0.1): whether both BM25 and vector sources
//!   returned this result (1.0) or only one did (0.5)

use crate::hybrid::result::{ConfidenceComponents, HybridResult};

/// Compute and attach confidence scores to a ranked result slice.
///
/// Results must already be sorted descending by `hybrid_score` before
/// calling this function.
pub fn apply_confidence(results: &mut [HybridResult]) {
    if results.is_empty() {
        return;
    }

    let scores: Vec<f32> = results.iter().map(|r| r.hybrid_score).collect();
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = scores.iter().cloned().fold(f32::INFINITY, f32::min);
    let range = (max - min).abs();
    let effective_range = if range < f32::EPSILON { 1.0 } else { range };

    let len = results.len();
    for i in 0..len {
        let score = scores[i];
        let next_score = if i + 1 < len { Some(scores[i + 1]) } else { None };

        let percentile = (score - min) / effective_range;
        let margin = match next_score {
            Some(next) => (score - next) / effective_range,
            None => percentile, // last result: margin equals its own percentile
        };
        let agreement = if results[i].has_both_sources() { 1.0 } else { 0.5 };

        let confidence = (0.6 * percentile + 0.3 * margin + 0.1 * agreement).clamp(0.0, 1.0);

        let components = ConfidenceComponents { percentile, margin, agreement };
        results[i].confidence_score = Some(confidence);
        results[i].confidence_components = Some(components);
        results[i].score_breakdown.fusion_path = format!(
            "{} [confidence={:.3}]",
            results[i].score_breakdown.fusion_path, confidence
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid::result::HybridResult;

    fn result(id: &str, hybrid: f32, bm25: Option<f32>, vec: Option<f32>) -> HybridResult {
        HybridResult::new(id, hybrid, bm25, vec, "rrf")
    }

    #[test]
    fn empty_slice_is_no_op() {
        let mut results: Vec<HybridResult> = vec![];
        apply_confidence(&mut results);
        assert!(results.is_empty());
    }

    #[test]
    fn confidence_is_set_on_all_results() {
        let mut results = vec![
            result("a", 0.9, Some(0.8), Some(0.9)),
            result("b", 0.6, Some(0.4), None),
            result("c", 0.3, None, Some(0.3)),
        ];
        apply_confidence(&mut results);
        assert!(results.iter().all(|r| r.confidence_score.is_some()));
    }

    #[test]
    fn confidence_is_in_zero_one_range() {
        let mut results = vec![
            result("a", 1.0, Some(1.0), Some(1.0)),
            result("b", 0.5, Some(0.5), None),
            result("c", 0.0, None, Some(0.0)),
        ];
        apply_confidence(&mut results);
        for r in &results {
            let c = r.confidence_score.unwrap();
            assert!((0.0..=1.0).contains(&c), "confidence {} out of range", c);
        }
    }

    #[test]
    fn top_result_has_highest_confidence() {
        let mut results = vec![
            result("top",    1.0, Some(1.0), Some(1.0)),
            result("middle", 0.5, Some(0.5), Some(0.5)),
            result("bottom", 0.1, Some(0.1), None),
        ];
        apply_confidence(&mut results);
        let top_c = results[0].confidence_score.unwrap();
        let bottom_c = results[2].confidence_score.unwrap();
        assert!(top_c >= bottom_c);
    }

    #[test]
    fn single_result_gets_confidence() {
        let mut results = vec![result("only", 0.7, Some(0.7), Some(0.7))];
        apply_confidence(&mut results);
        assert!(results[0].confidence_score.is_some());
        let c = results[0].confidence_score.unwrap();
        assert!((0.0..=1.0).contains(&c));
    }

    #[test]
    fn agreement_component_higher_for_both_sources() {
        let mut results = vec![
            result("both", 0.5, Some(0.5), Some(0.5)),
            result("one",  0.5, Some(0.5), None),
        ];
        apply_confidence(&mut results);
        // Same score, same position — only agreement differs
        let both_c = results[0].confidence_score.unwrap();
        let one_c = results[1].confidence_score.unwrap();
        // "both" has agreement=1.0, "one" has agreement=0.5 → both_c > one_c
        assert!(both_c > one_c);
    }

    #[test]
    fn confidence_components_are_populated() {
        let mut results = vec![result("a", 0.8, Some(0.8), Some(0.8))];
        apply_confidence(&mut results);
        assert!(results[0].confidence_components.is_some());
        let comp = results[0].confidence_components.as_ref().unwrap();
        assert!((0.0..=1.0).contains(&comp.percentile));
        assert!((0.0..=1.0).contains(&comp.margin));
        assert!(comp.agreement == 1.0 || comp.agreement == 0.5);
    }

    #[test]
    fn all_equal_scores_produce_valid_confidence() {
        // When all scores are equal, range = 0 → effective_range = 1
        let mut results = vec![
            result("a", 0.5, Some(0.5), Some(0.5)),
            result("b", 0.5, Some(0.5), Some(0.5)),
        ];
        apply_confidence(&mut results);
        for r in &results {
            let c = r.confidence_score.unwrap();
            assert!((0.0..=1.0).contains(&c));
        }
    }
}
