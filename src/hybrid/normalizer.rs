//! Score normalisation strategies for linear hybrid fusion.
//!
//! Normalisers map a collection of raw scores into a common range so they
//! can be combined by weight. They are only meaningful for linear fusion —
//! RRF operates on rank positions and ignores raw scores entirely.

/// A normalised score map: document id → normalised score in `[0.0, 1.0]`.
pub type NormalisedScores = std::collections::HashMap<String, f32>;

/// Normalise a set of (id, score) pairs into `[0, 1]` using min-max scaling.
///
/// All ids with equal scores map to 0.0.
pub fn min_max(scores: &[(String, f32)]) -> NormalisedScores {
    if scores.is_empty() {
        return NormalisedScores::new();
    }
    let values: Vec<f32> = scores.iter().map(|(_, s)| *s).collect();
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;

    scores
        .iter()
        .map(|(id, score)| {
            let norm = if range < f32::EPSILON {
                0.0
            } else {
                (score - min) / range
            };
            (id.clone(), norm)
        })
        .collect()
}

/// Normalise a set of (id, score) pairs using z-score standardisation.
///
/// All ids with zero standard deviation map to 0.0.
pub fn z_score(scores: &[(String, f32)]) -> NormalisedScores {
    if scores.is_empty() {
        return NormalisedScores::new();
    }
    let values: Vec<f32> = scores.iter().map(|(_, s)| *s).collect();
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();

    scores
        .iter()
        .map(|(id, score)| {
            let norm = if std < f32::EPSILON {
                0.0
            } else {
                (score - mean) / std
            };
            (id.clone(), norm)
        })
        .collect()
}

/// Pass scores through unchanged (no normalisation).
pub fn none(scores: &[(String, f32)]) -> NormalisedScores {
    scores.iter().map(|(id, s)| (id.clone(), *s)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(vals: &[(&str, f32)]) -> Vec<(String, f32)> {
        vals.iter().map(|(id, s)| (id.to_string(), *s)).collect()
    }

    // --- min_max ---

    #[test]
    fn min_max_empty_returns_empty() {
        assert!(min_max(&[]).is_empty());
    }

    #[test]
    fn min_max_single_item_maps_to_zero() {
        let result = min_max(&pairs(&[("a", 5.0)]));
        assert_eq!(result["a"], 0.0);
    }

    #[test]
    fn min_max_min_maps_to_zero_max_to_one() {
        let result = min_max(&pairs(&[("a", 0.0), ("b", 10.0)]));
        assert_eq!(result["a"], 0.0);
        assert_eq!(result["b"], 1.0);
    }

    #[test]
    fn min_max_midpoint_maps_to_half() {
        let result = min_max(&pairs(&[("lo", 0.0), ("mid", 5.0), ("hi", 10.0)]));
        assert!((result["mid"] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn min_max_equal_values_all_map_to_zero() {
        let result = min_max(&pairs(&[("a", 3.0), ("b", 3.0)]));
        assert_eq!(result["a"], 0.0);
        assert_eq!(result["b"], 0.0);
    }

    #[test]
    fn min_max_preserves_all_ids() {
        let p = pairs(&[("x", 1.0), ("y", 2.0), ("z", 3.0)]);
        let result = min_max(&p);
        assert_eq!(result.len(), 3);
    }

    // --- z_score ---

    #[test]
    fn z_score_empty_returns_empty() {
        assert!(z_score(&[]).is_empty());
    }

    #[test]
    fn z_score_zero_std_maps_all_to_zero() {
        let result = z_score(&pairs(&[("a", 5.0), ("b", 5.0)]));
        assert_eq!(result["a"], 0.0);
        assert_eq!(result["b"], 0.0);
    }

    #[test]
    fn z_score_mean_maps_to_zero() {
        // Mean of [1, 2, 3] = 2; z_score(2) = 0
        let result = z_score(&pairs(&[("a", 1.0), ("b", 2.0), ("c", 3.0)]));
        assert!(result["b"].abs() < 1e-5);
    }

    #[test]
    fn z_score_above_mean_is_positive() {
        let result = z_score(&pairs(&[("lo", 1.0), ("hi", 3.0)]));
        assert!(result["hi"] > 0.0);
        assert!(result["lo"] < 0.0);
    }

    #[test]
    fn z_score_preserves_all_ids() {
        let p = pairs(&[("x", 1.0), ("y", 2.0)]);
        assert_eq!(z_score(&p).len(), 2);
    }

    // --- none ---

    #[test]
    fn none_returns_scores_unchanged() {
        let p = pairs(&[("a", 3.7), ("b", 0.5)]);
        let result = none(&p);
        assert!((result["a"] - 3.7).abs() < 1e-6);
        assert!((result["b"] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn none_empty_returns_empty() {
        assert!(none(&[]).is_empty());
    }
}
