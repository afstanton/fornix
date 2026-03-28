//! Chain confidence computation for causal paths.
//!
//! Computes a composite confidence score for a causal path by combining
//! per-hop and per-entity confidence values with an exponential decay factor
//! that penalises longer chains.

const EPSILON: f32 = 1e-9;

/// Compute chain confidence for a causal path.
///
/// # Parameters
/// - `hop_confidences` — confidence of each relation (edge) in the path
/// - `entity_confidences` — confidence of entities along the path.
///   Length must equal `hop_confidences.len()` or `hop_confidences.len() + 1`.
/// - `decay` — exponential decay applied per hop (default: 0.9)
///
/// Returns a value in `[0.0, 1.0]`, or `0.0` if `hop_confidences` is empty.
/// Returns an error if the entity/hop count constraint is violated.
pub fn chain_confidence(
    hop_confidences: &[f32],
    entity_confidences: &[f32],
    decay: f32,
) -> Result<f32, String> {
    if hop_confidences.is_empty() {
        return Ok(0.0);
    }

    let hops = hop_confidences.len();
    let entities = entity_confidences.len();

    if entities != hops && entities != hops + 1 {
        return Err(format!(
            "entity_confidences length ({}) must equal hop_confidences length ({}) or hop_confidences length + 1",
            entities, hops
        ));
    }

    let hop_min = hop_confidences
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min)
        .clamp(0.0, 1.0);

    let entity_min = entity_confidences
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min)
        .clamp(0.0, 1.0);

    let ceiling = hop_min.min(entity_min);

    // Product of hop confidences with exponential decay: ∏ c_i * decay^i
    let raw_product: f32 = hop_confidences
        .iter()
        .enumerate()
        .fold(1.0_f32, |acc, (i, &c)| {
            acc * c.clamp(0.0, 1.0) * decay.powi(i as i32)
        });

    Ok(raw_product.min(ceiling).clamp(0.0, 1.0))
}

/// Approximate chain confidence without individual hop values.
///
/// Uses the mean confidence raised to the power of `num_hops`, with the
/// same decay schedule as `chain_confidence`.
pub fn approximate_chain_confidence(
    mean_confidence: f32,
    num_hops: usize,
    decay: f32,
) -> f32 {
    if num_hops == 0 {
        return 0.0;
    }
    let c = mean_confidence.clamp(0.0, 1.0);
    let exponent = (num_hops * (num_hops - 1)) as f32 / 2.0;
    (c.powi(num_hops as i32) * decay.powf(exponent)).clamp(0.0, 1.0)
}

/// Compute the geometric mean of causal_strength values.
///
/// Strengths missing from edge properties default to 1.0.
/// Returns 1.0 for an empty edge list.
pub fn chain_strength(strengths: &[f32]) -> f32 {
    if strengths.is_empty() {
        return 1.0;
    }
    let log_sum: f32 = strengths
        .iter()
        .map(|&s| s.max(EPSILON).ln())
        .sum();
    (log_sum / strengths.len() as f32).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- chain_confidence ---

    #[test]
    fn empty_hops_returns_zero() {
        assert_eq!(chain_confidence(&[], &[], 0.9).unwrap(), 0.0);
    }

    #[test]
    fn entity_count_equals_hops_is_valid() {
        let result = chain_confidence(&[0.9, 0.8], &[0.9, 0.8], 0.9);
        assert!(result.is_ok());
    }

    #[test]
    fn entity_count_equals_hops_plus_one_is_valid() {
        let result = chain_confidence(&[0.9, 0.8], &[0.9, 0.8, 0.7], 0.9);
        assert!(result.is_ok());
    }

    #[test]
    fn invalid_entity_count_returns_error() {
        let result = chain_confidence(&[0.9], &[0.9, 0.8, 0.7], 0.9);
        assert!(result.is_err());
    }

    #[test]
    fn single_hop_perfect_confidence_returns_one() {
        let result = chain_confidence(&[1.0], &[1.0], 0.9).unwrap();
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn single_hop_low_confidence_is_bounded() {
        let result = chain_confidence(&[0.3], &[1.0], 0.9).unwrap();
        assert!((result - 0.3).abs() < 1e-5);
    }

    #[test]
    fn ceiling_is_min_of_hops_and_entities() {
        // hop min = 0.6, entity min = 0.4 → ceiling = 0.4
        let result = chain_confidence(&[0.9, 0.6], &[1.0, 0.4, 1.0], 0.9).unwrap();
        assert!(result <= 0.4 + 1e-5);
    }

    #[test]
    fn longer_chain_has_lower_confidence() {
        let short = chain_confidence(&[0.9], &[0.9, 0.9], 0.9).unwrap();
        let long = chain_confidence(&[0.9, 0.9, 0.9], &[0.9, 0.9, 0.9, 0.9], 0.9).unwrap();
        assert!(short > long);
    }

    #[test]
    fn result_is_in_zero_one_range() {
        let result = chain_confidence(&[0.7, 0.8, 0.6], &[0.9, 0.7, 0.8, 0.5], 0.9).unwrap();
        assert!((0.0..=1.0).contains(&result));
    }

    #[test]
    fn zero_hop_confidence_makes_result_zero() {
        let result = chain_confidence(&[0.0, 0.9], &[0.9, 0.0], 0.9).unwrap();
        assert!(result < 1e-5);
    }

    // --- approximate_chain_confidence ---

    #[test]
    fn approx_zero_hops_returns_zero() {
        assert_eq!(approximate_chain_confidence(0.9, 0, 0.9), 0.0);
    }

    #[test]
    fn approx_single_hop_returns_mean_confidence() {
        let result = approximate_chain_confidence(0.8, 1, 0.9);
        assert!((result - 0.8).abs() < 1e-5);
    }

    #[test]
    fn approx_decreases_with_more_hops() {
        let one = approximate_chain_confidence(0.8, 1, 0.9);
        let three = approximate_chain_confidence(0.8, 3, 0.9);
        assert!(one > three);
    }

    #[test]
    fn approx_result_in_zero_one() {
        for hops in 1..=5 {
            let r = approximate_chain_confidence(0.7, hops, 0.9);
            assert!((0.0..=1.0).contains(&r));
        }
    }

    // --- chain_strength ---

    #[test]
    fn empty_strengths_returns_one() {
        assert!((chain_strength(&[]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn single_strength_returns_itself() {
        assert!((chain_strength(&[0.7]) - 0.7).abs() < 1e-5);
    }

    #[test]
    fn geometric_mean_of_equal_values() {
        assert!((chain_strength(&[0.5, 0.5]) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn chain_strength_is_less_than_max_and_greater_than_min() {
        let strengths = [0.6, 0.9, 0.7];
        let result = chain_strength(&strengths);
        let min = strengths.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = strengths.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(result >= min - 1e-5);
        assert!(result <= max + 1e-5);
    }

    #[test]
    fn chain_strength_of_all_ones_is_one() {
        assert!((chain_strength(&[1.0, 1.0, 1.0]) - 1.0).abs() < 1e-5);
    }
}
