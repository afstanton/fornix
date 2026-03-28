//! Vector space analysis utilities.
//!
//! Pure mathematical functions operating on embedding vectors and
//! search result sets. No I/O, no adapters — safe to call from any context.

use crate::vector::result::VectorResult;

const EPSILON: f32 = 1e-9;

// ============================================================================
// Cosine geometry
// ============================================================================

/// Compute the L2 norm (magnitude) of a vector.
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Normalise a vector to unit length.
///
/// Returns a zero vector if the input has near-zero magnitude.
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm(v);
    if norm < EPSILON {
        return vec![0.0; v.len()];
    }
    v.iter().map(|x| x / norm).collect()
}

/// Cosine similarity between two vectors in `[-1.0, 1.0]`.
///
/// Returns `0.0` if either vector has near-zero magnitude.
/// Returns an error if the vectors have different lengths.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, String> {
    if a.len() != b.len() {
        return Err(format!(
            "dimension mismatch: {} vs {}",
            a.len(),
            b.len()
        ));
    }
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);
    if norm_a < EPSILON || norm_b < EPSILON {
        return Ok(0.0);
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    Ok((dot / (norm_a * norm_b)).clamp(-1.0, 1.0))
}

/// Cosine distance between two vectors: `1.0 - cosine_similarity(a, b)`.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> Result<f32, String> {
    Ok(1.0 - cosine_similarity(a, b)?)
}

/// Centroid (normalised mean) of a set of vectors.
///
/// All vectors must have the same dimension.
/// Returns an error if the slice is empty or dimensions differ.
pub fn centroid(vectors: &[Vec<f32>]) -> Result<Vec<f32>, String> {
    if vectors.is_empty() {
        return Err("cannot compute centroid of an empty set".to_string());
    }
    let dim = vectors[0].len();
    let mut sum = vec![0.0_f32; dim];
    for v in vectors {
        if v.len() != dim {
            return Err(format!(
                "dimension mismatch in centroid: expected {}, got {}",
                dim,
                v.len()
            ));
        }
        for (i, x) in v.iter().enumerate() {
            sum[i] += x;
        }
    }
    let n = vectors.len() as f32;
    let mean: Vec<f32> = sum.iter().map(|x| x / n).collect();
    Ok(normalize(&mean))
}

/// Mean cosine distance of each vector from the set centroid.
///
/// A measure of how spread out a cluster of embeddings is.
/// Returns `0.0` for a single-vector input. Returns an error on dimension mismatch.
pub fn embedding_variance(vectors: &[Vec<f32>]) -> Result<f32, String> {
    if vectors.len() < 2 {
        return Ok(0.0);
    }
    let c = centroid(vectors)?;
    let total: f32 = vectors
        .iter()
        .map(|v| cosine_distance(v, &c))
        .collect::<Result<Vec<f32>, _>>()?
        .iter()
        .sum();
    Ok(total / vectors.len() as f32)
}

// ============================================================================
// Information-theoretic measures
// ============================================================================

/// Surprisal of a single similarity score: `-ln(similarity)`.
///
/// Higher similarity → lower surprisal (the result is expected).
/// Near-zero similarities are clamped to `epsilon` to avoid `-inf`.
pub fn surprisal(similarity: f32, epsilon: f32) -> f32 {
    let s = similarity.clamp(epsilon, 1.0);
    -s.ln()
}

/// Mean surprisal across a slice of search results.
///
/// Returns `0.0` for an empty slice.
pub fn mean_surprisal(results: &[VectorResult], epsilon: f32) -> f32 {
    if results.is_empty() {
        return 0.0;
    }
    let total: f32 = results
        .iter()
        .map(|r| surprisal(r.score(), epsilon))
        .sum();
    total / results.len() as f32
}

/// Shannon entropy of a probability distribution (nats, not bits).
///
/// Zero-or-negative probabilities are skipped to avoid log(0).
pub fn entropy(distribution: &[f32]) -> f32 {
    distribution
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

/// Shannon entropy of a result set, treating normalised similarity scores
/// as a probability distribution.
///
/// Returns `0.0` for an empty result set.
pub fn result_entropy(results: &[VectorResult]) -> f32 {
    if results.is_empty() {
        return 0.0;
    }
    let total: f32 = results.iter().map(|r| r.score()).sum();
    if total < EPSILON {
        return 0.0;
    }
    let probs: Vec<f32> = results.iter().map(|r| r.score() / total).collect();
    entropy(&probs)
}

/// Expected information gain from a retrieved citation set.
///
/// Combines region entropy with a citation density weight. The result is
/// in `[0.0, 1.0]`.
pub fn expected_information_gain(
    citation_count: usize,
    region_entropy: f32,
    baseline_density: f32,
) -> f32 {
    if citation_count == 0 {
        return 0.0;
    }
    let density = baseline_density.max(1.0);
    let citation_weight =
        ((1.0 + citation_count as f32).ln() / (1.0 + density).ln()).clamp(0.0, 2.0);
    (region_entropy * citation_weight).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::metadata::Metadata;
    use crate::vector::result::VectorResult;

    fn result(sim: f32) -> VectorResult {
        VectorResult::new("id", sim, Metadata::new(), None)
    }

    // --- l2_norm ---

    #[test]
    fn l2_norm_of_unit_vector() {
        assert!((l2_norm(&[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_norm_of_zero_vector() {
        assert!(l2_norm(&[0.0, 0.0]) < 1e-6);
    }

    #[test]
    fn l2_norm_pythagorean_triple() {
        // 3-4-5 triangle
        assert!((l2_norm(&[3.0, 4.0]) - 5.0).abs() < 1e-5);
    }

    // --- normalize ---

    #[test]
    fn normalize_unit_vector_unchanged() {
        let v = vec![1.0, 0.0, 0.0];
        let n = normalize(&v);
        assert!((l2_norm(&n) - 1.0).abs() < 1e-6);
        assert!((n[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_arbitrary_vector_has_unit_norm() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        assert!((l2_norm(&n) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_zero_vector_returns_zeros() {
        let n = normalize(&[0.0, 0.0, 0.0]);
        assert_eq!(n, vec![0.0, 0.0, 0.0]);
    }

    // --- cosine_similarity ---

    #[test]
    fn cosine_similarity_identical_vectors_is_one() {
        let v = vec![0.5, 0.5, 0.5];
        assert!((cosine_similarity(&v, &v).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors_is_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b).unwrap()).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_opposite_vectors_is_negative_one() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b).unwrap() + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_dimension_mismatch_is_error() {
        assert!(cosine_similarity(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn cosine_similarity_zero_vector_returns_zero() {
        assert!((cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]).unwrap()).abs() < 1e-6);
    }

    // --- cosine_distance ---

    #[test]
    fn cosine_distance_identical_is_zero() {
        let v = vec![1.0, 0.0];
        assert!(cosine_distance(&v, &v).unwrap() < 1e-6);
    }

    #[test]
    fn cosine_distance_orthogonal_is_one() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_distance(&a, &b).unwrap() - 1.0).abs() < 1e-6);
    }

    // --- centroid ---

    #[test]
    fn centroid_single_vector_is_normalised_self() {
        let v = vec![vec![3.0, 4.0]];
        let c = centroid(&v).unwrap();
        assert!((l2_norm(&c) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn centroid_symmetric_pair_is_normalised_mean() {
        // [1, 0] and [-1, 0] average to [0, 0], which normalises to zero
        let vecs = vec![vec![1.0, 0.0], vec![-1.0, 0.0]];
        let c = centroid(&vecs).unwrap();
        assert!(l2_norm(&c) < 1e-6);
    }

    #[test]
    fn centroid_empty_returns_error() {
        assert!(centroid(&[]).is_err());
    }

    #[test]
    fn centroid_dimension_mismatch_returns_error() {
        let vecs = vec![vec![1.0, 0.0], vec![1.0, 0.0, 0.0]];
        assert!(centroid(&vecs).is_err());
    }

    // --- embedding_variance ---

    #[test]
    fn variance_single_vector_is_zero() {
        assert!(embedding_variance(&[vec![1.0, 0.0]]).unwrap() < 1e-6);
    }

    #[test]
    fn variance_identical_vectors_is_zero() {
        let vecs = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        assert!(embedding_variance(&vecs).unwrap() < 1e-6);
    }

    #[test]
    fn variance_orthogonal_vectors_is_positive() {
        let vecs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!(embedding_variance(&vecs).unwrap() > 0.0);
    }

    // --- surprisal ---

    #[test]
    fn surprisal_of_one_is_zero() {
        assert!(surprisal(1.0, EPSILON).abs() < 1e-6);
    }

    #[test]
    fn surprisal_increases_as_similarity_decreases() {
        let high = surprisal(0.9, EPSILON);
        let low = surprisal(0.1, EPSILON);
        assert!(low > high);
    }

    #[test]
    fn surprisal_of_zero_is_clamped_to_epsilon() {
        // Should not panic or return -inf
        let s = surprisal(0.0, EPSILON);
        assert!(s.is_finite());
        assert!(s > 0.0);
    }

    // --- mean_surprisal ---

    #[test]
    fn mean_surprisal_empty_is_zero() {
        assert_eq!(mean_surprisal(&[], EPSILON), 0.0);
    }

    #[test]
    fn mean_surprisal_all_perfect_is_zero() {
        let results = vec![result(1.0), result(1.0)];
        assert!(mean_surprisal(&results, EPSILON).abs() < 1e-6);
    }

    #[test]
    fn mean_surprisal_lower_similarity_gives_higher_value() {
        let high = mean_surprisal(&[result(0.9)], EPSILON);
        let low = mean_surprisal(&[result(0.1)], EPSILON);
        assert!(low > high);
    }

    // --- entropy ---

    #[test]
    fn entropy_uniform_distribution() {
        // Uniform over 4 outcomes → entropy = ln(4) ≈ 1.386
        let d = vec![0.25, 0.25, 0.25, 0.25];
        let e = entropy(&d);
        assert!((e - (4.0_f32.ln())).abs() < 1e-5);
    }

    #[test]
    fn entropy_certain_outcome_is_zero() {
        let d = vec![1.0, 0.0, 0.0];
        assert!(entropy(&d).abs() < 1e-6);
    }

    #[test]
    fn entropy_zero_probabilities_are_skipped() {
        // Same as certain outcome
        let d = vec![0.0, 0.0, 1.0];
        assert!(entropy(&d).abs() < 1e-6);
    }

    // --- result_entropy ---

    #[test]
    fn result_entropy_empty_is_zero() {
        assert_eq!(result_entropy(&[]), 0.0);
    }

    #[test]
    fn result_entropy_single_result_is_zero() {
        assert!(result_entropy(&[result(0.8)]).abs() < 1e-6);
    }

    #[test]
    fn result_entropy_uniform_similarities_is_positive() {
        let results = vec![result(0.5), result(0.5), result(0.5)];
        assert!(result_entropy(&results) > 0.0);
    }

    #[test]
    fn result_entropy_all_zero_similarities_is_zero() {
        let results = vec![result(0.0), result(0.0)];
        assert_eq!(result_entropy(&results), 0.0);
    }

    // --- expected_information_gain ---

    #[test]
    fn eig_zero_citations_is_zero() {
        assert_eq!(expected_information_gain(0, 0.8, 1.0), 0.0);
    }

    #[test]
    fn eig_increases_with_citation_count() {
        let low = expected_information_gain(1, 0.5, 1.0);
        let high = expected_information_gain(10, 0.5, 1.0);
        assert!(high >= low);
    }

    #[test]
    fn eig_result_is_in_zero_one() {
        let v = expected_information_gain(5, 0.7, 1.0);
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn eig_zero_entropy_is_zero() {
        assert_eq!(expected_information_gain(5, 0.0, 1.0), 0.0);
    }
}
