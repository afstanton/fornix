//! Vector search result types.

use crate::common::metadata::Metadata;

/// A similarity score in the range `[0.0, 1.0]`.
///
/// Enforces the valid range at construction time rather than relying on
/// callers to remember to clamp. The inner value is always finite.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Similarity(f32);

impl Similarity {
    /// Construct a similarity score, clamping the value to `[0.0, 1.0]`.
    /// Non-finite values are treated as `0.0`.
    pub fn new(value: f32) -> Self {
        if value.is_finite() {
            Self(value.clamp(0.0, 1.0))
        } else {
            Self(0.0)
        }
    }

    /// The raw `f32` value in `[0.0, 1.0]`.
    pub fn value(self) -> f32 {
        self.0
    }
}

impl From<f32> for Similarity {
    fn from(v: f32) -> Self {
        Self::new(v)
    }
}

impl From<f64> for Similarity {
    fn from(v: f64) -> Self {
        Self::new(v as f32)
    }
}

impl std::fmt::Display for Similarity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.6}", self.0)
    }
}

/// A stored vector record, as returned by list operations.
#[derive(Debug, Clone)]
pub struct VectorRecord {
    /// Stable record identifier.
    pub id: String,
    /// Arbitrary metadata attached to this record.
    pub metadata: Metadata,
    /// The vector itself, if the adapter was asked to include it.
    pub vector: Option<Vec<f32>>,
}

/// A single nearest-neighbour search result.
#[derive(Debug, Clone)]
pub struct VectorResult {
    /// Stable record identifier.
    pub id: String,
    /// Cosine similarity to the query vector, in `[0.0, 1.0]`.
    pub similarity: Similarity,
    /// Metadata attached to this record.
    pub metadata: Metadata,
    /// The stored vector, if the adapter was asked to include it.
    pub vector: Option<Vec<f32>>,
}

impl VectorResult {
    /// Construct a result, clamping `similarity` to `[0.0, 1.0]`.
    pub fn new(
        id: impl Into<String>,
        similarity: f32,
        metadata: Metadata,
        vector: Option<Vec<f32>>,
    ) -> Self {
        Self {
            id: id.into(),
            similarity: Similarity::new(similarity),
            metadata,
            vector,
        }
    }

    /// Convenience accessor for the raw similarity value.
    pub fn score(&self) -> f32 {
        self.similarity.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Similarity ---

    #[test]
    fn similarity_clamps_above_one() {
        assert_eq!(Similarity::new(1.5).value(), 1.0);
    }

    #[test]
    fn similarity_clamps_below_zero() {
        assert_eq!(Similarity::new(-0.5).value(), 0.0);
    }

    #[test]
    fn similarity_preserves_valid_value() {
        assert!((Similarity::new(0.75).value() - 0.75).abs() < 1e-6);
    }

    #[test]
    fn similarity_nan_becomes_zero() {
        assert_eq!(Similarity::new(f32::NAN).value(), 0.0);
    }

    #[test]
    fn similarity_infinity_becomes_zero() {
        assert_eq!(Similarity::new(f32::INFINITY).value(), 0.0);
    }

    #[test]
    fn similarity_zero_is_valid() {
        assert_eq!(Similarity::new(0.0).value(), 0.0);
    }

    #[test]
    fn similarity_one_is_valid() {
        assert_eq!(Similarity::new(1.0).value(), 1.0);
    }

    #[test]
    fn from_f32() {
        let s: Similarity = 0.5_f32.into();
        assert!((s.value() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn from_f64() {
        let s: Similarity = 0.9_f64.into();
        assert!((s.value() - 0.9_f32).abs() < 1e-5);
    }

    #[test]
    fn display_shows_six_decimal_places() {
        let s = Similarity::new(0.75);
        assert_eq!(s.to_string(), "0.750000");
    }

    // --- VectorResult ---

    #[test]
    fn vector_result_new_clamps_similarity() {
        let r = VectorResult::new("id", 2.0, Default::default(), None);
        assert_eq!(r.score(), 1.0);
    }

    #[test]
    fn vector_result_score_accessor() {
        let r = VectorResult::new("id", 0.85, Default::default(), None);
        assert!((r.score() - 0.85).abs() < 1e-6);
    }

    #[test]
    fn vector_result_without_vector() {
        let r = VectorResult::new("id", 0.5, Default::default(), None);
        assert!(r.vector.is_none());
    }

    #[test]
    fn vector_result_with_vector() {
        let v = vec![0.1, 0.2, 0.3];
        let r = VectorResult::new("id", 0.5, Default::default(), Some(v.clone()));
        assert_eq!(r.vector, Some(v));
    }

    #[test]
    fn vector_result_id_is_stored() {
        let r = VectorResult::new("my-record", 0.5, Default::default(), None);
        assert_eq!(r.id, "my-record");
    }

    // --- VectorRecord ---

    #[test]
    fn vector_record_fields() {
        let rec = VectorRecord {
            id: "rec-1".to_string(),
            metadata: Default::default(),
            vector: Some(vec![1.0, 0.0]),
        };
        assert_eq!(rec.id, "rec-1");
        assert!(rec.vector.is_some());
    }
}
