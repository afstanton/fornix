//! Embedding-threshold routing strategy.
//!
//! Computes a "complexity score" for the query by comparing its embedding
//! to cached centroids of complex vs simple example queries, then routes
//! above-threshold queries to `model_a` (stronger) and below-threshold to
//! `model_b` (weaker).
//!
//! When no embedding provider is available, falls back to a lightweight
//! heuristic based on token count, vocabulary richness, and the presence
//! of complexity-signalling terms.

use crate::router::{
    error::Result,
    strategies::RoutingStrategy,
    types::{ModelInfo, RoutingDecision},
};

/// Terms that correlate with prompt complexity.
const COMPLEXITY_TERMS: &[&str] = &[
    "compare", "analyze", "explain", "evaluate", "architecture", "tradeoff",
    "ambiguity", "causality", "derive", "implement", "refactor", "optimize",
    "critique", "synthesize",
];

/// Cosine similarity between two equal-length vectors. Returns 0.0 on empty or
/// mismatched inputs.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < f32::EPSILON || nb < f32::EPSILON {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

/// Compute the centroid (element-wise mean) of a set of embedding vectors.
fn centroid(embeddings: &[Vec<f32>]) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }
    let dim = embeddings[0].len();
    let n = embeddings.len() as f32;
    let mut sum = vec![0.0_f32; dim];
    for v in embeddings {
        for (i, &x) in v.iter().enumerate() {
            sum[i] += x;
        }
    }
    sum.iter().map(|x| x / n).collect()
}

/// Configuration for the embedding-threshold strategy.
pub struct EmbeddingThresholdConfig {
    /// Score threshold; queries ≥ threshold go to `model_a`.
    pub threshold: f32,
    /// Stronger model (used for complex queries).
    pub model_a: String,
    pub provider_a: String,
    /// Weaker model (used for simple queries).
    pub model_b: String,
    pub provider_b: String,
    /// Centroid of complex example embeddings (pre-computed).
    pub complex_centroid: Vec<f32>,
    /// Centroid of simple example embeddings (pre-computed).
    pub simple_centroid: Vec<f32>,
}

impl EmbeddingThresholdConfig {
    /// Build from raw example embedding arrays.
    pub fn from_examples(
        threshold: f32,
        model_a: impl Into<String>,
        provider_a: impl Into<String>,
        model_b: impl Into<String>,
        provider_b: impl Into<String>,
        complex_examples: &[Vec<f32>],
        simple_examples: &[Vec<f32>],
    ) -> Self {
        Self {
            threshold,
            model_a: model_a.into(),
            provider_a: provider_a.into(),
            model_b: model_b.into(),
            provider_b: provider_b.into(),
            complex_centroid: centroid(complex_examples),
            simple_centroid: centroid(simple_examples),
        }
    }
}

/// Routes based on query complexity estimated from its embedding.
pub struct EmbeddingThreshold {
    config: EmbeddingThresholdConfig,
}

impl EmbeddingThreshold {
    pub fn new(config: EmbeddingThresholdConfig) -> Self {
        Self { config }
    }

    /// Compute a complexity score in [0, 1] from the query embedding.
    ///
    /// Score = cos_sim(q, complex_centroid) / (cos_sim(q, complex_centroid)
    ///         + cos_sim(q, simple_centroid))
    /// Returns 0.5 when centroids are empty or the denominator is zero.
    fn embedding_score(&self, embedding: &[f32]) -> f32 {
        if self.config.complex_centroid.is_empty() || self.config.simple_centroid.is_empty() {
            return 0.5;
        }
        let complex = cosine_similarity(embedding, &self.config.complex_centroid);
        let simple = cosine_similarity(embedding, &self.config.simple_centroid);
        let denom = complex + simple;
        if denom <= f32::EPSILON {
            return 0.5;
        }
        (complex / denom).clamp(0.0, 1.0)
    }

    /// Heuristic complexity score when no embedding is available.
    ///
    /// Combines four signals: token count, vocabulary richness,
    /// presence of long words, and matched complexity terms.
    fn heuristic_score(&self, content: &str) -> f32 {
        let lower = content.to_lowercase();
        let toks: Vec<&str> = lower
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|t| !t.is_empty())
            .collect();

        if toks.is_empty() {
            return 0.0;
        }

        let count = toks.len() as f32;
        let unique: std::collections::HashSet<&str> = toks.iter().cloned().collect();
        let unique_ratio = unique.len() as f32 / count;
        let long_ratio = toks.iter().filter(|t| t.len() >= 7).count() as f32 / count;
        let complexity_hit = toks
            .iter()
            .filter(|t| COMPLEXITY_TERMS.contains(t))
            .count() as f32
            / count;

        let token_signal = (count / 50.0).min(1.0); // saturates at 50 tokens
        (0.3 * token_signal + 0.3 * unique_ratio + 0.2 * long_ratio + 0.2 * complexity_hit)
            .clamp(0.0, 1.0)
    }

    /// Confidence = how far the score is from the threshold, scaled to [0, 1].
    fn confidence(&self, score: f32) -> f32 {
        ((score - self.config.threshold).abs() * 2.0).clamp(0.0, 1.0)
    }
}

impl RoutingStrategy for EmbeddingThreshold {
    fn name(&self) -> &'static str {
        "embedding_threshold"
    }

    fn route(
        &self,
        content: &str,
        embedding: Option<&[f32]>,
        _models: &[ModelInfo],
    ) -> Result<RoutingDecision> {
        let score = match embedding {
            Some(e) => self.embedding_score(e),
            None => self.heuristic_score(content),
        };

        let (model, provider) = if score >= self.config.threshold {
            (&self.config.model_a, &self.config.provider_a)
        } else {
            (&self.config.model_b, &self.config.provider_b)
        };

        Ok(RoutingDecision::new(model, provider)
            .with_reasoning(format!(
                "Embedding-threshold routing (score={:.3}, threshold={:.3})",
                score, self.config.threshold
            ))
            .with_confidence(self.confidence(score))
            .with_meta("complexity_score", score)
            .with_meta("threshold", self.config.threshold))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(threshold: f32) -> EmbeddingThresholdConfig {
        // Complex centroid points toward [1, 0]; simple toward [0, 1]
        EmbeddingThresholdConfig::from_examples(
            threshold,
            "strong-model", "provider-a",
            "weak-model", "provider-b",
            &[vec![1.0_f32, 0.0]],
            &[vec![0.0_f32, 1.0]],
        )
    }

    fn strat(threshold: f32) -> EmbeddingThreshold {
        EmbeddingThreshold::new(config(threshold))
    }

    // --- Embedding score ---

    #[test]
    fn complex_embedding_scores_high() {
        let s = strat(0.5);
        let score = s.embedding_score(&[1.0, 0.0]);
        assert!(score > 0.5, "score={}", score);
    }

    #[test]
    fn simple_embedding_scores_low() {
        let s = strat(0.5);
        let score = s.embedding_score(&[0.0, 1.0]);
        assert!(score < 0.5, "score={}", score);
    }

    #[test]
    fn no_centroids_returns_half() {
        let s = EmbeddingThreshold::new(EmbeddingThresholdConfig {
            threshold: 0.5,
            model_a: "a".into(), provider_a: "p".into(),
            model_b: "b".into(), provider_b: "p".into(),
            complex_centroid: Vec::new(),
            simple_centroid: Vec::new(),
        });
        assert!((s.embedding_score(&[1.0, 0.0]) - 0.5).abs() < 1e-6);
    }

    // --- Routing decisions ---

    #[test]
    fn complex_embedding_routes_to_model_a() {
        let d = strat(0.5)
            .route("q", Some(&[1.0, 0.0]), &[])
            .unwrap();
        assert_eq!(d.model, "strong-model");
    }

    #[test]
    fn simple_embedding_routes_to_model_b() {
        let d = strat(0.5)
            .route("q", Some(&[0.0, 1.0]), &[])
            .unwrap();
        assert_eq!(d.model, "weak-model");
    }

    #[test]
    fn high_threshold_always_routes_simple() {
        let d = strat(0.99)
            .route("q", Some(&[0.0, 1.0]), &[])
            .unwrap();
        // A clearly simple embedding should route to the weak model.
        assert_eq!(d.model, "weak-model");
    }

    #[test]
    fn decision_contains_complexity_score() {
        let d = strat(0.5)
            .route("q", Some(&[1.0, 0.0]), &[])
            .unwrap();
        assert!(d.metadata.contains_key("complexity_score"));
    }

    #[test]
    fn confidence_is_populated() {
        let d = strat(0.5)
            .route("q", Some(&[1.0, 0.0]), &[])
            .unwrap();
        assert!(d.confidence.is_some());
    }

    // --- Heuristic fallback ---

    #[test]
    fn simple_query_heuristic_low() {
        let s = strat(0.5);
        let score = s.heuristic_score("hi");
        assert!(score < 0.5, "score={}", score);
    }

    #[test]
    fn complex_query_heuristic_higher() {
        let s = strat(0.5);
        let simple = s.heuristic_score("hi");
        let complex = s.heuristic_score(
            "analyze and evaluate the architectural tradeoffs between \
             microservices and monolithic systems, considering scalability \
             and maintenance complexity",
        );
        assert!(complex > simple, "complex={} simple={}", complex, simple);
    }

    #[test]
    fn no_embedding_uses_heuristic() {
        // Providing None for embedding should not error
        let result = strat(0.5).route("hello world", None, &[]);
        assert!(result.is_ok());
    }

    // --- Geometry helpers ---

    #[test]
    fn cosine_similarity_identical() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn centroid_single_vector_is_itself() {
        let v = vec![vec![1.0_f32, 2.0, 3.0]];
        let c = centroid(&v);
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn centroid_two_symmetric_vectors_is_midpoint() {
        let vecs = vec![vec![0.0_f32, 1.0], vec![1.0, 0.0]];
        let c = centroid(&vecs);
        assert!((c[0] - 0.5).abs() < 1e-6);
        assert!((c[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn name_is_embedding_threshold() {
        assert_eq!(strat(0.5).name(), "embedding_threshold");
    }
}
