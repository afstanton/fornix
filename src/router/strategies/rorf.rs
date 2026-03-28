//! RoRF (Routing on Random Forests) strategy.
//!
//! A trained Random Forest classifier maps query embeddings to a binary
//! routing decision: `model_a` (stronger/more expensive) or `model_b`
//! (weaker/cheaper). The routing threshold controls the fraction of calls
//! sent to model_a — lower threshold = more calls to model_a.
//!
//! Follows the architecture from NotDiamond's RoRF paper:
//! - Binary labels: 0 = model_a preferred, 1 = model_b preferred
//! - `predict_proba` returns P(model_a) averaged across all trees
//! - Route to model_a when P(model_a) ≥ threshold
//!
//! The forest is serialisable (JSON via serde) so a trained model can be
//! saved to disk and loaded without re-training.

use crate::router::{
    error::{Error, Result},
    forest::{self, RandomForest, ForestParams},
    strategies::RoutingStrategy,
    types::{ModelInfo, RoutingDecision},
};

/// A trained RoRF router.
pub struct RoRFStrategy {
    forest: RandomForest,
    /// P(model_a) threshold. Queries with P(model_a) ≥ threshold → model_a.
    threshold: f32,
    model_a: String,
    provider_a: String,
    model_b: String,
    provider_b: String,
}

impl RoRFStrategy {
    /// Construct from a pre-trained forest.
    pub fn new(
        forest: RandomForest,
        threshold: f32,
        model_a: impl Into<String>,
        provider_a: impl Into<String>,
        model_b: impl Into<String>,
        provider_b: impl Into<String>,
    ) -> Self {
        assert!(
            (0.0..=1.0).contains(&threshold),
            "threshold must be in [0, 1]"
        );
        Self {
            forest,
            threshold,
            model_a: model_a.into(),
            provider_a: provider_a.into(),
            model_b: model_b.into(),
            provider_b: provider_b.into(),
        }
    }

    /// Train a new RoRF router from labelled embedding data.
    ///
    /// `features` — one embedding vector per training sample.
    /// `labels` — `0` if model_a was preferred, `1` if model_b was preferred.
    #[allow(clippy::too_many_arguments)]
    pub fn train(
        features: &[Vec<f32>],
        labels: &[u8],
        threshold: f32,
        model_a: impl Into<String>,
        provider_a: impl Into<String>,
        model_b: impl Into<String>,
        provider_b: impl Into<String>,
        params: ForestParams,
    ) -> Result<Self> {
        let trained = forest::train(features, labels, &params)?;
        Ok(Self::new(
            trained,
            threshold,
            model_a,
            provider_a,
            model_b,
            provider_b,
        ))
    }

    /// Confidence = how far P(model_a) is from the threshold, scaled to [0, 1].
    fn confidence(&self, prob_a: f32) -> f32 {
        ((prob_a - self.threshold).abs() * 2.0).clamp(0.0, 1.0)
    }

    /// Return the trained forest (for serialisation / inspection).
    pub fn forest(&self) -> &RandomForest {
        &self.forest
    }

    /// The routing threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}

impl RoutingStrategy for RoRFStrategy {
    fn name(&self) -> &'static str {
        "rorf"
    }

    fn route(
        &self,
        _content: &str,
        embedding: Option<&[f32]>,
        _models: &[ModelInfo],
    ) -> Result<RoutingDecision> {
        let emb = embedding.ok_or_else(|| {
            Error::config("RoRF strategy requires a query embedding")
        })?;

        let prob_a = self.forest.predict_proba(emb)?;
        let prob_b = 1.0 - prob_a;

        let (model, provider) = if prob_a >= self.threshold {
            (&self.model_a, &self.provider_a)
        } else {
            (&self.model_b, &self.provider_b)
        };

        Ok(RoutingDecision::new(model, provider)
            .with_reasoning(format!(
                "RoRF routing (P(model_a)={:.3}, threshold={:.3})",
                prob_a, self.threshold
            ))
            .with_confidence(self.confidence(prob_a))
            .with_meta("prob_model_a", prob_a)
            .with_meta("prob_model_b", prob_b)
            .with_meta("threshold", self.threshold))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::forest::{ForestParams, TreeParams};

    fn simple_data() -> (Vec<Vec<f32>>, Vec<u8>) {
        // Feature 0 < 0.5 → prefer model_a (label 0)
        // Feature 0 ≥ 0.5 → prefer model_b (label 1)
        let f: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![i as f32 / 20.0])
            .collect();
        let l: Vec<u8> = (0..20).map(|i| if i < 10 { 0 } else { 1 }).collect();
        (f, l)
    }

    fn trained_strategy(threshold: f32) -> RoRFStrategy {
        let (f, l) = simple_data();
        let params = ForestParams {
            n_estimators: 10,
            tree: TreeParams { max_depth: Some(5), ..Default::default() },
            ..Default::default()
        };
        RoRFStrategy::train(
            &f, &l,
            threshold,
            "strong-model", "provider-a",
            "weak-model", "provider-b",
            params,
        ).unwrap()
    }

    #[test]
    fn routes_simple_region_to_model_a() {
        let s = trained_strategy(0.5);
        let d = s.route("q", Some(&[0.1]), &[]).unwrap();
        assert_eq!(d.model, "strong-model");
    }

    #[test]
    fn routes_complex_region_to_model_b() {
        let s = trained_strategy(0.5);
        let d = s.route("q", Some(&[0.9]), &[]).unwrap();
        assert_eq!(d.model, "weak-model");
    }

    #[test]
    fn no_embedding_returns_error() {
        let s = trained_strategy(0.5);
        let err = s.route("q", None, &[]).unwrap_err();
        assert!(matches!(err, Error::Configuration(_)));
    }

    #[test]
    fn wrong_dimension_returns_error() {
        let s = trained_strategy(0.5);
        // Forest trained on 1-feature data; 2 features should fail
        let err = s.route("q", Some(&[0.5, 0.5]), &[]).unwrap_err();
        assert!(matches!(err, Error::Forest(_)));
    }

    #[test]
    fn confidence_is_set() {
        let s = trained_strategy(0.5);
        let d = s.route("q", Some(&[0.1]), &[]).unwrap();
        assert!(d.confidence.is_some());
    }

    #[test]
    fn metadata_contains_probabilities() {
        let s = trained_strategy(0.5);
        let d = s.route("q", Some(&[0.1]), &[]).unwrap();
        assert!(d.metadata.contains_key("prob_model_a"));
        assert!(d.metadata.contains_key("prob_model_b"));
    }

    #[test]
    fn high_threshold_routes_to_model_b_more() {
        // A query in the model_b region should still route to model_b
        // even when the threshold is very strict.
        let s = trained_strategy(0.99);
        let d = s.route("q", Some(&[0.9]), &[]).unwrap();
        assert_eq!(d.model, "weak-model");
    }

    #[test]
    fn forest_accessor() {
        let s = trained_strategy(0.5);
        assert_eq!(s.forest().n_features, 1);
    }

    #[test]
    fn name_is_rorf() {
        let s = trained_strategy(0.5);
        assert_eq!(s.name(), "rorf");
    }
}
