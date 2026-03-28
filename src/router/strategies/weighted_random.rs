//! Weighted random routing strategy.
//!
//! Each model in the pool carries a non-negative weight. Models are sampled
//! with probability proportional to their weight. Equivalent to the Ruby
//! `Random` strategy when weights are equal, but also supports load biasing.

use crate::router::{
    error::{Error, Result},
    strategies::RoutingStrategy,
    types::{ModelInfo, RoutingDecision},
};

/// A model paired with a routing weight.
#[derive(Debug, Clone)]
pub struct WeightedModel {
    pub model: ModelInfo,
    /// Non-negative weight (relative probability of being chosen).
    pub weight: f64,
}

impl WeightedModel {
    pub fn new(model: ModelInfo, weight: f64) -> Self {
        assert!(weight >= 0.0, "weight must be non-negative");
        Self { model, weight }
    }

    /// Uniform weight of 1.0.
    pub fn uniform(model: ModelInfo) -> Self {
        Self::new(model, 1.0)
    }
}

/// Samples from the model pool proportionally to configured weights.
///
/// Uses a deterministic seed when provided for reproducibility in tests;
/// in production pass `None` to use a time-based seed.
pub struct WeightedRandom {
    weighted: Vec<WeightedModel>,
    seed: Option<u64>,
}

impl WeightedRandom {
    pub fn new(weighted: Vec<WeightedModel>) -> Self {
        Self { weighted, seed: None }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    fn sample(&self, rng_value: f64) -> Option<&WeightedModel> {
        let total: f64 = self.weighted.iter().map(|w| w.weight).sum();
        if total <= 0.0 {
            return None;
        }
        let mut cumulative = 0.0;
        let target = rng_value * total;
        for wm in &self.weighted {
            cumulative += wm.weight;
            if cumulative >= target {
                return Some(wm);
            }
        }
        self.weighted.last()
    }

    /// LCG-based pseudo-random float in [0, 1).
    fn rand_f64(&self) -> f64 {
        let seed = self.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos() as u64
        });
        // Splitmix64 step
        let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z = z ^ (z >> 31);
        // Map to [0, 1)
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

impl RoutingStrategy for WeightedRandom {
    fn name(&self) -> &'static str {
        "weighted_random"
    }

    fn route(
        &self,
        _content: &str,
        _embedding: Option<&[f32]>,
        _models: &[ModelInfo],
    ) -> Result<RoutingDecision> {
        if self.weighted.is_empty() {
            return Err(Error::no_models("weighted_random requires at least one model"));
        }

        let r = self.rand_f64();
        let chosen = self.sample(r)
            .ok_or_else(|| Error::no_models("all model weights are zero"))?;

        Ok(RoutingDecision::new(&chosen.model.name, &chosen.model.provider)
            .with_reasoning("Weighted random selection"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::types::ModelInfo;

    fn mi(name: &str) -> ModelInfo {
        ModelInfo::new(name, "p")
    }

    #[test]
    fn single_model_always_chosen() {
        let strat = WeightedRandom::new(vec![WeightedModel::uniform(mi("solo"))]).with_seed(1);
        let d = strat.route("q", None, &[]).unwrap();
        assert_eq!(d.model, "solo");
    }

    #[test]
    fn zero_weight_model_never_chosen() {
        // Model-a has all the weight; model-b has none
        let strat = WeightedRandom::new(vec![
            WeightedModel::new(mi("model-a"), 1.0),
            WeightedModel::new(mi("model-b"), 0.0),
        ])
        .with_seed(42);
        let d = strat.route("q", None, &[]).unwrap();
        assert_eq!(d.model, "model-a");
    }

    #[test]
    fn sample_at_zero_picks_first() {
        let strat = WeightedRandom::new(vec![
            WeightedModel::new(mi("first"), 1.0),
            WeightedModel::new(mi("second"), 1.0),
        ]);
        let chosen = strat.sample(0.0).unwrap();
        assert_eq!(chosen.model.name, "first");
    }

    #[test]
    fn sample_at_one_picks_last() {
        let strat = WeightedRandom::new(vec![
            WeightedModel::new(mi("a"), 1.0),
            WeightedModel::new(mi("b"), 1.0),
        ]);
        // target = 1.0 * 2.0 = 2.0; cumulative only reaches 2.0 at last item
        let chosen = strat.sample(0.9999999).unwrap();
        assert_eq!(chosen.model.name, "b");
    }

    #[test]
    fn empty_pool_returns_error() {
        let err = WeightedRandom::new(vec![]).route("q", None, &[]).unwrap_err();
        assert!(matches!(err, Error::NoModels(_)));
    }

    #[test]
    fn reasoning_is_set() {
        let strat = WeightedRandom::new(vec![WeightedModel::uniform(mi("m"))]).with_seed(1);
        let d = strat.route("q", None, &[]).unwrap();
        assert!(d.reasoning.is_some());
    }

    #[test]
    fn name_is_weighted_random() {
        let strat = WeightedRandom::new(vec![WeightedModel::uniform(mi("m"))]);
        assert_eq!(strat.name(), "weighted_random");
    }

    #[test]
    fn distribution_roughly_proportional() {
        // Model-a has 3x the weight of model-b; expect ~75% of choices to be model-a
        // Use different seeds to simulate multiple calls
        let mut counts = std::collections::HashMap::new();
        for seed in 0..100u64 {
            let strat = WeightedRandom::new(vec![
                WeightedModel::new(mi("a"), 3.0),
                WeightedModel::new(mi("b"), 1.0),
            ])
            .with_seed(seed * 1000 + 7);
            let d = strat.route("q", None, &[]).unwrap();
            *counts.entry(d.model).or_insert(0) += 1;
        }
        let a_count = counts.get("a").copied().unwrap_or(0);
        // Should be roughly 75% ± 15%
        assert!(a_count >= 55 && a_count <= 95, "a_count={}", a_count);
    }
}
