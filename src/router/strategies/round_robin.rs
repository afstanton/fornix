//! Round-robin routing strategy.
//!
//! Cycles through the available model pool in order, distributing load evenly.
//! Thread-safe via `AtomicUsize`.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::router::{
    error::{Error, Result},
    strategies::RoutingStrategy,
    types::{ModelInfo, RoutingDecision},
};

/// Routes requests in round-robin order across the available model pool.
pub struct RoundRobin {
    counter: AtomicUsize,
}

impl RoundRobin {
    pub fn new() -> Self {
        Self { counter: AtomicUsize::new(0) }
    }

    /// Start the counter at a specific position (useful for testing).
    pub fn starting_at(index: usize) -> Self {
        Self { counter: AtomicUsize::new(index) }
    }
}

impl Default for RoundRobin {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingStrategy for RoundRobin {
    fn name(&self) -> &'static str {
        "round_robin"
    }

    fn route(
        &self,
        _content: &str,
        _embedding: Option<&[f32]>,
        models: &[ModelInfo],
    ) -> Result<RoutingDecision> {
        if models.is_empty() {
            return Err(Error::no_models("round-robin requires at least one model"));
        }

        // Fetch-and-increment; wraps naturally via modulo — no overflow risk
        // because usize wrapping is defined.
        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % models.len();
        let chosen = &models[idx];

        Ok(RoutingDecision::new(&chosen.name, &chosen.provider)
            .with_reasoning(format!("Round-robin (slot {})", idx)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::types::ModelInfo;

    fn models() -> Vec<ModelInfo> {
        vec![
            ModelInfo::new("model-a", "provider-a"),
            ModelInfo::new("model-b", "provider-b"),
            ModelInfo::new("model-c", "provider-c"),
        ]
    }

    #[test]
    fn cycles_through_all_models() {
        let rr = RoundRobin::new();
        let ms = models();
        let decisions: Vec<String> = (0..3)
            .map(|_| rr.route("q", None, &ms).unwrap().model.clone())
            .collect();
        assert_eq!(decisions[0], "model-a");
        assert_eq!(decisions[1], "model-b");
        assert_eq!(decisions[2], "model-c");
    }

    #[test]
    fn wraps_around_after_full_cycle() {
        let rr = RoundRobin::new();
        let ms = models();
        for _ in 0..3 {
            rr.route("q", None, &ms).unwrap();
        }
        let d = rr.route("q", None, &ms).unwrap();
        assert_eq!(d.model, "model-a");
    }

    #[test]
    fn starting_at_offset() {
        let rr = RoundRobin::starting_at(1);
        let ms = models();
        let d = rr.route("q", None, &ms).unwrap();
        assert_eq!(d.model, "model-b");
    }

    #[test]
    fn empty_pool_returns_error() {
        let err = RoundRobin::new().route("q", None, &[]).unwrap_err();
        assert!(matches!(err, Error::NoModels(_)));
    }

    #[test]
    fn single_model_always_chosen() {
        let rr = RoundRobin::new();
        let ms = vec![ModelInfo::new("solo", "p")];
        for _ in 0..5 {
            let d = rr.route("q", None, &ms).unwrap();
            assert_eq!(d.model, "solo");
        }
    }

    #[test]
    fn reasoning_contains_slot_index() {
        let rr = RoundRobin::new();
        let ms = models();
        let d = rr.route("q", None, &ms).unwrap();
        assert!(d.reasoning.as_deref().unwrap().contains("slot 0"));
    }

    #[test]
    fn name_is_round_robin() {
        assert_eq!(RoundRobin::new().name(), "round_robin");
    }
}
