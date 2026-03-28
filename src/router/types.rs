//! Router domain types: ModelInfo, ProviderConfig, RoutingDecision.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// A model available for routing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier (e.g. `"claude-sonnet-4-20250514"`).
    pub name: String,
    /// Provider this model belongs to (e.g. `"anthropic"`).
    pub provider: String,
    /// Optional model type tag (e.g. `"embedding"`, `"chat"`).
    pub model_type: Option<String>,
    /// Arbitrary metadata for this model.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ModelInfo {
    pub fn new(name: impl Into<String>, provider: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            provider: provider.into(),
            model_type: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_type(mut self, t: impl Into<String>) -> Self {
        self.model_type = Some(t.into());
        self
    }

    pub fn with_meta(mut self, key: impl Into<String>, val: impl Into<serde_json::Value>) -> Self {
        self.metadata.insert(key.into(), val.into());
        self
    }
}

/// Configuration for a single upstream LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Human-readable provider name (e.g. `"Anthropic"`).
    pub name: String,
    /// API slug used in model identifiers (e.g. `"anthropic"`).
    pub slug: String,
    /// Base URL override (e.g. for self-hosted models).
    pub base_url: Option<String>,
}

impl ProviderConfig {
    pub fn new(name: impl Into<String>, slug: impl Into<String>) -> Self {
        Self { name: name.into(), slug: slug.into(), base_url: None }
    }
}

/// The outcome of a routing decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// The selected model identifier.
    pub model: String,
    /// The provider for the selected model.
    pub provider: String,
    /// Human-readable explanation of why this model was chosen.
    pub reasoning: Option<String>,
    /// Confidence score for this decision, in `[0.0, 1.0]`.
    pub confidence: Option<f32>,
    /// Arbitrary extra metadata (strategy name, cost estimate, etc.).
    pub metadata: HashMap<String, serde_json::Value>,
}

impl RoutingDecision {
    pub fn new(model: impl Into<String>, provider: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            provider: provider.into(),
            reasoning: None,
            confidence: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_reasoning(mut self, r: impl Into<String>) -> Self {
        self.reasoning = Some(r.into());
        self
    }

    pub fn with_confidence(mut self, c: f32) -> Self {
        self.confidence = Some(c.clamp(0.0, 1.0));
        self
    }

    pub fn with_meta(mut self, key: impl Into<String>, val: impl Into<serde_json::Value>) -> Self {
        self.metadata.insert(key.into(), val.into());
        self
    }
}

/// Model complexity tier used for cost estimation.
/// Tier 1 = cheapest/smallest, Tier 5 = most capable/expensive.
pub fn model_tier(model_name: &str) -> u8 {
    let name = model_name.to_lowercase();
    if name.contains("nano") {
        return 1;
    }
    if name.contains("opus") || name.contains("gpt-5") || name.contains("o3")
        || name.contains("gpt-4.1") && !name.contains("mini")
    {
        return 4;
    }
    if name.contains("sonnet") || name.contains("gpt-4o") {
        return 3;
    }
    2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_info_new() {
        let m = ModelInfo::new("gpt-4o", "openai");
        assert_eq!(m.name, "gpt-4o");
        assert_eq!(m.provider, "openai");
        assert!(m.model_type.is_none());
    }

    #[test]
    fn model_info_with_type() {
        let m = ModelInfo::new("embed-v3", "openai").with_type("embedding");
        assert_eq!(m.model_type.as_deref(), Some("embedding"));
    }

    #[test]
    fn routing_decision_builder() {
        let d = RoutingDecision::new("claude-sonnet", "anthropic")
            .with_reasoning("RoundRobin")
            .with_confidence(0.9);
        assert_eq!(d.model, "claude-sonnet");
        assert_eq!(d.provider, "anthropic");
        assert_eq!(d.reasoning.as_deref(), Some("RoundRobin"));
        assert!((d.confidence.unwrap() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn confidence_is_clamped() {
        let d = RoutingDecision::new("m", "p").with_confidence(2.0);
        assert_eq!(d.confidence, Some(1.0));
    }

    #[test]
    fn model_tier_nano() {
        assert_eq!(model_tier("gpt-4-nano"), 1);
    }

    #[test]
    fn model_tier_sonnet() {
        assert_eq!(model_tier("claude-sonnet-4"), 3);
    }

    #[test]
    fn model_tier_gpt4o() {
        assert_eq!(model_tier("gpt-4o"), 3);
    }

    #[test]
    fn model_tier_opus() {
        assert_eq!(model_tier("claude-opus-4"), 4);
    }

    #[test]
    fn model_tier_default() {
        assert_eq!(model_tier("some-unknown-model"), 2);
    }
}
