//! Cost estimation and metrics calculation.

use crate::router::types::{model_tier, RoutingDecision};

/// Per-tier estimated cost per 1K input tokens (USD).
/// Mirrors the Ruby `DEFAULT_PRICING` table.
const TIER_PRICING: [(u8, f64); 5] = [
    (1, 0.05),
    (2, 0.15),
    (3, 0.50),
    (4, 1.25),
    (5, 2.00),
];

fn tier_price(tier: u8) -> f64 {
    TIER_PRICING
        .iter()
        .find(|(t, _)| *t == tier)
        .map(|(_, p)| *p)
        .unwrap_or(TIER_PRICING[1].1) // default to tier 2 price
}

/// Computes cost estimates and records decisions into a collector.
#[derive(Default)]
pub struct MetricsCalculator;

impl MetricsCalculator {
    pub fn new() -> Self {
        Self
    }

    /// Estimate the cost for routing a message to `model` at `provider`,
    /// based on approximate token count. Returns `None` when the message
    /// list is empty.
    ///
    /// `messages` is a slice of `(role, content)` pairs.
    pub fn estimated_cost(
        &self,
        model: &str,
        messages: &[(&str, &str)],
    ) -> Option<f64> {
        if messages.is_empty() {
            return None;
        }
        let total_tokens: usize = messages
            .iter()
            .map(|(_, content)| content.split_whitespace().count())
            .sum();
        if total_tokens == 0 {
            return None;
        }
        let tier = model_tier(model);
        let price_per_1k = tier_price(tier);
        Some((total_tokens as f64 / 1000.0) * price_per_1k)
    }

    /// Attach an estimated cost to a decision and record it.
    pub fn enrich_and_record(
        &self,
        decision: RoutingDecision,
        messages: &[(&str, &str)],
        collector: &crate::router::metrics::MetricsCollector,
    ) -> RoutingDecision {
        let enriched = if let Some(cost) = self.estimated_cost(&decision.model, messages) {
            decision.with_meta("estimated_cost", cost)
        } else {
            decision
        };
        collector.record(&enriched);
        enriched
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calc() -> MetricsCalculator {
        MetricsCalculator::new()
    }

    #[test]
    fn empty_messages_returns_none() {
        assert!(calc().estimated_cost("gpt-4o", &[]).is_none());
    }

    #[test]
    fn single_word_message_has_positive_cost() {
        let cost = calc()
            .estimated_cost("gpt-4o", &[("user", "hello")])
            .unwrap();
        assert!(cost > 0.0);
    }

    #[test]
    fn nano_model_cheaper_than_opus() {
        let msgs = &[("user", "write me a poem about the ocean")];
        let nano = calc().estimated_cost("gpt-4-nano", msgs).unwrap();
        let opus = calc().estimated_cost("claude-opus-4", msgs).unwrap();
        assert!(nano < opus, "nano={} opus={}", nano, opus);
    }

    #[test]
    fn cost_scales_with_token_count() {
        let short = calc()
            .estimated_cost("gpt-4o", &[("user", "hi")])
            .unwrap();
        let long = calc()
            .estimated_cost("gpt-4o", &[("user", "a ".repeat(100).trim())])
            .unwrap();
        assert!(long > short);
    }

    #[test]
    fn tier_pricing_table() {
        assert!((tier_price(1) - 0.05).abs() < 1e-9);
        assert!((tier_price(3) - 0.50).abs() < 1e-9);
        assert!((tier_price(4) - 1.25).abs() < 1e-9);
    }

    #[test]
    fn enrich_and_record_adds_cost_to_metadata() {
        use crate::router::metrics::MetricsCollector;
        use crate::router::types::RoutingDecision;

        let collector = MetricsCollector::new();
        let decision = RoutingDecision::new("gpt-4o", "openai");
        let enriched = calc().enrich_and_record(
            decision,
            &[("user", "explain rust ownership")],
            &collector,
        );
        assert!(enriched.metadata.contains_key("estimated_cost"));
        assert_eq!(collector.summary().count, 1);
    }

    #[test]
    fn unknown_tier_falls_back_to_tier_2_price() {
        let cost = calc()
            .estimated_cost("some-unknown-model-xyz", &[("user", "hello world")])
            .unwrap();
        let expected = (2.0 / 1000.0) * 0.15;
        assert!((cost - expected).abs() < 1e-10);
    }
}
