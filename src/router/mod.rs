//! LLM provider routing: strategy-based model selection with metrics.
//!
//! Routing strategies: regex, embedding threshold, random, round-robin.

/// Information about an LLM provider/model endpoint.
pub struct ModelInfo {
    pub provider: String,
    pub model: String,
    pub context_window: Option<usize>,
    pub supports_streaming: bool,
}

/// The outcome of a routing decision.
pub struct RoutingDecision {
    pub model: ModelInfo,
    pub strategy: String,
    pub score: Option<f32>,
}

/// Interface for LLM routing strategies.
///
/// Given a query and a set of candidate models, selects one.
pub trait RoutingStrategy: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn name(&self) -> &'static str;
    fn select(
        &self,
        query: &str,
        candidates: &[ModelInfo],
    ) -> Result<RoutingDecision, Self::Error>;
}

/// Interface for routing metrics collection and reporting.
pub trait RoutingMetrics: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn record(&self, decision: &RoutingDecision, latency_ms: u64, success: bool) -> Result<(), Self::Error>;
    fn summary(&self) -> Result<String, Self::Error>;
}
