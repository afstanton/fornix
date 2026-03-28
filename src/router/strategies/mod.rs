//! The `RoutingStrategy` trait and strategy registry.

use crate::router::{
    error::Result,
    types::{ModelInfo, RoutingDecision},
};

/// A routing strategy: given a query and a pool of available models,
/// return a `RoutingDecision`.
pub trait RoutingStrategy: Send + Sync {
    /// Human-readable name for this strategy.
    fn name(&self) -> &'static str;

    /// Select a model for the given query content.
    ///
    /// `content` is the concatenated text of the conversation (last user
    /// message, or the full message list joined). `embedding` is the
    /// pre-computed query embedding for strategies that need it.
    fn route(
        &self,
        content: &str,
        embedding: Option<&[f32]>,
        models: &[ModelInfo],
    ) -> Result<RoutingDecision>;
}

pub mod embedding_threshold;
pub mod regex;
pub mod rorf;
pub mod round_robin;
pub mod weighted_random;

pub use embedding_threshold::EmbeddingThreshold;
pub use regex::RegexStrategy;
pub use rorf::RoRFStrategy;
pub use round_robin::RoundRobin;
pub use weighted_random::WeightedRandom;
