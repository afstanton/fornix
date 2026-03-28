//! LLM routing: model selection strategies, cost metrics, and RoRF.
//!
//! The router selects which LLM to call for a given query from a configured
//! pool of models. Five strategies are provided:
//!
//! - [`strategies::RegexStrategy`] — rule-based routing by content patterns
//! - [`strategies::RoundRobin`] — even load distribution (thread-safe)
//! - [`strategies::WeightedRandom`] — probabilistic selection by weight
//! - [`strategies::EmbeddingThreshold`] — complexity routing via embedding centroids
//! - [`strategies::RoRFStrategy`] — learned routing via a trained Random Forest
//!
//! # RoRF
//!
//! RoRF trains a binary Random Forest classifier on `(embedding, label)` pairs
//! where label 0 = model_a preferred and label 1 = model_b preferred. At
//! inference time it embeds the query, runs it through the forest, and routes
//! to model_a when `P(model_a) ≥ threshold`.
//!
//! ```rust,no_run
//! use fornix::router::{
//!     strategies::{RoRFStrategy, RoutingStrategy},
//!     forest::ForestParams,
//! };
//!
//! // Training data: embeddings + labels (0 = use strong model, 1 = use weak model)
//! let features: Vec<Vec<f32>> = vec![vec![0.1], vec![0.9]];
//! let labels: Vec<u8> = vec![0, 1];
//!
//! let router = RoRFStrategy::train(
//!     &features,
//!     &labels,
//!     0.5,                // threshold
//!     "claude-opus-4",    // model_a (strong)
//!     "anthropic",
//!     "claude-sonnet-4",  // model_b (weak)
//!     "anthropic",
//!     ForestParams::default(),
//! ).unwrap();
//!
//! let decision = router.route("explain Rust lifetimes", Some(&[0.15]), &[]).unwrap();
//! assert_eq!(decision.model, "claude-opus-4");
//! ```
//!
//! # Cost metrics
//!
//! [`metrics::MetricsCalculator`] estimates per-request cost by tier (1–5)
//! and [`metrics::MetricsCollector`] records decisions for aggregate reporting.

pub mod error;
pub mod forest;
pub mod metrics;
pub mod strategies;
pub mod types;

pub use error::{Error, Result};
pub use forest::{train as train_forest, ForestParams, RandomForest};
pub use metrics::{MetricsCalculator, MetricsCollector};
pub use strategies::{
    EmbeddingThreshold, EmbeddingThresholdConfig, RegexRule, RegexStrategy, RoRFStrategy,
    RoundRobin, RoutingStrategy, WeightedModel, WeightedRandom,
};
pub use types::{model_tier, ModelInfo, ProviderConfig, RoutingDecision};
