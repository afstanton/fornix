//! Knowledge graph: entities, relations, temporal versioning,
//! causal traversal, community detection, and external references.
//!
//! Implementations:
//! - [`adapters::MemoryGraphAdapter`] — full in-process graph (no external deps)
//! - `adapters::PostgresGraphAdapter` — Postgres-backed (stubbed)
//!
//! # Architecture
//!
//! The graph is **bitemporal** — each record carries both valid-time
//! (when the fact was true in the world) and system-time (when it was
//! recorded). Records are never deleted; they are retracted or superseded,
//! preserving full history.
//!
//! Causal traversal uses DFS with cycle detection. Shortest-path computation
//! uses Dijkstra via `petgraph`.
//!
//! Community detection supports connected-components (built-in) and the
//! Leiden algorithm (via a native extension, stubbed).
//!
//! # Quick start
//!
//! ```rust,no_run
//! use fornix::graph::{GraphAdapter, GraphConfig, adapters::MemoryGraphAdapter};
//! use fornix::graph::adapter::{EntitySearchOptions, CausalOptions};
//!
//! # tokio_test::block_on(async {
//! let g = MemoryGraphAdapter::connect(GraphConfig::default()).await.unwrap();
//!
//! let rain  = g.create_entity("Heavy Rain", "Weather", None, None).await.unwrap();
//! let flood = g.create_entity("Flooding",   "Event",   None, None).await.unwrap();
//! g.create_relation(rain.id, flood.id, "CAUSES", None, None).await.unwrap();
//!
//! let paths = g.causal_descendants(rain.id, CausalOptions::default(), None).await.unwrap();
//! assert_eq!(paths[0].edges[0].relation_type, "CAUSES");
//! # });
//! ```

pub mod adapter;
pub mod adapters;
pub mod chain_confidence;
pub mod community;
pub mod config;
pub mod error;
pub mod schema;
pub mod types;

pub use adapter::{CausalOptions, EntitySearchOptions, GraphAdapter, RelationOptions};
pub use config::GraphConfig;
pub use error::{Error, Result};
pub use types::{
    AssertionState, CausalPath, Community, ConfidenceScores, Entity, ExternalRef, Relation,
};
