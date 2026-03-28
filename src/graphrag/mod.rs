//! GraphRAG — knowledge graph-augmented retrieval.
//!
//! Fuses structured knowledge graph data with vector and BM25 retrieval to
//! produce richer, more coherent context for LLM calls.
//!
//! # Search modes
//!
//! - [`search::LocalSearch`] — entity-seeded neighbourhood traversal with
//!   optional causal path expansion
//! - [`search::GlobalSearch`] — community-summary ranking by lexical overlap
//! - [`search::HybridSearch`] — combines both
//!
//! # Extraction
//!
//! LLM-backed extraction uses the [`types::EntityExtractor`] and
//! [`types::RelationExtractor`] traits; callers provide implementations.
//! Prompt construction happens inside the pipeline; only the `complete()`
//! call is delegated to the injected [`types::LlmClient`].
//!
//! # Antifragility
//!
//! [`types::IngestObservation`] records each ingestion batch and computes
//! an information-gain score used by the antifragility tracker.

pub mod config;
pub mod error;
pub mod schema;
pub mod search;
pub mod types;

pub use config::GraphRagConfig;
pub use error::{Error, Result};
pub use search::{GlobalSearch, GraphRagSearch, HybridSearch, LocalSearch};
pub use types::{
    EntityExtractor, ExtractedEntity, ExtractedRelation, ExtractionResult,
    IngestObservation, InformationGainWeights, LlmClient, RelationExtractor,
    SearchContext, SearchResult,
};
