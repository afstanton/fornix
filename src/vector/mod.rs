//! Vector storage and nearest-neighbour search.
//!
//! Provides a [`VectorAdapter`] trait and implementations for:
//! - [`adapters::MemoryVectorAdapter`] — in-process (no external deps)
//! - `adapters::PgvectorAdapter` — Postgres + pgvector extension (stubbed)
//! - `adapters::QdrantAdapter` — Qdrant (stubbed)
//!
//! Search results are typed as [`result::VectorResult`] with a [`result::Similarity`]
//! newtype that enforces `[0.0, 1.0]` at construction time.
//!
//! The [`analysis`] module provides pure mathematical functions over
//! embeddings and result sets (surprisal, entropy, cosine geometry, etc.).
//!
//! # Quick start
//!
//! ```rust,no_run
//! use fornix::vector::{VectorAdapter, VectorConfig, adapters::MemoryVectorAdapter};
//! use fornix::vector::adapter::SearchOptions;
//!
//! # async fn _doc() {
//! let adapter = MemoryVectorAdapter::connect(VectorConfig::with_dimension(2)).await.unwrap();
//! adapter.upsert("doc-1", vec![1.0, 0.0], None, None).await.unwrap();
//! let results = adapter
//!     .nearest_neighbors(&[1.0, 0.0], None, SearchOptions::default())
//!     .await
//!     .unwrap();
//! assert_eq!(results[0].id, "doc-1");
//! # }
//! ```

pub mod adapter;
pub mod adapters;
pub mod analysis;
pub mod config;
pub mod error;
pub mod filter;
pub mod result;

pub use adapter::{ListOptions, SearchOptions, VectorAdapter};
pub use config::VectorConfig;
pub use error::{Error, Result};
pub use filter::MetadataFilter;
pub use result::{Similarity, VectorRecord, VectorResult};
