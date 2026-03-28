//! Hybrid search: fused vector + BM25 with score normalisation and
//! confidence scoring.
//!
//! Fusion strategies:
//! - [`fusion::rrf::Rrf`] — Reciprocal Rank Fusion (rank-based, scale-invariant)
//! - [`fusion::linear::Linear`] — weighted linear combination of normalised scores
//!
//! Normalisation (for linear fusion):
//! - [`normalizer::min_max`] — scales to `[0, 1]`
//! - [`normalizer::z_score`] — zero mean, unit variance
//! - [`normalizer::none`] — pass-through
//!
//! Confidence scoring assigns a `[0, 1]` confidence estimate to each result
//! based on its percentile rank, gap to the next result, and whether both
//! sources agreed on it.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use fornix::hybrid::{HybridSearch, HybridConfig, search::HybridSearchOptions};
//! use fornix::bm25::{adapters::MemoryBm25Adapter, Bm25Config, adapter::IndexDocument};
//! use fornix::vector::{adapters::MemoryVectorAdapter, VectorConfig};
//!
//! # tokio_test::block_on(async {
//! let bm25 = MemoryBm25Adapter::connect(Bm25Config::default()).await.unwrap();
//! let vector = MemoryVectorAdapter::connect(VectorConfig::with_dimension(2)).await.unwrap();
//!
//! bm25.index(IndexDocument::new("doc-1", "rust systems programming"), None).await.unwrap();
//! vector.upsert("doc-1", vec![1.0, 0.0], None, None).await.unwrap();
//!
//! let hs = HybridSearch::new(bm25, vector, HybridConfig::default());
//! let results = hs.search("rust", &[1.0, 0.0], None, HybridSearchOptions::new())
//!     .await
//!     .unwrap();
//! assert_eq!(results[0].id, "doc-1");
//! # });
//! ```

pub mod confidence;
pub mod config;
pub mod error;
pub mod fusion;
pub mod normalizer;
pub mod result;
pub mod search;

pub use config::HybridConfig;
pub use error::{Error, Result};
pub use result::HybridResult;
pub use search::HybridSearch;
