//! BM25 full-text scoring and search.
//!
//! Implements the Okapi BM25 ranking algorithm with:
//! - [`tokenizer::Tokenizer`] — text → token stream (lowercase, alphanumeric,
//!   length filtering, stop words, optional suffix stemming)
//! - [`scorer::Scorer`] — pure BM25 mathematics (k1 / b parameters)
//! - [`adapters::MemoryBm25Adapter`] — full in-process inverted index
//! - `adapters::PostgresBm25Adapter` — Postgres-backed (stubbed)
//!
//! # Quick start
//!
//! ```rust,no_run
//! use fornix::bm25::{Bm25Config, Bm25Adapter, adapter::{IndexDocument, SearchOptions}};
//! use fornix::bm25::adapters::MemoryBm25Adapter;
//!
//! # async fn _doc() {
//! let adapter = MemoryBm25Adapter::connect(Bm25Config::default()).await.unwrap();
//!
//! adapter.index(IndexDocument::new("doc-1", "Rust is a systems programming language"), None).await.unwrap();
//! adapter.index(IndexDocument::new("doc-2", "Python is great for scripting"), None).await.unwrap();
//!
//! let results = adapter.search("rust systems", None, SearchOptions::default()).await.unwrap();
//! assert_eq!(results[0].id, "doc-1");
//! # }
//! ```

pub mod adapter;
pub mod adapters;
pub mod config;
pub mod error;
pub mod result;
pub mod scorer;
pub mod tokenizer;

pub use adapter::{Bm25Adapter, IndexDocument, SearchOptions};
pub use config::Bm25Config;
pub use error::{Error, Result};
pub use result::Bm25Result;
pub use scorer::Scorer;
pub use tokenizer::Tokenizer;
