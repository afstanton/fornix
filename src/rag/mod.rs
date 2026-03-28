//! RAG (Retrieval-Augmented Generation) pipeline components.
//!
//! This module provides the text-processing and retrieval-quality infrastructure
//! that sits between raw search results and the final context window presented
//! to an LLM.
//!
//! # Components
//!
//! - **Chunkers** — split documents into retrievable pieces:
//!   [`FixedLength`], [`TokenCount`], [`SentenceToken`], [`ParentChild`]
//! - **Rerankers** — reorder retrieved contexts by relevance:
//!   [`NullReranker`] (pass-through), with cross-encoder support planned
//! - **Output filters** — post-process the context list:
//!   [`MinScoreFilter`], [`DeduplicateFilter`], [`TruncateFilter`],
//!   [`FilterPipeline`]
//! - **Query gap tracker** — surfaces corpus gaps from missed queries
//! - **Evaluation** — [`Evaluator`] with precision, recall, faithfulness,
//!   and answer-relevance metrics
//!
//! # Quick start
//!
//! ```rust
//! use fornix::rag::chunkers::{Chunker, TokenCount};
//!
//! let chunker = TokenCount::new(200, 20);
//! let chunks = chunker.chunk("The quick brown fox jumps over the lazy dog.");
//! assert!(!chunks.is_empty());
//! ```

pub mod chunkers;
pub mod error;
pub mod evaluation;
pub mod output_filter;
pub mod query_gap_tracker;
pub mod rerankers;
pub mod tokenizer;
pub mod types;

pub use chunkers::{Chunker, FixedLength, ParentChild, SentenceToken, TokenCount};
pub use error::{Error, Result};
pub use output_filter::{DeduplicateFilter, FilterPipeline, MinScoreFilter, OutputFilter, TruncateFilter};
pub use query_gap_tracker::QueryGapTracker;
pub use rerankers::{NullReranker, Reranker};
pub use types::{Chunk, Context, FilteredResult, RagResult};
