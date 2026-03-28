//! Text chunking: the `Chunker` trait and all built-in implementations.
//!
//! Each chunker takes a `&str` and returns a `Vec<Chunk>`.

use crate::rag::types::Chunk;

/// The core chunking interface.
pub trait Chunker: Send + Sync {
    /// Split `text` into chunks.
    fn chunk(&self, text: &str) -> Vec<Chunk>;

    /// Human-readable name for this chunker.
    fn name(&self) -> &'static str;
}

pub mod fixed_length;
pub mod parent_child;
pub mod sentence_token;
pub mod token_count;

pub use fixed_length::FixedLength;
pub use parent_child::ParentChild;
pub use sentence_token::SentenceToken;
pub use token_count::TokenCount;
