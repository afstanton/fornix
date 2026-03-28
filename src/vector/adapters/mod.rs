//! Vector adapter implementations.

pub mod memory;
pub mod pgvector;
pub mod qdrant;

pub use memory::MemoryVectorAdapter;
