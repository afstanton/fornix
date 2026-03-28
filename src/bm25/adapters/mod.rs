//! BM25 adapter implementations.

pub mod memory;
pub mod postgres;

pub use memory::MemoryBm25Adapter;
