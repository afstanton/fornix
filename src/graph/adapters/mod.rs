//! Graph adapter implementations.

pub mod memory;
#[cfg(test)]
mod memory_tests;
pub mod postgres;

pub use memory::MemoryGraphAdapter;
