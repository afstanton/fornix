//! Cache adapter implementations.

pub mod memory;
pub mod null;
pub mod postgres;
pub mod redis;

pub use memory::MemoryCacheAdapter;
pub use null::NullCacheAdapter;
