//! Caching layer for embeddings and LLM responses.
//!
//! Provides a [`CacheAdapter`] trait and implementations for:
//! - [`adapters::MemoryCacheAdapter`] — in-process, TTL-aware (no external deps)
//! - [`adapters::NullCacheAdapter`] — no-op for disabling caching
//! - `adapters::PostgresCacheAdapter` — persistent (stubbed, requires sqlx)
//! - `adapters::RedisCacheAdapter` — persistent (stubbed, requires redis crate)
//!
//! Cache keys are built deterministically via [`key::CacheKey`].
//! Statistics are reported as a typed [`stats::CacheStats`] struct.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use fornix::cache::{CacheConfig, adapters::MemoryCacheAdapter, CacheAdapter};
//!
//! # tokio_test::block_on(async {
//! let adapter = MemoryCacheAdapter::connect(CacheConfig::default()).await.unwrap();
//! adapter.set("my-key", b"my-value".to_vec(), None, None).await.unwrap();
//! let value = adapter.get("my-key", None).await.unwrap();
//! assert_eq!(value, Some(b"my-value".to_vec()));
//! # });
//! ```

pub mod adapter;
pub mod adapters;
pub mod config;
pub mod error;
pub mod key;
pub mod stats;

pub use adapter::CacheAdapter;
pub use config::CacheConfig;
pub use error::{Error, Result};
pub use key::CacheKey;
pub use stats::CacheStats;
