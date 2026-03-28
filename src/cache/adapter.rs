//! The `CacheAdapter` trait — the core interface for all cache backends.

use async_trait::async_trait;
use crate::cache::{config::CacheConfig, error::Result, stats::CacheStats};
use crate::common::namespace::Namespace;
use crate::store::health::HealthReport;

/// The core interface for all cache backends.
///
/// Values are stored as raw bytes (`Vec<u8>`) for binary safety.
/// Callers are responsible for serialising and deserialising their own types.
/// TTL overrides at the call site take precedence over the adapter's
/// configured default TTL.
#[async_trait]
pub trait CacheAdapter: Send + Sync {
    /// Store a value under `key` in `namespace`.
    ///
    /// `ttl` overrides the adapter's configured default TTL.
    /// `None` means use the default; `Some(Duration::ZERO)` explicitly disables expiry.
    async fn set(
        &self,
        key: &str,
        value: Vec<u8>,
        namespace: Option<&Namespace>,
        ttl: Option<std::time::Duration>,
    ) -> Result<()>;

    /// Retrieve a value by `key` from `namespace`.
    ///
    /// Returns `None` if the key does not exist or has expired.
    async fn get(&self, key: &str, namespace: Option<&Namespace>) -> Result<Option<Vec<u8>>>;

    /// Delete a single entry. Returns `true` if the entry existed and was removed.
    async fn delete(&self, key: &str, namespace: Option<&Namespace>) -> Result<bool>;

    /// Delete all entries within `namespace`, or all entries if `None`.
    /// Returns the number of entries removed.
    async fn clear(&self, namespace: Option<&Namespace>) -> Result<usize>;

    /// Returns `true` if the key exists and has not expired.
    async fn exists(&self, key: &str, namespace: Option<&Namespace>) -> Result<bool>;

    /// Returns statistics for `namespace`, or aggregate stats if `None`.
    async fn stats(&self, namespace: Option<&Namespace>) -> Result<CacheStats>;

    /// Check the health of the cache backend.
    async fn healthcheck(&self) -> HealthReport;

    /// The human-readable name of this adapter.
    fn name(&self) -> &'static str;

    /// Whether the adapter is currently connected.
    fn is_connected(&self) -> bool;

    /// The configuration this adapter was built from.
    fn config(&self) -> &CacheConfig;
}
