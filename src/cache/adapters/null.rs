//! Null cache adapter — accepts all writes and returns nothing.
//!
//! Useful for disabling caching in test environments or when a cache
//! is optional and the caller wants a no-op implementation.

use async_trait::async_trait;

use crate::cache::{
    adapter::CacheAdapter,
    config::CacheConfig,
    error::Result,
    stats::CacheStats,
};
use crate::common::namespace::Namespace;
use crate::store::health::{HealthReport, HealthStatus};

/// A no-op cache adapter.
///
/// All writes succeed silently. All reads return `None`.
/// Useful for disabling caching without changing call sites.
pub struct NullCacheAdapter {
    config: CacheConfig,
}

impl NullCacheAdapter {
    /// Create a connected null adapter. Never fails.
    pub fn new(config: CacheConfig) -> Self {
        Self { config }
    }
}

impl Default for NullCacheAdapter {
    fn default() -> Self {
        Self::new(CacheConfig::default())
    }
}

#[async_trait]
impl CacheAdapter for NullCacheAdapter {
    fn name(&self) -> &'static str {
        "null"
    }

    fn is_connected(&self) -> bool {
        true
    }

    fn config(&self) -> &CacheConfig {
        &self.config
    }

    async fn set(
        &self,
        _key: &str,
        _value: Vec<u8>,
        _namespace: Option<&Namespace>,
        _ttl: Option<std::time::Duration>,
    ) -> Result<()> {
        Ok(())
    }

    async fn get(&self, _key: &str, _namespace: Option<&Namespace>) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }

    async fn delete(&self, _key: &str, _namespace: Option<&Namespace>) -> Result<bool> {
        Ok(false)
    }

    async fn clear(&self, _namespace: Option<&Namespace>) -> Result<usize> {
        Ok(0)
    }

    async fn exists(&self, _key: &str, _namespace: Option<&Namespace>) -> Result<bool> {
        Ok(false)
    }

    async fn stats(&self, _namespace: Option<&Namespace>) -> Result<CacheStats> {
        Ok(CacheStats::default())
    }

    async fn healthcheck(&self) -> HealthReport {
        HealthReport::begin("null-cache").finish(HealthStatus::Healthy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter() -> NullCacheAdapter {
        NullCacheAdapter::default()
    }

    #[tokio::test]
    async fn is_always_connected() {
        assert!(adapter().is_connected());
    }

    #[tokio::test]
    async fn name_is_null() {
        assert_eq!(adapter().name(), "null");
    }

    #[tokio::test]
    async fn set_always_succeeds() {
        let a = adapter();
        a.set("k", b"v".to_vec(), None, None).await.unwrap();
    }

    #[tokio::test]
    async fn get_always_returns_none() {
        let a = adapter();
        a.set("k", b"v".to_vec(), None, None).await.unwrap();
        assert_eq!(a.get("k", None).await.unwrap(), None);
    }

    #[tokio::test]
    async fn delete_always_returns_false() {
        let a = adapter();
        assert!(!a.delete("k", None).await.unwrap());
    }

    #[tokio::test]
    async fn clear_always_returns_zero() {
        let a = adapter();
        assert_eq!(a.clear(None).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn exists_always_returns_false() {
        let a = adapter();
        a.set("k", b"v".to_vec(), None, None).await.unwrap();
        assert!(!a.exists("k", None).await.unwrap());
    }

    #[tokio::test]
    async fn stats_always_returns_zeros() {
        let a = adapter();
        let s = a.stats(None).await.unwrap();
        assert_eq!(s, CacheStats::default());
    }

    #[tokio::test]
    async fn healthcheck_always_healthy() {
        let r = adapter().healthcheck().await;
        assert!(r.status.is_healthy());
    }
}
