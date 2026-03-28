//! In-process memory cache adapter.
//!
//! Useful for testing, development, and as a first-level cache in front of
//! a persistent backend. All data lives in the process and is lost on restart.
//! Thread-safe via `tokio::sync::Mutex`.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::cache::{
    adapter::CacheAdapter,
    config::CacheConfig,
    error::{Error, Result},
    stats::CacheStats,
};
use crate::common::namespace::Namespace;
use crate::store::health::{HealthReport, HealthStatus};

/// A single stored entry in the memory cache.
#[derive(Debug, Clone)]
struct Entry {
    value: Vec<u8>,
    expires_at: Option<SystemTime>,
    namespace: String,
}

impl Entry {
    fn is_expired(&self) -> bool {
        match self.expires_at {
            Some(t) => SystemTime::now() >= t,
            None => false,
        }
    }
}

/// Per-namespace statistics tracked internally.
#[derive(Debug, Default, Clone)]
struct NsStats {
    hits: u64,
    misses: u64,
    evictions: u64,
}

/// In-process memory cache with TTL support and per-namespace statistics.
pub struct MemoryCacheAdapter {
    config: CacheConfig,
    connected: bool,
    inner: Mutex<MemoryInner>,
}

struct MemoryInner {
    store: HashMap<String, Entry>,
    stats: HashMap<String, NsStats>,
}

impl MemoryInner {
    fn new() -> Self {
        Self {
            store: HashMap::new(),
            stats: HashMap::new(),
        }
    }

    fn resolve_ns<'a>(&self, namespace: Option<&'a Namespace>, default: &'a Namespace) -> &'a str {
        namespace
            .and_then(|ns| ns.as_deref())
            .or_else(|| default.as_deref())
            .unwrap_or("default")
    }

    fn namespaced_key(&self, key: &str, ns: &str) -> String {
        format!("{}:{}", ns, key)
    }

    fn purge_if_expired(&mut self, namespaced_key: &str) -> bool {
        if let Some(entry) = self.store.get(namespaced_key) {
            if entry.is_expired() {
                let ns = entry.namespace.clone();
                self.store.remove(namespaced_key);
                self.stats.entry(ns).or_default().evictions += 1;
                return true;
            }
        }
        false
    }

    fn ns_stats(&self, ns: &str) -> CacheStats {
        let s = self.stats.get(ns).cloned().unwrap_or_default();
        let size = self.store.values().filter(|e| e.namespace == ns).count();
        CacheStats { hits: s.hits, misses: s.misses, evictions: s.evictions, size }
    }

    fn aggregate_stats(&self) -> CacheStats {
        let mut agg = CacheStats::default();
        for ns_stats in self.stats.values() {
            agg.hits += ns_stats.hits;
            agg.misses += ns_stats.misses;
            agg.evictions += ns_stats.evictions;
        }
        agg.size = self.store.len();
        agg
    }
}

impl MemoryCacheAdapter {
    /// Create a new (disconnected) memory cache adapter.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            connected: false,
            inner: Mutex::new(MemoryInner::new()),
        }
    }

    /// Create and immediately connect a memory cache adapter.
    pub async fn connect(config: CacheConfig) -> Result<Self> {
        config.validate().map_err(|e| Error::config(e.to_string()))?;
        Ok(Self { config, connected: true, inner: Mutex::new(MemoryInner::new()) })
    }
}

#[async_trait]
impl CacheAdapter for MemoryCacheAdapter {
    fn name(&self) -> &'static str {
        "memory"
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    fn config(&self) -> &CacheConfig {
        &self.config
    }

    async fn set(
        &self,
        key: &str,
        value: Vec<u8>,
        namespace: Option<&Namespace>,
        ttl: Option<Duration>,
    ) -> Result<()> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let mut inner = self.inner.lock().await;
        let ns = inner.resolve_ns(namespace, &self.config.default_namespace).to_string();
        let namespaced_key = inner.namespaced_key(key, &ns);

        let effective_ttl = ttl.or(self.config.default_ttl);
        let expires_at = effective_ttl
            .filter(|d| !d.is_zero())
            .map(|d| SystemTime::now() + d);

        inner.store.insert(namespaced_key, Entry {
            value,
            expires_at,
            namespace: ns,
        });
        Ok(())
    }

    async fn get(&self, key: &str, namespace: Option<&Namespace>) -> Result<Option<Vec<u8>>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let mut inner = self.inner.lock().await;
        let ns = inner.resolve_ns(namespace, &self.config.default_namespace).to_string();
        let namespaced_key = inner.namespaced_key(key, &ns);

        if inner.purge_if_expired(&namespaced_key) {
            inner.stats.entry(ns).or_default().misses += 1;
            return Ok(None);
        }

        match inner.store.get(&namespaced_key) {
            Some(entry) => {
                inner.stats.entry(ns).or_default().hits += 1;
                Ok(Some(entry.value.clone()))
            }
            None => {
                inner.stats.entry(ns).or_default().misses += 1;
                Ok(None)
            }
        }
    }

    async fn delete(&self, key: &str, namespace: Option<&Namespace>) -> Result<bool> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let mut inner = self.inner.lock().await;
        let ns = inner.resolve_ns(namespace, &self.config.default_namespace).to_string();
        let namespaced_key = inner.namespaced_key(key, &ns);
        Ok(inner.store.remove(&namespaced_key).is_some())
    }

    async fn clear(&self, namespace: Option<&Namespace>) -> Result<usize> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let mut inner = self.inner.lock().await;
        match namespace {
            None => {
                let count = inner.store.len();
                inner.store.clear();
                inner.stats.clear();
                Ok(count)
            }
            Some(ns) => {
                let ns_str = ns.as_deref().unwrap_or("default").to_string();
                let keys: Vec<String> = inner.store
                    .iter()
                    .filter(|(_, e)| e.namespace == ns_str)
                    .map(|(k, _)| k.clone())
                    .collect();
                let count = keys.len();
                for k in keys {
                    inner.store.remove(&k);
                }
                inner.stats.remove(&ns_str);
                Ok(count)
            }
        }
    }

    async fn exists(&self, key: &str, namespace: Option<&Namespace>) -> Result<bool> {
        Ok(self.get(key, namespace).await?.is_some())
    }

    async fn stats(&self, namespace: Option<&Namespace>) -> Result<CacheStats> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let inner = self.inner.lock().await;
        let stats = match namespace {
            None => inner.aggregate_stats(),
            Some(ns) => {
                let ns_str = ns.as_deref().unwrap_or("default");
                inner.ns_stats(ns_str)
            }
        };
        Ok(stats)
    }

    async fn healthcheck(&self) -> HealthReport {
        let status = if self.connected {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unhealthy { reason: "not connected".to_string() }
        };
        HealthReport::begin("memory-cache").finish(status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::config::CacheConfig;

    async fn connected() -> MemoryCacheAdapter {
        MemoryCacheAdapter::connect(CacheConfig::default()).await.unwrap()
    }

    fn ns(s: &str) -> Namespace {
        Namespace::named(s)
    }

    // --- Connection lifecycle ---

    #[tokio::test]
    async fn new_adapter_is_disconnected() {
        let a = MemoryCacheAdapter::new(CacheConfig::default());
        assert!(!a.is_connected());
    }

    #[tokio::test]
    async fn connect_produces_connected_adapter() {
        let a = connected().await;
        assert!(a.is_connected());
    }

    #[tokio::test]
    async fn operations_fail_when_not_connected() {
        let a = MemoryCacheAdapter::new(CacheConfig::default());
        let err = a.set("k", b"v".to_vec(), None, None).await.unwrap_err();
        assert!(matches!(err, Error::NotConnected));
    }

    // --- Name ---

    #[tokio::test]
    async fn adapter_name_is_memory() {
        let a = connected().await;
        assert_eq!(a.name(), "memory");
    }

    // --- set / get round-trip ---

    #[tokio::test]
    async fn set_and_get_round_trip() {
        let a = connected().await;
        a.set("k", b"hello".to_vec(), None, None).await.unwrap();
        let v = a.get("k", None).await.unwrap();
        assert_eq!(v, Some(b"hello".to_vec()));
    }

    #[tokio::test]
    async fn get_missing_key_returns_none() {
        let a = connected().await;
        assert_eq!(a.get("missing", None).await.unwrap(), None);
    }

    #[tokio::test]
    async fn set_overwrites_existing_value() {
        let a = connected().await;
        a.set("k", b"v1".to_vec(), None, None).await.unwrap();
        a.set("k", b"v2".to_vec(), None, None).await.unwrap();
        assert_eq!(a.get("k", None).await.unwrap(), Some(b"v2".to_vec()));
    }

    // --- Namespacing ---

    #[tokio::test]
    async fn namespaces_are_isolated() {
        let a = connected().await;
        a.set("k", b"ns1-value".to_vec(), Some(&ns("ns1")), None).await.unwrap();
        a.set("k", b"ns2-value".to_vec(), Some(&ns("ns2")), None).await.unwrap();
        assert_eq!(a.get("k", Some(&ns("ns1"))).await.unwrap(), Some(b"ns1-value".to_vec()));
        assert_eq!(a.get("k", Some(&ns("ns2"))).await.unwrap(), Some(b"ns2-value".to_vec()));
    }

    // --- TTL ---

    #[tokio::test]
    async fn entry_with_far_future_ttl_is_accessible() {
        let a = connected().await;
        a.set("k", b"v".to_vec(), None, Some(Duration::from_secs(3600))).await.unwrap();
        assert_eq!(a.get("k", None).await.unwrap(), Some(b"v".to_vec()));
    }

    #[tokio::test]
    async fn entry_with_zero_duration_ttl_does_not_expire() {
        // Duration::ZERO means "no expiry" in our semantics
        let a = connected().await;
        a.set("k", b"v".to_vec(), None, Some(Duration::ZERO)).await.unwrap();
        assert_eq!(a.get("k", None).await.unwrap(), Some(b"v".to_vec()));
    }

    #[tokio::test]
    async fn expired_entry_returns_none() {
        let a = connected().await;
        // Set a 1 nanosecond TTL — guaranteed to be expired by the time get() runs
        a.set("k", b"v".to_vec(), None, Some(Duration::from_nanos(1))).await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert_eq!(a.get("k", None).await.unwrap(), None);
    }

    // --- delete ---

    #[tokio::test]
    async fn delete_existing_key_returns_true() {
        let a = connected().await;
        a.set("k", b"v".to_vec(), None, None).await.unwrap();
        assert!(a.delete("k", None).await.unwrap());
    }

    #[tokio::test]
    async fn delete_missing_key_returns_false() {
        let a = connected().await;
        assert!(!a.delete("nope", None).await.unwrap());
    }

    #[tokio::test]
    async fn deleted_key_is_no_longer_accessible() {
        let a = connected().await;
        a.set("k", b"v".to_vec(), None, None).await.unwrap();
        a.delete("k", None).await.unwrap();
        assert_eq!(a.get("k", None).await.unwrap(), None);
    }

    // --- clear ---

    #[tokio::test]
    async fn clear_namespace_removes_only_that_namespace() {
        let a = connected().await;
        a.set("k1", b"v".to_vec(), Some(&ns("a")), None).await.unwrap();
        a.set("k2", b"v".to_vec(), Some(&ns("b")), None).await.unwrap();
        let removed = a.clear(Some(&ns("a"))).await.unwrap();
        assert_eq!(removed, 1);
        assert_eq!(a.get("k1", Some(&ns("a"))).await.unwrap(), None);
        assert_eq!(a.get("k2", Some(&ns("b"))).await.unwrap(), Some(b"v".to_vec()));
    }

    #[tokio::test]
    async fn clear_all_removes_everything() {
        let a = connected().await;
        a.set("k1", b"v".to_vec(), Some(&ns("a")), None).await.unwrap();
        a.set("k2", b"v".to_vec(), Some(&ns("b")), None).await.unwrap();
        let removed = a.clear(None).await.unwrap();
        assert_eq!(removed, 2);
        assert_eq!(a.get("k1", Some(&ns("a"))).await.unwrap(), None);
    }

    // --- exists ---

    #[tokio::test]
    async fn exists_true_for_present_key() {
        let a = connected().await;
        a.set("k", b"v".to_vec(), None, None).await.unwrap();
        assert!(a.exists("k", None).await.unwrap());
    }

    #[tokio::test]
    async fn exists_false_for_missing_key() {
        let a = connected().await;
        assert!(!a.exists("missing", None).await.unwrap());
    }

    #[tokio::test]
    async fn exists_false_for_expired_key() {
        let a = connected().await;
        a.set("k", b"v".to_vec(), None, Some(Duration::from_nanos(1))).await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert!(!a.exists("k", None).await.unwrap());
    }

    // --- stats ---

    #[tokio::test]
    async fn stats_reflect_hits_and_misses() {
        let a = connected().await;
        a.set("k", b"v".to_vec(), Some(&ns("s")), None).await.unwrap();
        a.get("k", Some(&ns("s"))).await.unwrap();   // hit
        a.get("nope", Some(&ns("s"))).await.unwrap(); // miss

        let s = a.stats(Some(&ns("s"))).await.unwrap();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
    }

    #[tokio::test]
    async fn stats_count_evictions_on_expired_get() {
        let a = connected().await;
        a.set("k", b"v".to_vec(), Some(&ns("e")), Some(Duration::from_nanos(1))).await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        a.get("k", Some(&ns("e"))).await.unwrap(); // triggers eviction

        let s = a.stats(Some(&ns("e"))).await.unwrap();
        assert_eq!(s.evictions, 1);
    }

    #[tokio::test]
    async fn aggregate_stats_sum_across_namespaces() {
        let a = connected().await;
        a.set("k", b"v".to_vec(), Some(&ns("x")), None).await.unwrap();
        a.set("k", b"v".to_vec(), Some(&ns("y")), None).await.unwrap();
        a.get("k", Some(&ns("x"))).await.unwrap();
        a.get("k", Some(&ns("y"))).await.unwrap();

        let s = a.stats(None).await.unwrap();
        assert_eq!(s.hits, 2);
        assert_eq!(s.size, 2);
    }

    // --- healthcheck ---

    #[tokio::test]
    async fn healthcheck_healthy_when_connected() {
        let a = connected().await;
        let r = a.healthcheck().await;
        assert!(r.status.is_healthy());
    }

    #[tokio::test]
    async fn healthcheck_unhealthy_when_not_connected() {
        let a = MemoryCacheAdapter::new(CacheConfig::default());
        let r = a.healthcheck().await;
        assert!(!r.status.is_usable());
    }
}
