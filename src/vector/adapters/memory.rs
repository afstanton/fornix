//! In-process memory vector adapter.
//!
//! Useful for testing and development. All data is lost on drop.
//! Thread-safe via `tokio::sync::RwLock` (reads are far more common
//! than writes in typical search workloads).

use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::common::metadata::Metadata;
use crate::common::namespace::Namespace;
use crate::common::pagination::{Page, PageParams};
use crate::store::health::{HealthReport, HealthStatus};
use crate::vector::{
    adapter::{ListOptions, SearchOptions, VectorAdapter},
    analysis::cosine_similarity,
    config::VectorConfig,
    error::{Error, Result},
    filter::MetadataFilter,
    result::{VectorRecord, VectorResult},
};

/// A stored entry inside the memory adapter.
#[derive(Clone)]
struct Entry {
    vector: Vec<f32>,
    metadata: Metadata,
}

/// Inner state protected by the RwLock.
struct Inner {
    /// namespace → (id → entry)
    store: HashMap<String, HashMap<String, Entry>>,
}

impl Inner {
    fn new() -> Self {
        Self { store: HashMap::new() }
    }

    fn ns_key(ns: &str) -> String {
        ns.to_string()
    }

    fn namespace_entries(&self, ns: &str) -> Option<&HashMap<String, Entry>> {
        self.store.get(&Self::ns_key(ns))
    }

    fn namespace_entries_mut(&mut self, ns: &str) -> &mut HashMap<String, Entry> {
        self.store.entry(Self::ns_key(ns)).or_default()
    }
}

/// In-memory vector adapter with cosine similarity search.
pub struct MemoryVectorAdapter {
    config: VectorConfig,
    connected: bool,
    inner: RwLock<Inner>,
}

impl MemoryVectorAdapter {
    /// Create a new (disconnected) adapter.
    pub fn new(config: VectorConfig) -> Self {
        Self {
            config,
            connected: false,
            inner: RwLock::new(Inner::new()),
        }
    }

    /// Create and immediately connect an adapter.
    pub async fn connect(config: VectorConfig) -> Result<Self> {
        config
            .validate()
            .map_err(|e| Error::config(e.to_string()))?;
        Ok(Self {
            config,
            connected: true,
            inner: RwLock::new(Inner::new()),
        })
    }

    fn resolve_ns<'a>(&'a self, ns: Option<&'a Namespace>) -> &'a str {
        ns.and_then(|n| n.as_deref())
            .or_else(|| self.config.default_namespace.as_deref())
            .unwrap_or("default")
    }
}

#[async_trait]
impl VectorAdapter for MemoryVectorAdapter {
    fn name(&self) -> &'static str {
        "memory"
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    fn config(&self) -> &VectorConfig {
        &self.config
    }

    async fn upsert(
        &self,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<Metadata>,
        namespace: Option<&Namespace>,
    ) -> Result<()> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        self.config
            .check_dimension(vector.len())
            .map_err(|e| Error::dimension_mismatch(
                match e {
                    crate::store::error::Error::DimensionMismatch { expected, .. } => expected,
                    _ => 0,
                },
                vector.len(),
            ))?;

        let ns = self.resolve_ns(namespace).to_string();
        let mut inner = self.inner.write().await;
        inner.namespace_entries_mut(&ns).insert(
            id.to_string(),
            Entry {
                vector,
                metadata: metadata.unwrap_or_default(),
            },
        );
        Ok(())
    }

    async fn nearest_neighbors(
        &self,
        query: &[f32],
        namespace: Option<&Namespace>,
        options: SearchOptions,
    ) -> Result<Vec<VectorResult>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        self.config
            .check_dimension(query.len())
            .map_err(|e| match e {
                crate::store::error::Error::DimensionMismatch { expected, actual } => {
                    Error::dimension_mismatch(expected, actual)
                }
                _ => Error::operation(e.to_string()),
            })?;

        let ns = self.resolve_ns(namespace).to_string();
        let inner = self.inner.read().await;
        let filter = options.filter.unwrap_or_default();

        let mut results: Vec<VectorResult> = inner
            .namespace_entries(&ns)
            .into_iter()
            .flat_map(|entries| entries.iter())
            .filter(|(_, entry)| filter.matches(&entry.metadata))
            .filter_map(|(id, entry)| {
                let sim = cosine_similarity(query, &entry.vector).ok()?;
                if let Some(min) = options.min_similarity {
                    if sim < min {
                        return None;
                    }
                }
                Some(VectorResult::new(
                    id.clone(),
                    sim,
                    entry.metadata.clone(),
                    if options.include_vectors { Some(entry.vector.clone()) } else { None },
                ))
            })
            .collect();

        results.sort_by(|a, b| {
            b.score()
                .partial_cmp(&a.score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(options.limit);
        Ok(results)
    }

    async fn list(
        &self,
        namespace: Option<&Namespace>,
        page: PageParams,
        options: ListOptions,
    ) -> Result<Page<VectorRecord>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace).to_string();
        let inner = self.inner.read().await;
        let filter = options.filter.unwrap_or_default();

        let mut records: Vec<(String, VectorRecord)> = inner
            .namespace_entries(&ns)
            .into_iter()
            .flat_map(|entries| entries.iter())
            .filter(|(_, entry)| filter.matches(&entry.metadata))
            .map(|(id, entry)| {
                (
                    id.clone(),
                    VectorRecord {
                        id: id.clone(),
                        metadata: entry.metadata.clone(),
                        vector: if options.include_vectors {
                            Some(entry.vector.clone())
                        } else {
                            None
                        },
                    },
                )
            })
            .collect();

        // Deterministic ordering for pagination
        records.sort_by(|(a, _), (b, _)| a.cmp(b));

        // Cursor-based pagination: cursor is the last id seen
        let start = if let Some(cursor) = &page.cursor {
            records
                .iter()
                .position(|(id, _)| id == cursor)
                .map(|i| i + 1)
                .unwrap_or(0)
        } else {
            0
        };

        let total = records.len();
        let slice: Vec<VectorRecord> = records
            .into_iter()
            .skip(start)
            .take(page.limit)
            .map(|(_, rec)| rec)
            .collect();

        let next_cursor = if start + slice.len() < total {
            slice.last().map(|rec| rec.id.clone())
        } else {
            None
        };

        Ok(match next_cursor {
            Some(cursor) => Page::with_cursor(slice, cursor, Some(total)),
            None => Page::last(slice, Some(total)),
        })
    }

    async fn delete(&self, id: &str, namespace: Option<&Namespace>) -> Result<bool> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace).to_string();
        let mut inner = self.inner.write().await;
        Ok(inner
            .namespace_entries_mut(&ns)
            .remove(id)
            .is_some())
    }

    async fn delete_namespace(&self, namespace: &Namespace) -> Result<usize> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = namespace.as_deref().unwrap_or("default");
        let mut inner = self.inner.write().await;
        Ok(inner.store.remove(ns).map(|m| m.len()).unwrap_or(0))
    }

    async fn count(&self, namespace: Option<&Namespace>) -> Result<usize> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let inner = self.inner.read().await;
        Ok(match namespace {
            Some(ns) => inner
                .namespace_entries(ns.as_deref().unwrap_or("default"))
                .map(|m| m.len())
                .unwrap_or(0),
            None => inner.store.values().map(|m| m.len()).sum(),
        })
    }

    async fn healthcheck(&self) -> HealthReport {
        let status = if self.connected {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unhealthy { reason: "not connected".to_string() }
        };
        HealthReport::begin("memory-vector").finish(status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::metadata::Metadata;
    use crate::common::pagination::PageParams;
    use crate::vector::filter::MetadataFilter;
    use serde_json::json;

    async fn adapter_dim(dim: usize) -> MemoryVectorAdapter {
        MemoryVectorAdapter::connect(VectorConfig::with_dimension(dim)).await.unwrap()
    }

    async fn adapter() -> MemoryVectorAdapter {
        MemoryVectorAdapter::connect(VectorConfig::default()).await.unwrap()
    }

    fn ns(s: &str) -> Namespace { Namespace::named(s) }

    fn vec2(a: f32, b: f32) -> Vec<f32> { vec![a, b] }

    fn opts() -> SearchOptions { SearchOptions::default() }

    // --- lifecycle ---

    #[tokio::test]
    async fn new_is_disconnected() {
        let a = MemoryVectorAdapter::new(VectorConfig::default());
        assert!(!a.is_connected());
    }

    #[tokio::test]
    async fn connect_produces_connected_adapter() {
        let a = adapter().await;
        assert!(a.is_connected());
    }

    #[tokio::test]
    async fn operations_fail_when_disconnected() {
        let a = MemoryVectorAdapter::new(VectorConfig::default());
        let err = a.upsert("id", vec![1.0], None, None).await.unwrap_err();
        assert!(matches!(err, Error::NotConnected));
    }

    #[tokio::test]
    async fn name_is_memory() {
        assert_eq!(adapter().await.name(), "memory");
    }

    // --- upsert / count ---

    #[tokio::test]
    async fn upsert_increments_count() {
        let a = adapter().await;
        a.upsert("a", vec2(1.0, 0.0), None, None).await.unwrap();
        assert_eq!(a.count(None).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn upsert_same_id_replaces() {
        let a = adapter().await;
        a.upsert("a", vec2(1.0, 0.0), None, None).await.unwrap();
        a.upsert("a", vec2(0.0, 1.0), None, None).await.unwrap();
        assert_eq!(a.count(None).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn upsert_rejects_wrong_dimension() {
        let a = adapter_dim(2).await;
        let err = a.upsert("a", vec![1.0, 2.0, 3.0], None, None).await.unwrap_err();
        assert!(matches!(err, Error::DimensionMismatch { .. }));
    }

    // --- nearest_neighbors ---

    #[tokio::test]
    async fn nearest_neighbors_returns_closest_first() {
        let a = adapter().await;
        a.upsert("far",  vec2(0.0, 1.0), None, None).await.unwrap();
        a.upsert("near", vec2(1.0, 0.1), None, None).await.unwrap();

        let results = a
            .nearest_neighbors(&vec2(1.0, 0.0), None, opts())
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].id, "near");
    }

    #[tokio::test]
    async fn nearest_neighbors_respects_limit() {
        let a = adapter().await;
        for i in 0..5 {
            a.upsert(&i.to_string(), vec2(i as f32, 0.0), None, None)
                .await
                .unwrap();
        }
        let results = a
            .nearest_neighbors(&vec2(1.0, 0.0), None, opts().with_limit(2))
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn nearest_neighbors_respects_min_similarity() {
        let a = adapter().await;
        a.upsert("close",    vec2(1.0, 0.0),  None, None).await.unwrap();
        a.upsert("far",      vec2(0.0, 1.0),  None, None).await.unwrap();

        let results = a
            .nearest_neighbors(
                &vec2(1.0, 0.0),
                None,
                opts().with_min_similarity(0.9),
            )
            .await
            .unwrap();

        assert!(results.iter().all(|r| r.score() >= 0.9));
    }

    #[tokio::test]
    async fn nearest_neighbors_respects_metadata_filter() {
        let a = adapter().await;
        let mut m = Metadata::new();
        m.insert("type".to_string(), json!("doc"));
        a.upsert("doc",   vec2(1.0, 0.0), Some(m), None).await.unwrap();
        a.upsert("other", vec2(1.0, 0.0), None,    None).await.unwrap();

        let filter = MetadataFilter::new().with("type", json!("doc"));
        let results = a
            .nearest_neighbors(&vec2(1.0, 0.0), None, opts().with_filter(filter))
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc");
    }

    #[tokio::test]
    async fn nearest_neighbors_include_vectors() {
        let a = adapter().await;
        a.upsert("a", vec2(1.0, 0.0), None, None).await.unwrap();
        let results = a
            .nearest_neighbors(&vec2(1.0, 0.0), None, opts().include_vectors())
            .await
            .unwrap();
        assert!(results[0].vector.is_some());
    }

    #[tokio::test]
    async fn nearest_neighbors_without_vectors_has_none() {
        let a = adapter().await;
        a.upsert("a", vec2(1.0, 0.0), None, None).await.unwrap();
        let results = a.nearest_neighbors(&vec2(1.0, 0.0), None, opts()).await.unwrap();
        assert!(results[0].vector.is_none());
    }

    // --- namespacing ---

    #[tokio::test]
    async fn namespaces_are_isolated() {
        let a = adapter().await;
        a.upsert("id", vec2(1.0, 0.0), None, Some(&ns("ns1"))).await.unwrap();
        a.upsert("id", vec2(0.0, 1.0), None, Some(&ns("ns2"))).await.unwrap();

        assert_eq!(a.count(Some(&ns("ns1"))).await.unwrap(), 1);
        assert_eq!(a.count(Some(&ns("ns2"))).await.unwrap(), 1);
        assert_eq!(a.count(None).await.unwrap(), 2);
    }

    // --- delete ---

    #[tokio::test]
    async fn delete_existing_returns_true() {
        let a = adapter().await;
        a.upsert("a", vec2(1.0, 0.0), None, None).await.unwrap();
        assert!(a.delete("a", None).await.unwrap());
        assert_eq!(a.count(None).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn delete_missing_returns_false() {
        let a = adapter().await;
        assert!(!a.delete("nope", None).await.unwrap());
    }

    // --- delete_namespace ---

    #[tokio::test]
    async fn delete_namespace_removes_all_in_namespace() {
        let a = adapter().await;
        a.upsert("a", vec2(1.0, 0.0), None, Some(&ns("x"))).await.unwrap();
        a.upsert("b", vec2(0.0, 1.0), None, Some(&ns("x"))).await.unwrap();
        a.upsert("c", vec2(1.0, 0.0), None, Some(&ns("y"))).await.unwrap();

        let removed = a.delete_namespace(&ns("x")).await.unwrap();
        assert_eq!(removed, 2);
        assert_eq!(a.count(Some(&ns("x"))).await.unwrap(), 0);
        assert_eq!(a.count(Some(&ns("y"))).await.unwrap(), 1);
    }

    // --- list / pagination ---

    #[tokio::test]
    async fn list_returns_all_records() {
        let a = adapter().await;
        for i in 0..3 {
            a.upsert(&i.to_string(), vec2(i as f32, 0.0), None, None)
                .await
                .unwrap();
        }
        let page = a.list(None, PageParams::first(100), ListOptions::default()).await.unwrap();
        assert_eq!(page.items.len(), 3);
        assert_eq!(page.total, Some(3));
    }

    #[tokio::test]
    async fn list_pagination_with_cursor() {
        let a = adapter().await;
        for i in 0..5 {
            a.upsert(&format!("rec-{:02}", i), vec2(i as f32, 0.0), None, None)
                .await
                .unwrap();
        }
        let first = a.list(None, PageParams::first(2), ListOptions::default()).await.unwrap();
        assert_eq!(first.items.len(), 2);
        assert!(first.has_next());

        let cursor = first.next_cursor.unwrap();
        let second = a.list(None, PageParams::after(cursor, 2), ListOptions::default()).await.unwrap();
        assert_eq!(second.items.len(), 2);
    }

    #[tokio::test]
    async fn list_respects_metadata_filter() {
        let a = adapter().await;
        let mut m = Metadata::new();
        m.insert("keep".to_string(), json!(true));
        a.upsert("keep",   vec2(1.0, 0.0), Some(m), None).await.unwrap();
        a.upsert("drop",   vec2(1.0, 0.0), None,    None).await.unwrap();

        let filter = MetadataFilter::new().with("keep", json!(true));
        let page = a
            .list(None, PageParams::first(100), ListOptions { filter: Some(filter), include_vectors: false })
            .await
            .unwrap();
        assert_eq!(page.items.len(), 1);
        assert_eq!(page.items[0].id, "keep");
    }

    // --- healthcheck ---

    #[tokio::test]
    async fn healthcheck_healthy_when_connected() {
        let r = adapter().await.healthcheck().await;
        assert!(r.status.is_healthy());
    }

    #[tokio::test]
    async fn healthcheck_unhealthy_when_not_connected() {
        let a = MemoryVectorAdapter::new(VectorConfig::default());
        let r = a.healthcheck().await;
        assert!(!r.status.is_usable());
    }
}
