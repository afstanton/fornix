//! The `Bm25Adapter` trait — the core interface for all BM25 backends.

use async_trait::async_trait;
use std::collections::HashMap;
use crate::bm25::{config::Bm25Config, error::Result, result::Bm25Result};
use crate::common::namespace::Namespace;
use crate::store::health::HealthReport;

/// A document to be indexed.
///
/// Fields are named text segments. For a blog post this might be
/// `{"title": "...", "body": "..."}`. For a single-field index, a single
/// entry with an empty key is conventional.
#[derive(Debug, Clone)]
pub struct IndexDocument {
    /// Stable identifier for this document.
    pub id: String,
    /// Named text fields. Each field is indexed and scored independently,
    /// then combined into a per-document total score.
    pub fields: HashMap<String, String>,
}

impl IndexDocument {
    /// Construct a single-field document.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        let mut fields = HashMap::new();
        fields.insert("text".to_string(), text.into());
        Self { id: id.into(), fields }
    }

    /// Construct a multi-field document from an iterator of (field, text) pairs.
    pub fn with_fields(
        id: impl Into<String>,
        fields: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        Self {
            id: id.into(),
            fields: fields.into_iter().map(|(k, v)| (k.into(), v.into())).collect(),
        }
    }
}

/// Options for a BM25 search query.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Maximum number of results to return.
    pub limit: usize,
    /// Minimum score threshold; documents below this are excluded.
    pub min_score: Option<f32>,
    /// Which fields to search. Empty means all indexed fields.
    pub fields: Vec<String>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self { limit: 20, min_score: None, fields: Vec::new() }
    }
}

impl SearchOptions {
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_min_score(mut self, min: f32) -> Self {
        self.min_score = Some(min);
        self
    }

    pub fn with_fields(mut self, fields: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.fields = fields.into_iter().map(Into::into).collect();
        self
    }
}

/// The core interface for all BM25 backends.
///
/// Unlike the Ruby implementation which is query-scoped (adapter constructed
/// per search), the Rust adapter is long-lived and manages its own index
/// lifecycle.
#[async_trait]
pub trait Bm25Adapter: Send + Sync {
    /// The human-readable name of this adapter.
    fn name(&self) -> &'static str;

    /// Whether the adapter is currently connected.
    fn is_connected(&self) -> bool;

    /// The configuration this adapter was built from.
    fn config(&self) -> &Bm25Config;

    /// Index a document, replacing any existing document with the same id.
    async fn index(
        &self,
        document: IndexDocument,
        namespace: Option<&Namespace>,
    ) -> Result<()>;

    /// Remove a document by id. Returns `true` if it existed.
    async fn remove(
        &self,
        id: &str,
        namespace: Option<&Namespace>,
    ) -> Result<bool>;

    /// Search for documents matching `query`.
    /// Results are returned in descending score order.
    async fn search(
        &self,
        query: &str,
        namespace: Option<&Namespace>,
        options: SearchOptions,
    ) -> Result<Vec<Bm25Result>>;

    /// Number of indexed documents in `namespace`, or all namespaces if `None`.
    async fn count(&self, namespace: Option<&Namespace>) -> Result<usize>;

    /// Remove all documents in `namespace`, or all documents if `None`.
    /// Returns the number of documents removed.
    async fn clear(&self, namespace: Option<&Namespace>) -> Result<usize>;

    /// Health check for the backend.
    async fn healthcheck(&self) -> HealthReport;
}
