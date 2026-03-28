//! The `VectorAdapter` trait — the core interface for all vector backends.

use async_trait::async_trait;
use crate::common::namespace::Namespace;
use crate::common::pagination::{Page, PageParams};
use crate::store::health::HealthReport;
use crate::vector::{
    config::VectorConfig,
    error::Result,
    filter::MetadataFilter,
    result::{VectorRecord, VectorResult},
};

/// Options for a nearest-neighbour search.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Maximum number of results to return.
    pub limit: usize,
    /// Optional metadata filter applied to candidates before scoring.
    pub filter: Option<MetadataFilter>,
    /// Minimum similarity score; results below this threshold are excluded.
    pub min_similarity: Option<f32>,
    /// Whether to include the stored vector in each result.
    pub include_vectors: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: 20,
            filter: None,
            min_similarity: None,
            include_vectors: false,
        }
    }
}

impl SearchOptions {
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_filter(mut self, filter: MetadataFilter) -> Self {
        self.filter = Some(filter);
        self
    }

    pub fn with_min_similarity(mut self, min: f32) -> Self {
        self.min_similarity = Some(min);
        self
    }

    pub fn include_vectors(mut self) -> Self {
        self.include_vectors = true;
        self
    }
}

/// Options for a list operation.
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct ListOptions {
    /// Optional metadata filter.
    pub filter: Option<MetadataFilter>,
    /// Whether to include the stored vector in each record.
    pub include_vectors: bool,
}


/// The core interface for all vector storage backends.
#[async_trait]
pub trait VectorAdapter: Send + Sync {
    /// The human-readable name of this adapter.
    fn name(&self) -> &'static str;

    /// Whether the adapter is currently connected.
    fn is_connected(&self) -> bool;

    /// The configuration this adapter was built from.
    fn config(&self) -> &VectorConfig;

    /// Insert or update a record. If a record with `id` already exists in
    /// `namespace`, its vector and metadata are replaced.
    async fn upsert(
        &self,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<crate::common::metadata::Metadata>,
        namespace: Option<&Namespace>,
    ) -> Result<()>;

    /// Return the `limit` nearest neighbours to `query` in `namespace`,
    /// ordered by descending similarity.
    async fn nearest_neighbors(
        &self,
        query: &[f32],
        namespace: Option<&Namespace>,
        options: SearchOptions,
    ) -> Result<Vec<VectorResult>>;

    /// List records in `namespace` with optional filtering and pagination.
    async fn list(
        &self,
        namespace: Option<&Namespace>,
        page: PageParams,
        options: ListOptions,
    ) -> Result<Page<VectorRecord>>;

    /// Delete a single record. Returns `true` if it existed.
    async fn delete(&self, id: &str, namespace: Option<&Namespace>) -> Result<bool>;

    /// Delete all records in a namespace. Returns the number removed.
    async fn delete_namespace(&self, namespace: &Namespace) -> Result<usize>;

    /// Count records in `namespace`, or all records if `None`.
    async fn count(&self, namespace: Option<&Namespace>) -> Result<usize>;

    /// Health check for the backend connection.
    async fn healthcheck(&self) -> HealthReport;
}
