//! Vector storage and nearest-neighbour search.
//!
//! Adapters: memory, pgvector, qdrant.

use std::collections::HashMap;

/// A stored vector record returned from search or listing.
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub namespace: Option<String>,
}

/// A nearest-neighbour search result with similarity score.
pub struct VectorResult {
    pub record: VectorRecord,
    pub similarity: f32,
}

/// Core interface for vector storage backends.
pub trait VectorAdapter: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn upsert(
        &self,
        id: &str,
        vector: &[f32],
        metadata: Option<HashMap<String, serde_json::Value>>,
        namespace: Option<&str>,
    ) -> Result<(), Self::Error>;

    fn nearest_neighbors(
        &self,
        vector: &[f32],
        limit: usize,
        namespace: Option<&str>,
        min_similarity: Option<f32>,
        include_vectors: bool,
    ) -> Result<Vec<VectorResult>, Self::Error>;

    fn list(
        &self,
        namespace: Option<&str>,
        limit: usize,
        include_vectors: bool,
    ) -> Result<Vec<VectorRecord>, Self::Error>;

    fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), Self::Error>;

    fn delete_namespace(&self, namespace: &str) -> Result<(), Self::Error>;

    fn count(&self, namespace: Option<&str>) -> Result<usize, Self::Error>;

    fn healthy(&self) -> Result<bool, Self::Error>;
}
