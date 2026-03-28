//! BM25 full-text scoring and search.
//!
//! Adapters: synthetic (in-process), pg_bm25, pg_textsearch, paradedb.

/// A single BM25 search result.
pub struct Bm25Result {
    pub id: String,
    pub score: f32,
    pub content: Option<String>,
}

/// Core interface for BM25 search backends.
///
/// Unlike the vector adapter the BM25 adapter is query-scoped —
/// it is constructed per-search with the query baked in.
pub trait Bm25Adapter: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Execute the search and return ranked results.
    fn search(&self) -> Result<Vec<Bm25Result>, Self::Error>;
}

/// Factory interface for constructing a [`Bm25Adapter`] for a given query.
pub trait Bm25AdapterFactory: Send + Sync {
    type Adapter: Bm25Adapter;
    type Error: std::error::Error + Send + Sync + 'static;

    fn build(&self, query: &str) -> Result<Self::Adapter, Self::Error>;
}
