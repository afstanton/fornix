//! GraphRAG: entity extraction, ingestion pipeline, community summarisation,
//! federation, and graph-augmented retrieval.
//!
//! Depends on the `graph`, `hybrid`, and `rag` modules.

/// An extracted entity from raw text.
pub struct ExtractedEntity {
    pub name: String,
    pub entity_type: String,
    pub description: Option<String>,
    pub confidence: f32,
}

/// An extracted relation from raw text.
pub struct ExtractedRelation {
    pub from_name: String,
    pub to_name: String,
    pub relation_type: String,
    pub description: Option<String>,
    pub confidence: f32,
}

/// The result of an extraction pass over a text chunk.
pub struct ExtractionResult {
    pub entities: Vec<ExtractedEntity>,
    pub relations: Vec<ExtractedRelation>,
}

/// Interface for entity and relation extraction from text.
///
/// Implementations may use an LLM, MITIE, or other NER backend.
pub trait EntityRelationExtractor: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn extract(&self, text: &str) -> Result<ExtractionResult, Self::Error>;
}

/// Interface for GraphRAG ingestion pipelines.
pub trait IngestionPipeline: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn ingest(&self, text: &str, source_id: Option<&str>) -> Result<(), Self::Error>;
}

/// Interface for a federated graph corpus (remote or local).
pub trait FederationAdapter: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn search(&self, query: &str, limit: usize) -> Result<Vec<crate::rag::RagContext>, Self::Error>;
}

/// Interface for corpus-level analytics (coverage, drift, gap detection).
pub trait CorpusAnalytics: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn coverage_score(&self) -> Result<f32, Self::Error>;
    fn detect_gaps(&self, query: &str) -> Result<Vec<String>, Self::Error>;
}
