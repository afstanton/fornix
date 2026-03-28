//! The `GraphAdapter` trait — core interface for all graph backends.

use async_trait::async_trait;
use std::collections::HashMap;
use std::time::SystemTime;

use crate::common::namespace::Namespace;
use crate::graph::{
    config::GraphConfig,
    error::Result,
    types::{
        CausalPath, ChangelogEntry, Community, Entity, ExternalRef, Relation, RetargetStats,
    },
};
use crate::store::health::HealthReport;

/// Options for entity search.
#[derive(Debug, Clone, Default)]
pub struct EntitySearchOptions {
    /// Filter by entity type.
    pub entity_type: Option<String>,
    /// Filter by substring match on name.
    pub query: Option<String>,
    /// Maximum number of results. Default: 20.
    pub limit: usize,
    /// Minimum composite confidence threshold.
    pub min_confidence: Option<f32>,
}

impl EntitySearchOptions {
    pub fn new() -> Self {
        Self { limit: 20, ..Default::default() }
    }
    pub fn with_type(mut self, t: impl Into<String>) -> Self {
        self.entity_type = Some(t.into());
        self
    }
    pub fn with_query(mut self, q: impl Into<String>) -> Self {
        self.query = Some(q.into());
        self
    }
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
    pub fn with_min_confidence(mut self, min: f32) -> Self {
        self.min_confidence = Some(min);
        self
    }
}

/// Options for relation lookup.
#[derive(Debug, Clone, Default)]
pub struct RelationOptions {
    pub from_id: Option<u64>,
    pub to_id: Option<u64>,
    pub relation_type: Option<String>,
    pub min_confidence: Option<f32>,
    /// When `true`, only currently-active relations are returned regardless
    /// of the adapter's `current_state_reads` setting.
    pub current_only: Option<bool>,
}

/// Direction of neighbour traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TraversalDirection {
    Outgoing,
    Incoming,
    #[default]
    Both,
}

/// Options for causal traversal.
#[derive(Debug, Clone)]
pub struct CausalOptions {
    /// Causal types to follow. `None` means all causal types.
    pub types: Option<Vec<String>>,
    pub max_depth: usize,
    pub max_paths: usize,
    pub max_branching_factor: Option<usize>,
    pub min_causal_strength: Option<f32>,
    /// Optional valid-time point for temporal filtering.
    pub as_of: Option<SystemTime>,
}

impl Default for CausalOptions {
    fn default() -> Self {
        Self {
            types: None,
            max_depth: 5,
            max_paths: 50,
            max_branching_factor: None,
            min_causal_strength: None,
            as_of: None,
        }
    }
}

/// The core interface for all graph storage backends.
#[async_trait]
pub trait GraphAdapter: Send + Sync {
    /// The human-readable name of this adapter.
    fn name(&self) -> &'static str;

    /// Whether the adapter is currently connected.
    fn is_connected(&self) -> bool;

    /// The configuration this adapter was built from.
    fn config(&self) -> &GraphConfig;

    // =========================================================================
    // Entity CRUD
    // =========================================================================

    async fn create_entity(
        &self,
        name: &str,
        entity_type: &str,
        properties: Option<crate::common::metadata::Metadata>,
        namespace: Option<&Namespace>,
    ) -> Result<Entity>;

    async fn find_entity(
        &self,
        id: u64,
        namespace: Option<&Namespace>,
    ) -> Result<Option<Entity>>;

    async fn find_entity_by_name(
        &self,
        name: &str,
        namespace: Option<&Namespace>,
    ) -> Result<Option<Entity>>;

    async fn search_entities(
        &self,
        options: EntitySearchOptions,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<Entity>>;

    async fn update_entity(
        &self,
        id: u64,
        updates: HashMap<String, serde_json::Value>,
        namespace: Option<&Namespace>,
    ) -> Result<Option<Entity>>;

    async fn delete_entity(
        &self,
        id: u64,
        namespace: Option<&Namespace>,
    ) -> Result<bool>;

    // =========================================================================
    // Relation CRUD
    // =========================================================================

    async fn create_relation(
        &self,
        from_id: u64,
        to_id: u64,
        relation_type: &str,
        properties: Option<crate::common::metadata::Metadata>,
        namespace: Option<&Namespace>,
    ) -> Result<Relation>;

    async fn find_relations(
        &self,
        options: RelationOptions,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<Relation>>;

    async fn update_relation(
        &self,
        id: u64,
        updates: HashMap<String, serde_json::Value>,
        namespace: Option<&Namespace>,
    ) -> Result<Option<Relation>>;

    async fn delete_relation(
        &self,
        id: u64,
        namespace: Option<&Namespace>,
    ) -> Result<bool>;

    async fn upsert_relation_embedding(
        &self,
        relation_id: u64,
        vector: Vec<f32>,
        key: Option<&str>,
        namespace: Option<&Namespace>,
    ) -> Result<Option<Relation>>;

    async fn find_relations_by_embedding(
        &self,
        vector: &[f32],
        limit: usize,
        min_similarity: f32,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<Relation>>;

    // =========================================================================
    // Traversal
    // =========================================================================

    async fn neighbors(
        &self,
        entity_id: u64,
        depth: usize,
        direction: TraversalDirection,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<Entity>>;

    async fn shortest_path(
        &self,
        from_id: u64,
        to_id: u64,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<Entity>>;

    async fn subgraph(
        &self,
        seed_ids: &[u64],
        depth: usize,
        namespace: Option<&Namespace>,
    ) -> Result<(Vec<Entity>, Vec<Relation>)>;

    // =========================================================================
    // Causal traversal
    // =========================================================================

    async fn causal_descendants(
        &self,
        entity_id: u64,
        options: CausalOptions,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<CausalPath>>;

    async fn causal_ancestors(
        &self,
        entity_id: u64,
        options: CausalOptions,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<CausalPath>>;

    async fn causal_paths(
        &self,
        from_id: u64,
        to_id: u64,
        options: CausalOptions,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<CausalPath>>;

    // =========================================================================
    // Temporal API
    // =========================================================================

    async fn find_entities_as_of(
        &self,
        at: SystemTime,
        options: EntitySearchOptions,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<Entity>>;

    async fn find_relations_as_of(
        &self,
        at: SystemTime,
        options: RelationOptions,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<Relation>>;

    async fn supersede_entity(
        &self,
        old_id: u64,
        new_attributes: HashMap<String, serde_json::Value>,
        namespace: Option<&Namespace>,
    ) -> Result<Entity>;

    async fn supersede_relation(
        &self,
        old_id: u64,
        new_attributes: HashMap<String, serde_json::Value>,
        namespace: Option<&Namespace>,
    ) -> Result<Relation>;

    async fn retract_entity(
        &self,
        id: u64,
        valid_to: Option<SystemTime>,
        namespace: Option<&Namespace>,
    ) -> Result<Entity>;

    async fn retract_relation(
        &self,
        id: u64,
        valid_to: Option<SystemTime>,
        namespace: Option<&Namespace>,
    ) -> Result<Relation>;

    async fn changelog(
        &self,
        since: Option<SystemTime>,
        until: Option<SystemTime>,
        record_type: Option<&str>,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<ChangelogEntry>>;

    // =========================================================================
    // External references
    // =========================================================================

    async fn add_external_ref(
        &self,
        entity_id: u64,
        ext_ref: ExternalRef,
        namespace: Option<&Namespace>,
    ) -> Result<Entity>;

    async fn find_by_external_ref(
        &self,
        source: &str,
        external_id: &str,
        active_only: bool,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<Entity>>;

    // =========================================================================
    // Relation retargeting
    // =========================================================================

    async fn retarget_relations(
        &self,
        old_entity_id: u64,
        new_entity_id: u64,
        namespace: Option<&Namespace>,
    ) -> Result<RetargetStats>;

    // =========================================================================
    // Community detection
    // =========================================================================

    async fn detect_communities(
        &self,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<Community>>;

    // =========================================================================
    // Health
    // =========================================================================

    async fn healthcheck(&self) -> HealthReport;
}
