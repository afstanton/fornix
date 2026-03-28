//! Knowledge graph: entities, relations, causal traversal, temporal versioning.
//!
//! Adapters: postgres, neo4j, sparql, rgl, memory, file.

use std::collections::HashMap;

/// A graph entity (node).
pub struct Entity {
    pub id: String,
    pub entity_type: String,
    pub name: Option<String>,
    pub properties: HashMap<String, serde_json::Value>,
    pub confidence: Option<f32>,
}

/// A directed relation (edge) between two entities.
pub struct Relation {
    pub id: String,
    pub from_id: String,
    pub to_id: String,
    pub relation_type: String,
    pub properties: HashMap<String, serde_json::Value>,
    pub confidence: Option<f32>,
}

/// A chain of causal hops between entities.
pub struct CausalPath {
    pub nodes: Vec<Entity>,
    pub edges: Vec<Relation>,
    pub chain_strength: f32,
    pub chain_confidence: Option<f32>,
    pub is_complete: bool,
}

/// Core interface for knowledge graph backends.
///
/// Includes entity/relation CRUD, traversal, temporal versioning,
/// causal path-finding, and community detection.
pub trait GraphAdapter: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    // --- Entity CRUD ---

    fn create_entity(&self, entity: Entity) -> Result<Entity, Self::Error>;
    fn find_entity(&self, id: &str) -> Result<Option<Entity>, Self::Error>;
    fn search_entities(&self, query: &str, limit: usize) -> Result<Vec<Entity>, Self::Error>;
    fn update_entity(&self, id: &str, properties: HashMap<String, serde_json::Value>) -> Result<Entity, Self::Error>;
    fn delete_entity(&self, id: &str) -> Result<(), Self::Error>;

    // --- Relation CRUD ---

    fn create_relation(&self, relation: Relation) -> Result<Relation, Self::Error>;
    fn find_relations(&self, from_id: Option<&str>, to_id: Option<&str>) -> Result<Vec<Relation>, Self::Error>;
    fn update_relation(&self, id: &str, properties: HashMap<String, serde_json::Value>) -> Result<Relation, Self::Error>;
    fn delete_relation(&self, id: &str) -> Result<(), Self::Error>;

    // --- Traversal ---

    fn neighbors(&self, entity_id: &str, depth: usize) -> Result<Vec<Entity>, Self::Error>;
    fn shortest_path(&self, from_id: &str, to_id: &str) -> Result<Vec<Entity>, Self::Error>;
    fn subgraph(&self, seed_ids: &[&str], depth: usize) -> Result<(Vec<Entity>, Vec<Relation>), Self::Error>;

    // --- Causal traversal ---

    fn causal_descendants(
        &self,
        entity_id: &str,
        max_depth: usize,
        max_paths: usize,
    ) -> Result<Vec<CausalPath>, Self::Error>;

    fn causal_ancestors(
        &self,
        entity_id: &str,
        max_depth: usize,
        max_paths: usize,
    ) -> Result<Vec<CausalPath>, Self::Error>;

    fn causal_paths(
        &self,
        from_id: &str,
        to_id: &str,
        max_depth: usize,
        max_paths: usize,
    ) -> Result<Vec<CausalPath>, Self::Error>;

    // --- Temporal ---

    fn find_entities_as_of(&self, timestamp: std::time::SystemTime) -> Result<Vec<Entity>, Self::Error>;
    fn find_relations_as_of(&self, timestamp: std::time::SystemTime) -> Result<Vec<Relation>, Self::Error>;
    fn supersede_entity(&self, id: &str, replacement: Entity) -> Result<Entity, Self::Error>;
    fn supersede_relation(&self, id: &str, replacement: Relation) -> Result<Relation, Self::Error>;
    fn retract_entity(&self, id: &str) -> Result<(), Self::Error>;
    fn retract_relation(&self, id: &str) -> Result<(), Self::Error>;

    // --- Community detection ---

    fn detect_communities(&self) -> Result<Vec<Vec<String>>, Self::Error>;

    // --- Health ---

    fn healthy(&self) -> Result<bool, Self::Error>;
}
