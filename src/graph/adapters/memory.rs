//! In-process memory graph adapter.
//!
//! A fully featured knowledge graph with temporal versioning, causal
//! traversal, community detection, and external reference support.
//!
//! Uses `DashMap` for concurrent access and `petgraph` for shortest-path
//! computation. All data is in-memory and lost on drop.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use async_trait::async_trait;
use dashmap::DashMap;
use petgraph::algo::dijkstra;
use petgraph::graph::{DiGraph, NodeIndex};

use crate::common::metadata::Metadata;
use crate::common::namespace::Namespace;
use crate::graph::{
    adapter::{
        CausalOptions, EntitySearchOptions, GraphAdapter, RelationOptions, TraversalDirection,
    },
    chain_confidence::{chain_confidence, chain_strength},
    community::{connected_components, leiden, Edge},
    config::{CommunityAlgorithm, GraphConfig},
    error::{Error, Result},
    schema::{self, ALL_CAUSAL_TYPES},
    types::{
        AssertionState, CausalPath, ChangelogEntry, Community, ConfidenceScores, Entity,
        ExternalRef, Relation, RetargetStats,
    },
};
use crate::store::health::{HealthReport, HealthStatus};

// ============================================================================
// Internal row types
// ============================================================================

#[derive(Debug, Clone)]
struct EntityRow {
    id: u64,
    name: String,
    entity_type: String,
    properties: Metadata,
    valid_from: Option<SystemTime>,
    valid_to: Option<SystemTime>,
    system_from: Option<SystemTime>,
    system_to: Option<SystemTime>,
    superseded_by: Option<u64>,
    assertion_state: AssertionState,
    confidence: ConfidenceScores,
}

impl EntityRow {
    fn active_at(&self, at: SystemTime) -> bool {
        if !self.assertion_state.is_active() {
            return false;
        }
        if let Some(from) = self.valid_from {
            if at < from {
                return false;
            }
        }
        if let Some(to) = self.valid_to {
            if at >= to {
                return false;
            }
        }
        true
    }

    fn is_active(&self) -> bool {
        self.active_at(SystemTime::now())
    }

    fn to_entity(&self) -> Entity {
        Entity {
            id: self.id,
            name: self.name.clone(),
            entity_type: self.entity_type.clone(),
            properties: self.properties.clone(),
            valid_from: self.valid_from,
            valid_to: self.valid_to,
            system_from: self.system_from,
            system_to: self.system_to,
            superseded_by: self.superseded_by,
            assertion_state: self.assertion_state,
            confidence: self.confidence.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct RelationRow {
    id: u64,
    from_id: u64,
    to_id: u64,
    relation_type: String,
    properties: Metadata,
    valid_from: Option<SystemTime>,
    valid_to: Option<SystemTime>,
    system_from: Option<SystemTime>,
    system_to: Option<SystemTime>,
    superseded_by: Option<u64>,
    assertion_state: AssertionState,
    confidence: ConfidenceScores,
}

impl RelationRow {
    fn active_at(&self, at: SystemTime) -> bool {
        if !self.assertion_state.is_active() {
            return false;
        }
        if let Some(from) = self.valid_from {
            if at < from {
                return false;
            }
        }
        if let Some(to) = self.valid_to {
            if at >= to {
                return false;
            }
        }
        true
    }

    fn is_active(&self) -> bool {
        self.active_at(SystemTime::now())
    }

    fn to_relation(&self) -> Relation {
        Relation {
            id: self.id,
            from_id: self.from_id,
            to_id: self.to_id,
            relation_type: self.relation_type.clone(),
            properties: self.properties.clone(),
            valid_from: self.valid_from,
            valid_to: self.valid_to,
            system_from: self.system_from,
            system_to: self.system_to,
            superseded_by: self.superseded_by,
            assertion_state: self.assertion_state,
            confidence: self.confidence.clone(),
        }
    }

    fn causal_strength(&self) -> f32 {
        self.properties
            .get(schema::CAUSAL_STRENGTH)
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
            .unwrap_or(1.0)
    }

    fn strength_passes(&self, min: Option<f32>) -> bool {
        match min {
            Some(m) => self.causal_strength() >= m,
            None => true,
        }
    }
}

// ============================================================================
// Namespace store
// ============================================================================

#[derive(Debug, Default)]
struct NamespaceStore {
    entities: DashMap<u64, EntityRow>,
    relations: DashMap<u64, RelationRow>,
    changelog: std::sync::Mutex<Vec<ChangelogEntry>>,
}

impl NamespaceStore {
    fn new() -> Self {
        Self::default()
    }

    fn active_entities(&self) -> Vec<EntityRow> {
        self.entities
            .iter()
            .filter(|e| e.is_active())
            .map(|e| e.clone())
            .collect()
    }

    fn active_relations(&self) -> Vec<RelationRow> {
        self.relations
            .iter()
            .filter(|r| r.is_active())
            .map(|r| r.clone())
            .collect()
    }

    fn append_changelog(&self, entry: ChangelogEntry, enabled: bool) {
        if !enabled {
            return;
        }
        if let Ok(mut log) = self.changelog.lock() {
            log.push(entry);
        }
    }
}

// ============================================================================
// Adapter
// ============================================================================

/// In-memory graph adapter.
pub struct MemoryGraphAdapter {
    config: GraphConfig,
    connected: bool,
    entity_seq: AtomicU64,
    relation_seq: AtomicU64,
    /// namespace key → NamespaceStore
    stores: DashMap<String, NamespaceStore>,
}

impl MemoryGraphAdapter {
    /// Create a new (disconnected) adapter.
    pub fn new(config: GraphConfig) -> Self {
        Self {
            config,
            connected: false,
            entity_seq: AtomicU64::new(1),
            relation_seq: AtomicU64::new(1),
            stores: DashMap::new(),
        }
    }

    /// Create and immediately connect an adapter.
    pub async fn connect(config: GraphConfig) -> Result<Self> {
        config.validate().map_err(|e| Error::config(e.to_string()))?;
        Ok(Self {
            config,
            connected: true,
            entity_seq: AtomicU64::new(1),
            relation_seq: AtomicU64::new(1),
            stores: DashMap::new(),
        })
    }

    fn resolve_ns<'a>(&'a self, ns: Option<&'a Namespace>) -> &'a str {
        ns.and_then(|n| n.as_deref()).unwrap_or("default")
    }

    fn store(&self, ns: &str) -> dashmap::mapref::one::RefMut<String, NamespaceStore> {
        self.stores
            .entry(ns.to_string())
            .or_insert_with(NamespaceStore::new)
    }

    fn next_entity_id(&self) -> u64 {
        self.entity_seq.fetch_add(1, Ordering::Relaxed)
    }

    fn next_relation_id(&self) -> u64 {
        self.relation_seq.fetch_add(1, Ordering::Relaxed)
    }

    fn now(&self) -> Option<SystemTime> {
        if self.config.default_valid_from_now {
            Some(SystemTime::now())
        } else {
            None
        }
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < f32::EPSILON || nb < f32::EPSILON {
            return 0.0;
        }
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }

    /// DFS-based causal traversal (descendants).
    fn causal_descendants_impl(
        &self,
        ns_store: &NamespaceStore,
        start_id: u64,
        options: &CausalOptions,
    ) -> Vec<CausalPath> {
        let causal_set: std::collections::HashSet<&str> = options
            .types
            .as_deref()
            .unwrap_or(ALL_CAUSAL_TYPES)
            .iter()
            .map(|s| s.as_str())
            .collect();

        let moment = options.as_of;

        let Some(start_row) = ns_store.entities.get(&start_id) else {
            return Vec::new();
        };
        if let Some(at) = moment {
            if !start_row.active_at(at) {
                return Vec::new();
            }
        }

        let mut paths = Vec::new();
        let mut stack: Vec<(Vec<u64>, Vec<u64>, std::collections::HashSet<u64>)> = vec![(
            vec![start_id],
            Vec::new(),
            std::collections::HashSet::from([start_id]),
        )];

        while let Some((node_ids, edge_ids, visited)) = stack.pop() {
            if paths.len() >= options.max_paths {
                break;
            }
            let current_id = *node_ids.last().unwrap();
            let depth = node_ids.len() - 1;

            let candidates: Vec<u64> = if depth < options.max_depth {
                ns_store
                    .relations
                    .iter()
                    .filter(|r| {
                        r.from_id == current_id
                            && causal_set.contains(r.relation_type.as_str())
                            && r.strength_passes(options.min_causal_strength)
                            && moment.map_or(true, |at| r.active_at(at))
                            && !visited.contains(&r.to_id)
                    })
                    .map(|r| r.id)
                    .take(options.max_branching_factor.unwrap_or(usize::MAX))
                    .collect()
            } else {
                Vec::new()
            };

            if candidates.is_empty() {
                if !edge_ids.is_empty() {
                    paths.push(self.build_causal_path(
                        ns_store, &node_ids, &edge_ids, false,
                    ));
                }
            } else {
                for rel_id in candidates {
                    let to_id = ns_store.relations.get(&rel_id).unwrap().to_id;
                    if ns_store.entities.get(&to_id).is_none() {
                        continue;
                    }
                    let mut new_visited = visited.clone();
                    new_visited.insert(to_id);
                    let mut new_nodes = node_ids.clone();
                    new_nodes.push(to_id);
                    let mut new_edges = edge_ids.clone();
                    new_edges.push(rel_id);
                    stack.push((new_nodes, new_edges, new_visited));
                }
            }
        }
        paths
    }

    /// DFS-based causal traversal (ancestors).
    fn causal_ancestors_impl(
        &self,
        ns_store: &NamespaceStore,
        start_id: u64,
        options: &CausalOptions,
    ) -> Vec<CausalPath> {
        let causal_set: std::collections::HashSet<&str> = options
            .types
            .as_deref()
            .unwrap_or(ALL_CAUSAL_TYPES)
            .iter()
            .map(|s| s.as_str())
            .collect();

        let moment = options.as_of;

        if ns_store.entities.get(&start_id).is_none() {
            return Vec::new();
        }

        let mut paths = Vec::new();

        // Find all incoming causal relations from start
        let initial_rels: Vec<u64> = ns_store
            .relations
            .iter()
            .filter(|r| {
                r.to_id == start_id
                    && causal_set.contains(r.relation_type.as_str())
                    && r.strength_passes(options.min_causal_strength)
                    && moment.map_or(true, |at| r.active_at(at))
            })
            .take(options.max_branching_factor.unwrap_or(usize::MAX))
            .map(|r| r.id)
            .collect();

        for rel_id in initial_rels {
            let parent_id = ns_store.relations.get(&rel_id).unwrap().from_id;
            if ns_store.entities.get(&parent_id).is_none() {
                continue;
            }

            let mut stack: Vec<(Vec<u64>, Vec<u64>, std::collections::HashSet<u64>)> = vec![(
                vec![parent_id],
                vec![rel_id],
                std::collections::HashSet::from([start_id, parent_id]),
            )];

            while let Some((node_ids, edge_ids, visited)) = stack.pop() {
                if paths.len() >= options.max_paths {
                    break;
                }
                let first_id = node_ids[0];
                let depth = node_ids.len();

                let candidates: Vec<u64> = if depth < options.max_depth {
                    ns_store
                        .relations
                        .iter()
                        .filter(|r| {
                            r.to_id == first_id
                                && causal_set.contains(r.relation_type.as_str())
                                && r.strength_passes(options.min_causal_strength)
                                && moment.map_or(true, |at| r.active_at(at))
                                && !visited.contains(&r.from_id)
                        })
                        .take(options.max_branching_factor.unwrap_or(usize::MAX))
                        .map(|r| r.id)
                        .collect()
                } else {
                    Vec::new()
                };

                if candidates.is_empty() {
                    paths.push(self.build_causal_path(
                        ns_store, &node_ids, &edge_ids, false,
                    ));
                } else {
                    for candidate_rel_id in candidates {
                        let grandparent_id =
                            ns_store.relations.get(&candidate_rel_id).unwrap().from_id;
                        if ns_store.entities.get(&grandparent_id).is_none() {
                            continue;
                        }
                        let mut new_visited = visited.clone();
                        new_visited.insert(grandparent_id);
                        let mut new_nodes = vec![grandparent_id];
                        new_nodes.extend_from_slice(&node_ids);
                        let mut new_edges = vec![candidate_rel_id];
                        new_edges.extend_from_slice(&edge_ids);
                        stack.push((new_nodes, new_edges, new_visited));
                    }
                }
            }
        }
        paths
    }

    fn build_causal_path(
        &self,
        ns_store: &NamespaceStore,
        node_ids: &[u64],
        edge_ids: &[u64],
        is_complete: bool,
    ) -> CausalPath {
        let nodes: Vec<Entity> = node_ids
            .iter()
            .filter_map(|id| ns_store.entities.get(id).map(|r| r.to_entity()))
            .collect();

        let edges: Vec<Relation> = edge_ids
            .iter()
            .filter_map(|id| ns_store.relations.get(id).map(|r| r.to_relation()))
            .collect();

        let causal_types: Vec<String> = edges.iter().map(|r| r.relation_type.clone()).collect();

        let strengths: Vec<f32> = edge_ids
            .iter()
            .filter_map(|id| ns_store.relations.get(id).map(|r| r.causal_strength()))
            .collect();
        let strength = chain_strength(&strengths);

        let hop_confidences: Vec<f32> = edges
            .iter()
            .map(|r| r.confidence.overall.unwrap_or(1.0))
            .collect();
        let entity_confidences: Vec<f32> = nodes
            .iter()
            .map(|e| e.confidence.overall.unwrap_or(1.0))
            .collect();

        let chain_conf = chain_confidence(&hop_confidences, &entity_confidences, 0.9).ok();

        let weakest_hop = hop_confidences
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);

        CausalPath {
            nodes,
            edges,
            causal_types,
            chain_strength: strength,
            is_complete,
            chain_confidence: chain_conf,
            hop_confidences,
            confidence_ceiling_hop: weakest_hop,
        }
    }
}

#[async_trait]
impl GraphAdapter for MemoryGraphAdapter {
    fn name(&self) -> &'static str {
        "memory"
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    fn config(&self) -> &GraphConfig {
        &self.config
    }

    // =========================================================================
    // Entity CRUD
    // =========================================================================

    async fn create_entity(
        &self,
        name: &str,
        entity_type: &str,
        properties: Option<Metadata>,
        namespace: Option<&Namespace>,
    ) -> Result<Entity> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let id = self.next_entity_id();
        let now_opt = self.now();
        let ns = self.resolve_ns(namespace).to_string();
        let row = EntityRow {
            id,
            name: name.to_string(),
            entity_type: entity_type.to_string(),
            properties: properties.unwrap_or_default(),
            valid_from: now_opt,
            valid_to: None,
            system_from: Some(SystemTime::now()),
            system_to: None,
            superseded_by: None,
            assertion_state: AssertionState::Active,
            confidence: ConfidenceScores::default(),
        };
        let entity = row.to_entity();
        self.store(&ns).entities.insert(id, row.clone());
        self.store(&ns).append_changelog(
            ChangelogEntry {
                operation: "create".to_string(),
                record_type: "entity".to_string(),
                record_id: id,
                system_from: SystemTime::now(),
                payload: HashMap::new(),
            },
            self.config.changelog_enabled,
        );
        Ok(entity)
    }

    async fn find_entity(
        &self,
        id: u64,
        namespace: Option<&Namespace>,
    ) -> Result<Option<Entity>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        Ok(store_ref.and_then(|s| {
            s.entities.get(&id).map(|r| r.to_entity())
        }))
    }

    async fn find_entity_by_name(
        &self,
        name: &str,
        namespace: Option<&Namespace>,
    ) -> Result<Option<Entity>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        Ok(store_ref.and_then(|s| {
            s.entities
                .iter()
                .find(|r| r.name == name)
                .map(|r| r.to_entity())
        }))
    }

    async fn search_entities(
        &self,
        options: EntitySearchOptions,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<Entity>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok(Vec::new());
        };

        let mut results: Vec<Entity> = s
            .entities
            .iter()
            .filter(|r| {
                if let Some(t) = &options.entity_type {
                    if &r.entity_type != t {
                        return false;
                    }
                }
                if let Some(q) = &options.query {
                    if !r.name.to_lowercase().contains(&q.to_lowercase()) {
                        return false;
                    }
                }
                if let Some(min) = options.min_confidence {
                    if r.confidence.overall.map_or(false, |c| c < min) {
                        return false;
                    }
                }
                true
            })
            .map(|r| r.to_entity())
            .collect();

        results.sort_by(|a, b| a.name.cmp(&b.name));
        results.truncate(options.limit);
        Ok(results)
    }

    async fn update_entity(
        &self,
        id: u64,
        updates: HashMap<String, serde_json::Value>,
        namespace: Option<&Namespace>,
    ) -> Result<Option<Entity>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok(None);
        };
        let Some(mut row) = s.entities.get_mut(&id) else {
            return Ok(None);
        };
        if let Some(v) = updates.get("name").and_then(|v| v.as_str()) {
            row.name = v.to_string();
        }
        if let Some(v) = updates.get("entity_type").and_then(|v| v.as_str()) {
            row.entity_type = v.to_string();
        }
        if let Some(serde_json::Value::Object(props)) = updates.get("properties") {
            for (k, v) in props {
                row.properties.insert(k.clone(), v.clone());
            }
        }
        Ok(Some(row.to_entity()))
    }

    async fn delete_entity(
        &self,
        id: u64,
        namespace: Option<&Namespace>,
    ) -> Result<bool> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok(false);
        };
        if s.entities.remove(&id).is_none() {
            return Ok(false);
        }
        // Remove all relations touching this entity
        s.relations.retain(|_, r| r.from_id != id && r.to_id != id);
        Ok(true)
    }

    // =========================================================================
    // Relation CRUD
    // =========================================================================

    async fn create_relation(
        &self,
        from_id: u64,
        to_id: u64,
        relation_type: &str,
        properties: Option<Metadata>,
        namespace: Option<&Namespace>,
    ) -> Result<Relation> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let id = self.next_relation_id();
        let now_opt = self.now();
        let ns = self.resolve_ns(namespace).to_string();
        let row = RelationRow {
            id,
            from_id,
            to_id,
            relation_type: relation_type.to_string(),
            properties: properties.unwrap_or_default(),
            valid_from: now_opt,
            valid_to: None,
            system_from: Some(SystemTime::now()),
            system_to: None,
            superseded_by: None,
            assertion_state: AssertionState::Active,
            confidence: ConfidenceScores::default(),
        };
        let relation = row.to_relation();
        self.store(&ns).relations.insert(id, row);
        Ok(relation)
    }

    async fn find_relations(
        &self,
        options: RelationOptions,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<Relation>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok(Vec::new());
        };

        let current_only = options.current_only
            .unwrap_or(self.config.current_state_reads);

        let mut results: Vec<Relation> = s
            .relations
            .iter()
            .filter(|r| {
                if current_only && !r.is_active() {
                    return false;
                }
                if let Some(from) = options.from_id {
                    if r.from_id != from {
                        return false;
                    }
                }
                if let Some(to) = options.to_id {
                    if r.to_id != to {
                        return false;
                    }
                }
                if let Some(ref t) = options.relation_type {
                    if &r.relation_type != t {
                        return false;
                    }
                }
                if let Some(min) = options.min_confidence {
                    if r.confidence.overall.map_or(false, |c| c < min) {
                        return false;
                    }
                }
                true
            })
            .map(|r| r.to_relation())
            .collect();

        results.sort_by_key(|r| r.id);
        Ok(results)
    }

    async fn update_relation(
        &self,
        id: u64,
        updates: HashMap<String, serde_json::Value>,
        namespace: Option<&Namespace>,
    ) -> Result<Option<Relation>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok(None);
        };
        let Some(mut row) = s.relations.get_mut(&id) else {
            return Ok(None);
        };
        if let Some(v) = updates.get("relation_type").and_then(|v| v.as_str()) {
            row.relation_type = v.to_string();
        }
        if let Some(serde_json::Value::Object(props)) = updates.get("properties") {
            for (k, v) in props {
                row.properties.insert(k.clone(), v.clone());
            }
        }
        Ok(Some(row.to_relation()))
    }

    async fn delete_relation(
        &self,
        id: u64,
        namespace: Option<&Namespace>,
    ) -> Result<bool> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        Ok(store_ref.map_or(false, |s| s.relations.remove(&id).is_some()))
    }

    async fn upsert_relation_embedding(
        &self,
        relation_id: u64,
        vector: Vec<f32>,
        key: Option<&str>,
        namespace: Option<&Namespace>,
    ) -> Result<Option<Relation>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok(None);
        };
        let Some(mut row) = s.relations.get_mut(&relation_id) else {
            return Ok(None);
        };
        let embed_key = key.unwrap_or("embedding").to_string();
        row.properties.insert(embed_key, serde_json::json!(vector));
        Ok(Some(row.to_relation()))
    }

    async fn find_relations_by_embedding(
        &self,
        vector: &[f32],
        limit: usize,
        min_similarity: f32,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<Relation>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok(Vec::new());
        };

        let mut scored: Vec<(f32, Relation)> = s
            .relations
            .iter()
            .filter_map(|r| {
                let stored: Vec<f32> = r
                    .properties
                    .get("embedding")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
                    .unwrap_or_default();
                if stored.is_empty() {
                    return None;
                }
                let sim = Self::cosine_similarity(vector, &stored);
                if sim < min_similarity {
                    return None;
                }
                Some((sim, r.to_relation()))
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored.into_iter().map(|(_, r)| r).collect())
    }

    // =========================================================================
    // Traversal — using petgraph
    // =========================================================================

    async fn neighbors(
        &self,
        entity_id: u64,
        depth: usize,
        direction: TraversalDirection,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<Entity>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        if depth < 1 {
            return Err(Error::invalid_arg("depth must be >= 1"));
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok(Vec::new());
        };

        let mut visited = std::collections::HashSet::new();
        visited.insert(entity_id);
        let mut frontier = vec![entity_id];

        for _ in 0..depth {
            let mut next_frontier = Vec::new();
            for &id in &frontier {
                for r in s.relations.iter() {
                    if !r.is_active() {
                        continue;
                    }
                    let neighbor = match direction {
                        TraversalDirection::Outgoing if r.from_id == id => Some(r.to_id),
                        TraversalDirection::Incoming if r.to_id == id => Some(r.from_id),
                        TraversalDirection::Both if r.from_id == id => Some(r.to_id),
                        TraversalDirection::Both if r.to_id == id => Some(r.from_id),
                        _ => None,
                    };
                    if let Some(nid) = neighbor {
                        if !visited.contains(&nid) {
                            visited.insert(nid);
                            next_frontier.push(nid);
                        }
                    }
                }
            }
            frontier = next_frontier;
        }

        visited.remove(&entity_id);
        Ok(visited
            .iter()
            .filter_map(|id| s.entities.get(id).map(|r| r.to_entity()))
            .collect())
    }

    async fn shortest_path(
        &self,
        from_id: u64,
        to_id: u64,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<Entity>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok(Vec::new());
        };

        // Build petgraph DiGraph from active entities/relations
        let active_entities: Vec<u64> = s.active_entities().iter().map(|e| e.id).collect();
        let active_relations: Vec<(u64, u64)> = s
            .active_relations()
            .iter()
            .map(|r| (r.from_id, r.to_id))
            .collect();

        let mut graph: DiGraph<u64, ()> = DiGraph::new();
        let mut node_map: HashMap<u64, NodeIndex> = HashMap::new();

        for &eid in &active_entities {
            let idx = graph.add_node(eid);
            node_map.insert(eid, idx);
        }

        for (from, to) in active_relations {
            if let (Some(&fi), Some(&ti)) = (node_map.get(&from), node_map.get(&to)) {
                graph.add_edge(fi, ti, ());
            }
        }

        let Some(&start_idx) = node_map.get(&from_id) else {
            return Ok(Vec::new());
        };
        let Some(&end_idx) = node_map.get(&to_id) else {
            return Ok(Vec::new());
        };

        // Dijkstra from start
        let costs = dijkstra(&graph, start_idx, Some(end_idx), |_| 1u32);

        if !costs.contains_key(&end_idx) {
            return Ok(Vec::new()); // no path
        }

        // Reconstruct path by walking backwards through predecessors
        // petgraph's dijkstra doesn't give predecessors directly, so we do BFS
        let mut path_nodes = vec![to_id];
        let mut current = to_id;

        'outer: while current != from_id {
            // Find the predecessor with cost = current_cost - 1
            let current_cost = costs[&node_map[&current]];
            if current_cost == 0 {
                break;
            }
            for (&pred_id, &pred_idx) in &node_map {
                if let Some(&pred_cost) = costs.get(&pred_idx) {
                    if pred_cost + 1 == current_cost {
                        // Check there's an edge pred→current
                        if graph.contains_edge(pred_idx, node_map[&current]) {
                            path_nodes.push(pred_id);
                            current = pred_id;
                            continue 'outer;
                        }
                    }
                }
            }
            break;
        }

        path_nodes.reverse();
        Ok(path_nodes
            .iter()
            .filter_map(|id| s.entities.get(id).map(|r| r.to_entity()))
            .collect())
    }

    async fn subgraph(
        &self,
        seed_ids: &[u64],
        depth: usize,
        namespace: Option<&Namespace>,
    ) -> Result<(Vec<Entity>, Vec<Relation>)> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok((Vec::new(), Vec::new()));
        };

        let mut all_ids: std::collections::HashSet<u64> = seed_ids.iter().cloned().collect();
        let mut frontier: Vec<u64> = seed_ids.to_vec();

        for _ in 0..depth {
            let mut next = Vec::new();
            for &id in &frontier {
                for r in s.relations.iter() {
                    if !r.is_active() {
                        continue;
                    }
                    if r.from_id == id && !all_ids.contains(&r.to_id) {
                        all_ids.insert(r.to_id);
                        next.push(r.to_id);
                    }
                    if r.to_id == id && !all_ids.contains(&r.from_id) {
                        all_ids.insert(r.from_id);
                        next.push(r.from_id);
                    }
                }
            }
            frontier = next;
        }

        let entities: Vec<Entity> = all_ids
            .iter()
            .filter_map(|id| s.entities.get(id).map(|r| r.to_entity()))
            .collect();

        let relations: Vec<Relation> = s
            .relations
            .iter()
            .filter(|r| {
                r.is_active()
                    && all_ids.contains(&r.from_id)
                    && all_ids.contains(&r.to_id)
            })
            .map(|r| r.to_relation())
            .collect();

        Ok((entities, relations))
    }

    // =========================================================================
    // Causal traversal
    // =========================================================================

    async fn causal_descendants(
        &self,
        entity_id: u64,
        options: CausalOptions,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<CausalPath>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok(Vec::new());
        };
        Ok(self.causal_descendants_impl(&s, entity_id, &options))
    }

    async fn causal_ancestors(
        &self,
        entity_id: u64,
        options: CausalOptions,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<CausalPath>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok(Vec::new());
        };
        Ok(self.causal_ancestors_impl(&s, entity_id, &options))
    }

    async fn causal_paths(
        &self,
        from_id: u64,
        to_id: u64,
        options: CausalOptions,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<CausalPath>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok(Vec::new());
        };

        let causal_set: std::collections::HashSet<&str> = options
            .types
            .as_deref()
            .unwrap_or(ALL_CAUSAL_TYPES)
            .iter()
            .map(|s| s.as_str())
            .collect();

        if s.entities.get(&from_id).is_none() || s.entities.get(&to_id).is_none() {
            return Ok(Vec::new());
        }

        let mut paths = Vec::new();
        let mut stack: Vec<(Vec<u64>, Vec<u64>, std::collections::HashSet<u64>)> = vec![(
            vec![from_id],
            Vec::new(),
            std::collections::HashSet::from([from_id]),
        )];

        while let Some((node_ids, edge_ids, visited)) = stack.pop() {
            if paths.len() >= options.max_paths {
                break;
            }
            let current_id = *node_ids.last().unwrap();
            let depth = node_ids.len() - 1;

            if current_id == to_id {
                paths.push(self.build_causal_path(&s, &node_ids, &edge_ids, true));
                continue;
            }

            if depth >= options.max_depth {
                continue;
            }

            for r in s.relations.iter() {
                if r.from_id != current_id
                    || !causal_set.contains(r.relation_type.as_str())
                    || visited.contains(&r.to_id)
                    || s.entities.get(&r.to_id).is_none()
                {
                    continue;
                }
                let mut new_visited = visited.clone();
                new_visited.insert(r.to_id);
                let mut new_nodes = node_ids.clone();
                new_nodes.push(r.to_id);
                let mut new_edges = edge_ids.clone();
                new_edges.push(r.id);
                stack.push((new_nodes, new_edges, new_visited));
            }
        }

        paths.sort_by(|a, b| {
            b.chain_strength
                .partial_cmp(&a.chain_strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(paths)
    }

    // =========================================================================
    // Temporal API
    // =========================================================================

    async fn find_entities_as_of(
        &self,
        at: SystemTime,
        options: EntitySearchOptions,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<Entity>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok(Vec::new());
        };

        let mut results: Vec<Entity> = s
            .entities
            .iter()
            .filter(|r| {
                r.active_at(at)
                    && options.entity_type.as_ref().map_or(true, |t| &r.entity_type == t)
                    && options.query.as_ref().map_or(true, |q| {
                        r.name.to_lowercase().contains(&q.to_lowercase())
                    })
            })
            .map(|r| r.to_entity())
            .collect();

        results.sort_by(|a, b| a.name.cmp(&b.name));
        results.truncate(options.limit);
        Ok(results)
    }

    async fn find_relations_as_of(
        &self,
        at: SystemTime,
        options: RelationOptions,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<Relation>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok(Vec::new());
        };

        let mut results: Vec<Relation> = s
            .relations
            .iter()
            .filter(|r| {
                r.active_at(at)
                    && options.from_id.map_or(true, |id| r.from_id == id)
                    && options.to_id.map_or(true, |id| r.to_id == id)
                    && options.relation_type.as_ref().map_or(true, |t| &r.relation_type == t)
            })
            .map(|r| r.to_relation())
            .collect();

        results.sort_by_key(|r| r.id);
        Ok(results)
    }

    async fn supersede_entity(
        &self,
        old_id: u64,
        new_attributes: HashMap<String, serde_json::Value>,
        namespace: Option<&Namespace>,
    ) -> Result<Entity> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace).to_string();
        let store_ref = self.stores.get(&ns);
        let Some(s) = store_ref else {
            return Err(Error::not_found(format!("entity {}", old_id)));
        };
        let Some(mut old_row) = s.entities.get_mut(&old_id) else {
            return Err(Error::not_found(format!("entity {}", old_id)));
        };

        let now = SystemTime::now();
        old_row.valid_to = Some(now);
        old_row.system_to = Some(now);
        old_row.assertion_state = AssertionState::Superseded;
        drop(old_row);

        let old_row_snap = s.entities.get(&old_id).unwrap().clone();
        drop(s);

        let new_name = new_attributes
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(&old_row_snap.name)
            .to_string();
        let new_type = new_attributes
            .get("entity_type")
            .and_then(|v| v.as_str())
            .unwrap_or(&old_row_snap.entity_type)
            .to_string();
        let new_props = match new_attributes.get("properties") {
            Some(serde_json::Value::Object(m)) => m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            _ => old_row_snap.properties.clone(),
        };

        let new_id = self.next_entity_id();
        let new_row = EntityRow {
            id: new_id,
            name: new_name,
            entity_type: new_type,
            properties: new_props,
            valid_from: Some(now),
            valid_to: None,
            system_from: Some(now),
            system_to: None,
            superseded_by: None,
            assertion_state: AssertionState::Active,
            confidence: ConfidenceScores::default(),
        };
        let entity = new_row.to_entity();
        self.store(&ns).entities.insert(new_id, new_row);

        // Update old record's superseded_by
        if let Some(mut old) = self.stores.get(&ns).and_then(|s| s.entities.get_mut(&old_id)) {
            old.superseded_by = Some(new_id);
        }

        Ok(entity)
    }

    async fn supersede_relation(
        &self,
        old_id: u64,
        new_attributes: HashMap<String, serde_json::Value>,
        namespace: Option<&Namespace>,
    ) -> Result<Relation> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace).to_string();
        let store_ref = self.stores.get(&ns);
        let Some(s) = store_ref else {
            return Err(Error::not_found(format!("relation {}", old_id)));
        };
        let Some(mut old_row) = s.relations.get_mut(&old_id) else {
            return Err(Error::not_found(format!("relation {}", old_id)));
        };

        let now = SystemTime::now();
        old_row.valid_to = Some(now);
        old_row.system_to = Some(now);
        old_row.assertion_state = AssertionState::Superseded;
        let old_snap = old_row.clone();
        drop(old_row);
        drop(s);

        let new_from = new_attributes.get("from_id").and_then(|v| v.as_u64()).unwrap_or(old_snap.from_id);
        let new_to = new_attributes.get("to_id").and_then(|v| v.as_u64()).unwrap_or(old_snap.to_id);
        let new_type = new_attributes
            .get("relation_type")
            .and_then(|v| v.as_str())
            .unwrap_or(&old_snap.relation_type)
            .to_string();

        let new_id = self.next_relation_id();
        let new_row = RelationRow {
            id: new_id,
            from_id: new_from,
            to_id: new_to,
            relation_type: new_type,
            properties: old_snap.properties.clone(),
            valid_from: Some(now),
            valid_to: None,
            system_from: Some(now),
            system_to: None,
            superseded_by: None,
            assertion_state: AssertionState::Active,
            confidence: ConfidenceScores::default(),
        };
        let relation = new_row.to_relation();
        self.store(&ns).relations.insert(new_id, new_row);

        if let Some(s) = self.stores.get(&ns) {
            if let Some(mut old) = s.relations.get_mut(&old_id) {
                old.superseded_by = Some(new_id);
            }
        }

        Ok(relation)
    }

    async fn retract_entity(
        &self,
        id: u64,
        valid_to: Option<SystemTime>,
        namespace: Option<&Namespace>,
    ) -> Result<Entity> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Err(Error::not_found(format!("entity {}", id)));
        };
        let Some(mut row) = s.entities.get_mut(&id) else {
            return Err(Error::not_found(format!("entity {}", id)));
        };
        let now = SystemTime::now();
        row.valid_to = valid_to.or(Some(now));
        row.system_to = Some(now);
        row.assertion_state = AssertionState::Retracted;
        Ok(row.to_entity())
    }

    async fn retract_relation(
        &self,
        id: u64,
        valid_to: Option<SystemTime>,
        namespace: Option<&Namespace>,
    ) -> Result<Relation> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Err(Error::not_found(format!("relation {}", id)));
        };
        let Some(mut row) = s.relations.get_mut(&id) else {
            return Err(Error::not_found(format!("relation {}", id)));
        };
        let now = SystemTime::now();
        row.valid_to = valid_to.or(Some(now));
        row.system_to = Some(now);
        row.assertion_state = AssertionState::Retracted;
        Ok(row.to_relation())
    }

    async fn changelog(
        &self,
        since: Option<SystemTime>,
        until: Option<SystemTime>,
        record_type: Option<&str>,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<ChangelogEntry>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok(Vec::new());
        };
        let log = s.changelog.lock().map_err(|_| Error::operation("changelog lock poisoned"))?;
        let mut entries: Vec<ChangelogEntry> = log
            .iter()
            .filter(|e| {
                record_type.map_or(true, |t| e.record_type == t)
                    && since.map_or(true, |s| e.system_from >= s)
                    && until.map_or(true, |u| e.system_from <= u)
            })
            .cloned()
            .collect();
        entries.sort_by_key(|e| e.system_from);
        Ok(entries)
    }

    // =========================================================================
    // External references
    // =========================================================================

    async fn add_external_ref(
        &self,
        entity_id: u64,
        ext_ref: ExternalRef,
        namespace: Option<&Namespace>,
    ) -> Result<Entity> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Err(Error::not_found(format!("entity {}", entity_id)));
        };
        let Some(mut row) = s.entities.get_mut(&entity_id) else {
            return Err(Error::not_found(format!("entity {}", entity_id)));
        };

        let mut refs = match row.properties.get(schema::EXTERNAL_REFS) {
            Some(serde_json::Value::Array(arr)) => arr.clone(),
            _ => Vec::new(),
        };

        // Remove any existing ref with same source+external_id
        refs.retain(|v| {
            let obj = match v.as_object() {
                Some(o) => o,
                None => return true,
            };
            !(obj.get("source").and_then(|v| v.as_str()) == Some(&ext_ref.source)
                && obj.get("external_id").and_then(|v| v.as_str()) == Some(&ext_ref.external_id))
        });

        refs.push(serde_json::json!({
            "source": ext_ref.source,
            "external_id": ext_ref.external_id,
            "confidence": ext_ref.confidence,
        }));

        row.properties.insert(schema::EXTERNAL_REFS.to_string(), serde_json::Value::Array(refs));
        Ok(row.to_entity())
    }

    async fn find_by_external_ref(
        &self,
        source: &str,
        external_id: &str,
        active_only: bool,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<Entity>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok(Vec::new());
        };

        Ok(s.entities
            .iter()
            .filter(|row| {
                if active_only && !row.is_active() {
                    return false;
                }
                let refs = match row.properties.get(schema::EXTERNAL_REFS) {
                    Some(serde_json::Value::Array(arr)) => arr,
                    _ => return false,
                };
                refs.iter().any(|v| {
                    let obj = match v.as_object() {
                        Some(o) => o,
                        None => return false,
                    };
                    obj.get("source").and_then(|v| v.as_str()) == Some(source)
                        && obj.get("external_id").and_then(|v| v.as_str()) == Some(external_id)
                })
            })
            .map(|r| r.to_entity())
            .collect())
    }

    // =========================================================================
    // Retargeting
    // =========================================================================

    async fn retarget_relations(
        &self,
        old_entity_id: u64,
        new_entity_id: u64,
        namespace: Option<&Namespace>,
    ) -> Result<RetargetStats> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok(RetargetStats { repointed: 0, self_loops_removed: 0, duplicates_merged: 0 });
        };

        let mut repointed = 0usize;
        for mut r in s.relations.iter_mut() {
            let mut changed = false;
            if r.from_id == old_entity_id {
                r.from_id = new_entity_id;
                changed = true;
            }
            if r.to_id == old_entity_id {
                r.to_id = new_entity_id;
                changed = true;
            }
            if changed {
                repointed += 1;
            }
        }

        // Remove self-loops
        let self_loops_removed = {
            let before = s.relations.len();
            s.relations.retain(|_, r| r.from_id != new_entity_id || r.to_id != new_entity_id);
            before - s.relations.len()
        };

        Ok(RetargetStats { repointed, self_loops_removed, duplicates_merged: 0 })
    }

    // =========================================================================
    // Community detection
    // =========================================================================

    async fn detect_communities(
        &self,
        namespace: Option<&Namespace>,
    ) -> Result<Vec<Community>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace);
        let store_ref = self.stores.get(ns);
        let Some(s) = store_ref else {
            return Ok(Vec::new());
        };

        let active_entities = s.active_entities();
        let active_relations: Vec<RelationRow> = s
            .active_relations()
            .into_iter()
            .filter(|r| !r.relation_type.starts_with("__"))
            .collect();

        let node_ids: Vec<String> = active_entities.iter().map(|e| e.id.to_string()).collect();
        let edges: Vec<Edge> = active_relations
            .iter()
            .map(|r| Edge::unweighted(r.from_id.to_string(), r.to_id.to_string()))
            .collect();

        let components = match self.config.default_community_algorithm {
            CommunityAlgorithm::ConnectedComponents => connected_components(&node_ids, &edges),
            CommunityAlgorithm::Leiden => leiden(&node_ids, &edges),
        };

        let entity_map: HashMap<u64, &EntityRow> =
            active_entities.iter().map(|e| (e.id, e)).collect();

        let communities = components
            .into_iter()
            .map(|component_ids| {
                let entities: Vec<Entity> = component_ids
                    .iter()
                    .filter_map(|id_str| {
                        id_str.parse::<u64>().ok().and_then(|id| entity_map.get(&id).map(|r| r.to_entity()))
                    })
                    .collect();

                let entity_id_set: std::collections::HashSet<u64> =
                    entities.iter().map(|e| e.id).collect();

                let edge_count = active_relations
                    .iter()
                    .filter(|r| entity_id_set.contains(&r.from_id) && entity_id_set.contains(&r.to_id))
                    .count();

                let n = entities.len();
                let max_edges = if n > 1 { (n * (n - 1)) / 2 } else { 0 };
                let density = if max_edges == 0 { 0.0 } else { edge_count as f64 / max_edges as f64 };

                let central_entity = if entities.is_empty() {
                    None
                } else {
                    let mut degree: HashMap<u64, usize> = HashMap::new();
                    for r in &active_relations {
                        if entity_id_set.contains(&r.from_id) && entity_id_set.contains(&r.to_id) {
                            *degree.entry(r.from_id).or_insert(0) += 1;
                            *degree.entry(r.to_id).or_insert(0) += 1;
                        }
                    }
                    entities
                        .iter()
                        .max_by_key(|e| degree.get(&e.id).copied().unwrap_or(0))
                        .cloned()
                };

                Community { entities, density, central_entity }
            })
            .collect();

        Ok(communities)
    }

    async fn healthcheck(&self) -> HealthReport {
        let status = if self.connected {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unhealthy { reason: "not connected".to_string() }
        };
        HealthReport::begin("memory-graph").finish(status)
    }
}
