//! Graph domain types: Entity, Relation, CausalPath, ExternalRef,
//! ClaimSource, Community, and AssertionState.

use std::collections::HashMap;
use std::time::SystemTime;

use crate::common::metadata::Metadata;

/// The assertion state of a temporal record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AssertionState {
    /// The record is currently valid.
    #[default]
    Active,
    /// The record has been explicitly retracted (logically deleted).
    Retracted,
    /// The record has been replaced by a newer version.
    Superseded,
}

impl AssertionState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Retracted => "retracted",
            Self::Superseded => "superseded",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "retracted" => Self::Retracted,
            "superseded" => Self::Superseded,
            _ => Self::Active,
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active)
    }
}

impl std::fmt::Display for AssertionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Confidence scores decomposed into source-specific components.
#[derive(Debug, Clone, Default)]
pub struct ConfidenceScores {
    /// Overall composite confidence.
    pub overall: Option<f32>,
    /// Confidence contributed by an LLM extraction pass.
    pub llm: Option<f32>,
    /// Confidence from frequency of observation.
    pub frequency: Option<f32>,
    /// Confidence from corroborating sources.
    pub corroboration: Option<f32>,
    /// Confidence after human or automated resolution.
    pub resolution: Option<f32>,
}

/// A reference to an external identifier system for an entity.
#[derive(Debug, Clone)]
pub struct ExternalRef {
    /// The name of the external identifier system (e.g. `"wikidata"`).
    pub source: String,
    /// The identifier within that system.
    pub external_id: String,
    /// Confidence that this mapping is correct.
    pub confidence: Option<f32>,
    /// When this mapping became valid (valid-time).
    pub valid_from: Option<SystemTime>,
    /// When this mapping ceased to be valid (valid-time).
    pub valid_to: Option<SystemTime>,
    /// Additional properties for this reference.
    pub properties: Option<Metadata>,
}

impl ExternalRef {
    /// Returns `true` if this reference is active at the given time.
    pub fn active_at(&self, at: SystemTime) -> bool {
        if let Some(from) = self.valid_from
            && at < from {
                return false;
            }
        if let Some(to) = self.valid_to
            && at >= to {
                return false;
            }
        true
    }

    /// Returns `true` if this reference is currently active.
    pub fn is_active(&self) -> bool {
        self.active_at(SystemTime::now())
    }
}

/// A source citation attached to a claim in an entity or relation.
#[derive(Debug, Clone)]
pub struct ClaimSource {
    pub source_id: Option<String>,
    pub locator: Option<String>,
    pub locator_label: Option<String>,
    pub page_number: Option<u32>,
    pub segment_index: Option<u32>,
    pub char_offset_start: Option<u32>,
    pub char_offset_end: Option<u32>,
    pub extraction_method: Option<String>,
    pub extraction_confidence: Option<f32>,
    pub properties: Option<Metadata>,
}

impl ClaimSource {
    /// A human-readable citation string for this source reference.
    pub fn citation_string(&self) -> Option<String> {
        if let Some(locator) = &self.locator {
            let s = locator.trim();
            if !s.is_empty() {
                return Some(s.to_string());
            }
        }
        if let (Some(p), Some(s)) = (self.page_number, self.segment_index) {
            return Some(format!("p.{} segment {}", p, s));
        }
        if let Some(s) = self.segment_index {
            return Some(format!("segment {}", s));
        }
        if let (Some(start), Some(end)) = (self.char_offset_start, self.char_offset_end) {
            return Some(format!("chars {}-{}", start, end));
        }
        None
    }
}

/// A graph entity (node).
#[derive(Debug, Clone)]
pub struct Entity {
    /// Auto-assigned stable identifier.
    pub id: u64,
    /// Human-readable name.
    pub name: String,
    /// Entity type label (e.g. `"Company"`, `"Person"`).
    pub entity_type: String,
    /// Arbitrary properties.
    pub properties: Metadata,
    /// Valid-time start.
    pub valid_from: Option<SystemTime>,
    /// Valid-time end.
    pub valid_to: Option<SystemTime>,
    /// Transaction-time start.
    pub system_from: Option<SystemTime>,
    /// Transaction-time end (set when superseded).
    pub system_to: Option<SystemTime>,
    /// ID of the record that supersedes this one.
    pub superseded_by: Option<u64>,
    /// Assertion state of this record.
    pub assertion_state: AssertionState,
    /// Decomposed confidence scores.
    pub confidence: ConfidenceScores,
}

impl Entity {
    /// Returns `true` if this entity is active at the given valid-time.
    pub fn active_at(&self, at: SystemTime) -> bool {
        if !self.assertion_state.is_active() {
            return false;
        }
        if let Some(from) = self.valid_from
            && at < from {
                return false;
            }
        if let Some(to) = self.valid_to
            && at >= to {
                return false;
            }
        true
    }

    /// Returns `true` if this entity is currently active.
    pub fn is_active(&self) -> bool {
        self.active_at(SystemTime::now())
    }

    /// Returns external refs stored in properties.
    pub fn external_refs(&self) -> Vec<ExternalRef> {
        let Some(refs_val) = self.properties.get(crate::graph::schema::EXTERNAL_REFS) else {
            return Vec::new();
        };
        let Some(arr) = refs_val.as_array() else {
            return Vec::new();
        };
        arr.iter()
            .filter_map(|v| {
                let obj = v.as_object()?;
                Some(ExternalRef {
                    source: obj.get("source")?.as_str()?.to_string(),
                    external_id: obj.get("external_id")?.as_str()?.to_string(),
                    confidence: obj.get("confidence").and_then(|v| v.as_f64()).map(|f| f as f32),
                    valid_from: None, // time deserialisation deferred to adapter layer
                    valid_to: None,
                    properties: None,
                })
            })
            .collect()
    }
}

/// A directed relation (edge) between two entities.
#[derive(Debug, Clone)]
pub struct Relation {
    /// Auto-assigned stable identifier.
    pub id: u64,
    /// Source entity id.
    pub from_id: u64,
    /// Target entity id.
    pub to_id: u64,
    /// Relation type label (e.g. `"CAUSES"`, `"RELATED_TO"`).
    pub relation_type: String,
    /// Arbitrary properties.
    pub properties: Metadata,
    /// Valid-time start.
    pub valid_from: Option<SystemTime>,
    /// Valid-time end.
    pub valid_to: Option<SystemTime>,
    /// Transaction-time start.
    pub system_from: Option<SystemTime>,
    /// Transaction-time end (set when superseded).
    pub system_to: Option<SystemTime>,
    /// ID of the record that supersedes this one.
    pub superseded_by: Option<u64>,
    /// Assertion state of this record.
    pub assertion_state: AssertionState,
    /// Decomposed confidence scores.
    pub confidence: ConfidenceScores,
}

impl Relation {
    /// Returns `true` if this relation is active at the given valid-time.
    pub fn active_at(&self, at: SystemTime) -> bool {
        if !self.assertion_state.is_active() {
            return false;
        }
        if let Some(from) = self.valid_from
            && at < from {
                return false;
            }
        if let Some(to) = self.valid_to
            && at >= to {
                return false;
            }
        true
    }

    /// Returns `true` if this relation type is a causal type.
    pub fn is_causal(&self) -> bool {
        crate::graph::schema::is_causal(&self.relation_type)
    }

    /// Returns `true` if this relation is an infrastructure relation.
    pub fn is_infrastructure(&self) -> bool {
        crate::graph::schema::is_infrastructure(&self.relation_type)
    }

    /// Returns the `causal_strength` property value if present.
    pub fn causal_strength(&self) -> Option<f32> {
        self.properties
            .get(crate::graph::schema::CAUSAL_STRENGTH)
            .and_then(|v| v.as_f64())
            .map(|f| f as f32)
    }
}

/// A path of causal edges through the graph.
/// Nodes are in causal order (cause first, effect last).
#[derive(Debug, Clone)]
pub struct CausalPath {
    /// Entities in causal order.
    pub nodes: Vec<Entity>,
    /// Relations connecting adjacent nodes.
    pub edges: Vec<Relation>,
    /// The relation type at each step.
    pub causal_types: Vec<String>,
    /// Geometric mean of causal_strength values along the path.
    pub chain_strength: f32,
    /// Whether the path reaches the intended target entity.
    pub is_complete: bool,
    /// Overall chain confidence computed by the `chain_confidence` algorithm.
    pub chain_confidence: Option<f32>,
    /// Per-hop confidence values.
    pub hop_confidences: Vec<f32>,
    /// Index of the weakest hop (lowest confidence).
    pub confidence_ceiling_hop: Option<usize>,
}

impl CausalPath {
    /// Number of hops (edges) in this path.
    pub fn hop_count(&self) -> usize {
        self.edges.len()
    }
}

/// A detected community (cluster) of entities.
#[derive(Debug, Clone)]
pub struct Community {
    /// The entities in this community.
    pub entities: Vec<Entity>,
    /// Edge density: actual edges / maximum possible edges.
    pub density: f64,
    /// The entity with the highest degree within the community.
    pub central_entity: Option<Entity>,
}

/// Statistics from a relation retargeting operation.
#[derive(Debug, Clone)]
pub struct RetargetStats {
    pub repointed: usize,
    pub self_loops_removed: usize,
    pub duplicates_merged: usize,
}

/// A changelog entry recording a state transition.
#[derive(Debug, Clone)]
pub struct ChangelogEntry {
    pub operation: String,
    pub record_type: String,
    pub record_id: u64,
    pub system_from: SystemTime,
    pub payload: HashMap<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- AssertionState ---

    #[test]
    fn active_state_is_active() {
        assert!(AssertionState::Active.is_active());
    }

    #[test]
    fn retracted_is_not_active() {
        assert!(!AssertionState::Retracted.is_active());
    }

    #[test]
    fn superseded_is_not_active() {
        assert!(!AssertionState::Superseded.is_active());
    }

    #[test]
    fn display_active() {
        assert_eq!(AssertionState::Active.to_string(), "active");
    }

    #[test]
    fn display_retracted() {
        assert_eq!(AssertionState::Retracted.to_string(), "retracted");
    }

    #[test]
    fn from_str_active() {
        assert_eq!(AssertionState::from_str("active"), AssertionState::Active);
    }

    #[test]
    fn from_str_retracted() {
        assert_eq!(AssertionState::from_str("retracted"), AssertionState::Retracted);
    }

    #[test]
    fn from_str_unknown_defaults_to_active() {
        assert_eq!(AssertionState::from_str("unknown"), AssertionState::Active);
    }

    // --- Entity::active_at ---

    fn entity(state: AssertionState, valid_from: Option<SystemTime>, valid_to: Option<SystemTime>) -> Entity {
        Entity {
            id: 1,
            name: "test".to_string(),
            entity_type: "Test".to_string(),
            properties: Default::default(),
            valid_from,
            valid_to,
            system_from: None,
            system_to: None,
            superseded_by: None,
            assertion_state: state,
            confidence: Default::default(),
        }
    }

    #[test]
    fn active_entity_no_time_bounds_is_always_active() {
        let e = entity(AssertionState::Active, None, None);
        assert!(e.is_active());
    }

    #[test]
    fn retracted_entity_is_not_active() {
        let e = entity(AssertionState::Retracted, None, None);
        assert!(!e.is_active());
    }

    #[test]
    fn entity_before_valid_from_is_not_active() {
        use std::time::Duration;
        let future = SystemTime::now() + Duration::from_secs(3600);
        let e = entity(AssertionState::Active, Some(future), None);
        assert!(!e.is_active());
    }

    #[test]
    fn entity_after_valid_to_is_not_active() {
        use std::time::Duration;
        let past = SystemTime::now() - Duration::from_secs(3600);
        let e = entity(AssertionState::Active, None, Some(past));
        assert!(!e.is_active());
    }

    // --- Relation helpers ---

    fn relation(rel_type: &str) -> Relation {
        Relation {
            id: 1,
            from_id: 1,
            to_id: 2,
            relation_type: rel_type.to_string(),
            properties: Default::default(),
            valid_from: None,
            valid_to: None,
            system_from: None,
            system_to: None,
            superseded_by: None,
            assertion_state: AssertionState::Active,
            confidence: Default::default(),
        }
    }

    #[test]
    fn causes_relation_is_causal() {
        assert!(relation("CAUSES").is_causal());
    }

    #[test]
    fn related_to_is_not_causal() {
        assert!(!relation("RELATED_TO").is_causal());
    }

    #[test]
    fn version_supersedes_is_infrastructure() {
        assert!(relation("__version_supersedes__").is_infrastructure());
    }

    #[test]
    fn causal_strength_returns_none_when_absent() {
        assert!(relation("CAUSES").causal_strength().is_none());
    }

    #[test]
    fn causal_strength_returns_value_when_present() {
        let mut r = relation("CAUSES");
        r.properties.insert("causal_strength".to_string(), serde_json::json!(0.8));
        assert!((r.causal_strength().unwrap() - 0.8).abs() < 1e-5);
    }

    // --- ExternalRef::active_at ---

    #[test]
    fn external_ref_no_bounds_is_active() {
        let r = ExternalRef {
            source: "wikidata".to_string(),
            external_id: "Q123".to_string(),
            confidence: Some(0.9),
            valid_from: None,
            valid_to: None,
            properties: None,
        };
        assert!(r.is_active());
    }

    #[test]
    fn external_ref_expired_is_not_active() {
        use std::time::Duration;
        let past = SystemTime::now() - Duration::from_secs(3600);
        let r = ExternalRef {
            source: "lei".to_string(),
            external_id: "X1".to_string(),
            confidence: None,
            valid_from: None,
            valid_to: Some(past),
            properties: None,
        };
        assert!(!r.is_active());
    }

    // --- ClaimSource::citation_string ---

    #[test]
    fn citation_string_prefers_locator() {
        let cs = ClaimSource {
            source_id: None,
            locator: Some("section 3.2".to_string()),
            locator_label: None,
            page_number: Some(5),
            segment_index: Some(2),
            char_offset_start: None,
            char_offset_end: None,
            extraction_method: None,
            extraction_confidence: None,
            properties: None,
        };
        assert_eq!(cs.citation_string(), Some("section 3.2".to_string()));
    }

    #[test]
    fn citation_string_falls_back_to_page_segment() {
        let cs = ClaimSource {
            source_id: None,
            locator: None,
            locator_label: None,
            page_number: Some(3),
            segment_index: Some(1),
            char_offset_start: None,
            char_offset_end: None,
            extraction_method: None,
            extraction_confidence: None,
            properties: None,
        };
        assert_eq!(cs.citation_string(), Some("p.3 segment 1".to_string()));
    }

    #[test]
    fn citation_string_none_when_no_info() {
        let cs = ClaimSource {
            source_id: None,
            locator: None,
            locator_label: None,
            page_number: None,
            segment_index: None,
            char_offset_start: None,
            char_offset_end: None,
            extraction_method: None,
            extraction_confidence: None,
            properties: None,
        };
        assert!(cs.citation_string().is_none());
    }

    // --- CausalPath ---

    #[test]
    fn hop_count_returns_edge_count() {
        let path = CausalPath {
            nodes: Vec::new(),
            edges: vec![relation("CAUSES"), relation("ENABLES")],
            causal_types: vec!["CAUSES".to_string(), "ENABLES".to_string()],
            chain_strength: 1.0,
            is_complete: true,
            chain_confidence: None,
            hop_confidences: Vec::new(),
            confidence_ceiling_hop: None,
        };
        assert_eq!(path.hop_count(), 2);
    }
}
