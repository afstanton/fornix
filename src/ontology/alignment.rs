//! Cross-ontology alignment types.
//!
//! Alignment edges map concepts between registered ontologies using the
//! SKOS match vocabulary. Version tags record which registry versions the
//! alignment was computed against so staleness can be detected later.
//!
//! This module provides the pure data types only. Storage and computation
//! of alignment edges is handled by the Ruby layer (cortex-ontology).
//! The Rust types are used for:
//! - Deserialising alignment payloads passed across the Ruby boundary
//! - Materialisation logic inside a `MaterializedOntology`
//! - Filtering alignment matches in downstream modules

use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// The SKOS match relation vocabulary used for cross-ontology alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkosRelation {
    /// Concepts are exactly equivalent.
    ExactMatch,
    /// Concepts are closely but not exactly equivalent.
    CloseMatch,
    /// The source concept is broader (more general) than the target.
    BroadMatch,
    /// The source concept is narrower (more specific) than the target.
    NarrowMatch,
    /// Concepts are associatively related but not equivalent.
    RelatedMatch,
}

impl SkosRelation {
    /// Returns the string key used in storage and serialisation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ExactMatch => "exact_match",
            Self::CloseMatch => "close_match",
            Self::BroadMatch => "broad_match",
            Self::NarrowMatch => "narrow_match",
            Self::RelatedMatch => "related_match",
        }
    }



    /// Returns `true` for relations that indicate semantic equivalence
    /// (`exact_match` or `close_match`). Used by materialisation to decide
    /// which types to merge across source ontologies.
    pub fn is_equivalence(&self) -> bool {
        matches!(self, Self::ExactMatch | Self::CloseMatch)
    }
}

impl std::fmt::Display for SkosRelation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SkosRelation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "exact_match" => Ok(Self::ExactMatch),
            "close_match" => Ok(Self::CloseMatch),
            "broad_match" => Ok(Self::BroadMatch),
            "narrow_match" => Ok(Self::NarrowMatch),
            "related_match" => Ok(Self::RelatedMatch),
            _ => Err(format!("unknown SKOS relation: {}", s)),
        }
    }
}

/// Whether an alignment match applies to entity types or relation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentKind {
    Entity,
    Relation,
}

impl AlignmentKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Entity => "entity",
            Self::Relation => "relation",
        }
    }
}

impl FromStr for AlignmentKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "entity" => Ok(Self::Entity),
            "relation" => Ok(Self::Relation),
            _ => Err(format!("unknown alignment kind: {}", s)),
        }
    }
}

impl std::fmt::Display for AlignmentKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single cross-ontology alignment match.
///
/// Records which type in the source ontology corresponds to which type in
/// the target ontology, with what confidence and SKOS relation, computed
/// against which registry versions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AlignmentMatch {
    /// Whether this is an entity-type or relation-type alignment.
    pub kind: AlignmentKind,
    /// The type name in the source ontology.
    pub source_type: String,
    /// The type name in the target ontology.
    pub target_type: String,
    /// The SKOS match relation characterising the alignment.
    pub skos_relation: SkosRelation,
    /// Confidence score in [0.0, 1.0].
    pub score: f32,
    /// The alignment strategy used (e.g. `"name_similarity"`,
    /// `"embedding_similarity"`).
    pub strategy: String,
    /// The registry version of the source ontology at alignment time.
    pub source_version: Option<String>,
    /// The registry version of the target ontology at alignment time.
    pub target_version: Option<String>,
    /// The domain tag on this alignment edge.
    pub domain: Option<String>,
    /// Materialisation approval status.
    /// - `None` = unreviewed (included in materialisation by default)
    /// - `Some(true)` = explicitly approved
    /// - `Some(false)` = explicitly rejected (excluded from materialisation)
    pub approved: Option<bool>,
}

impl AlignmentMatch {
    /// Returns `true` if this match should be included in materialisation
    /// when `approved_only` is `false` (the default): all non-rejected edges.
    pub fn usable_for_materialization(&self) -> bool {
        self.approved != Some(false)
    }

    /// Returns `true` if this match is explicitly approved for materialisation.
    pub fn explicitly_approved(&self) -> bool {
        self.approved == Some(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── SkosRelation ──

    #[test]
    fn skos_relation_as_str() {
        assert_eq!(SkosRelation::ExactMatch.as_str(), "exact_match");
        assert_eq!(SkosRelation::CloseMatch.as_str(), "close_match");
        assert_eq!(SkosRelation::BroadMatch.as_str(), "broad_match");
        assert_eq!(SkosRelation::NarrowMatch.as_str(), "narrow_match");
        assert_eq!(SkosRelation::RelatedMatch.as_str(), "related_match");
    }

    #[test]
    fn skos_relation_from_str() {
        assert_eq!(SkosRelation::from_str("exact_match"), Ok(SkosRelation::ExactMatch));
        assert_eq!(SkosRelation::from_str("close_match"), Ok(SkosRelation::CloseMatch));
        assert_eq!(SkosRelation::from_str("related_match"), Ok(SkosRelation::RelatedMatch));
        assert!(SkosRelation::from_str("unknown").is_err());
    }

    #[test]
    fn skos_relation_is_equivalence() {
        assert!(SkosRelation::ExactMatch.is_equivalence());
        assert!(SkosRelation::CloseMatch.is_equivalence());
        assert!(!SkosRelation::BroadMatch.is_equivalence());
        assert!(!SkosRelation::NarrowMatch.is_equivalence());
        assert!(!SkosRelation::RelatedMatch.is_equivalence());
    }

    #[test]
    fn skos_relation_display() {
        assert_eq!(SkosRelation::NarrowMatch.to_string(), "narrow_match");
    }

    #[test]
    fn skos_relation_round_trips_via_json() {
        let r = SkosRelation::BroadMatch;
        let json = serde_json::to_string(&r).unwrap();
        let restored: SkosRelation = serde_json::from_str(&json).unwrap();
        assert_eq!(r, restored);
    }

    // ── AlignmentKind ──

    #[test]
    fn alignment_kind_as_str() {
        assert_eq!(AlignmentKind::Entity.as_str(), "entity");
        assert_eq!(AlignmentKind::Relation.as_str(), "relation");
    }

    #[test]
    fn alignment_kind_from_str() {
        assert_eq!(AlignmentKind::from_str("entity"), Ok(AlignmentKind::Entity));
        assert_eq!(AlignmentKind::from_str("relation"), Ok(AlignmentKind::Relation));
        assert!(AlignmentKind::from_str("other").is_err());
    }

    #[test]
    fn alignment_kind_display() {
        assert_eq!(AlignmentKind::Entity.to_string(), "entity");
    }

    #[test]
    fn alignment_kind_round_trips_via_json() {
        let k = AlignmentKind::Relation;
        let json = serde_json::to_string(&k).unwrap();
        let restored: AlignmentKind = serde_json::from_str(&json).unwrap();
        assert_eq!(k, restored);
    }

    // ── AlignmentMatch ──

    fn match_fixture(approved: Option<bool>) -> AlignmentMatch {
        AlignmentMatch {
            kind: AlignmentKind::Entity,
            source_type: "Regulation".to_string(),
            target_type: "Act".to_string(),
            skos_relation: SkosRelation::CloseMatch,
            score: 0.83,
            strategy: "embedding_similarity".to_string(),
            source_version: Some("1.0.0".to_string()),
            target_version: Some("1.0.0".to_string()),
            domain: Some("regulatory".to_string()),
            approved,
        }
    }

    #[test]
    fn unreviewed_match_is_usable() {
        assert!(match_fixture(None).usable_for_materialization());
    }

    #[test]
    fn approved_match_is_usable() {
        assert!(match_fixture(Some(true)).usable_for_materialization());
    }

    #[test]
    fn rejected_match_is_not_usable() {
        assert!(!match_fixture(Some(false)).usable_for_materialization());
    }

    #[test]
    fn explicitly_approved() {
        assert!(match_fixture(Some(true)).explicitly_approved());
        assert!(!match_fixture(None).explicitly_approved());
        assert!(!match_fixture(Some(false)).explicitly_approved());
    }

    #[test]
    fn alignment_match_round_trips_via_json() {
        let m = match_fixture(Some(true));
        let json = serde_json::to_string(&m).unwrap();
        let restored: AlignmentMatch = serde_json::from_str(&json).unwrap();
        assert_eq!(m, restored);
    }

    #[test]
    fn alignment_match_round_trips_with_null_approved() {
        let m = match_fixture(None);
        let json = serde_json::to_string(&m).unwrap();
        let restored: AlignmentMatch = serde_json::from_str(&json).unwrap();
        assert_eq!(m.approved, restored.approved);
    }
}
