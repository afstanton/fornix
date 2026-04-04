//! Core ontology domain types.
//!
//! The central type is [`Definition`] — a complete, versioned ontology schema
//! that specifies the entity types, relation types, and property constraints
//! for a domain. All other types in this module are components of a
//! `Definition`.
//!
//! `Definition` is fully serialisable to/from JSON so it can be passed across
//! the Ruby–Rust boundary via the Magnus native extension.
//!
//! # Provenance
//!
//! Ontologies imported from external sources (OWL, SKOS, JSON-LD) carry full
//! import provenance in [`ImportProvenance`]. Authored or derived ontologies
//! have `provenance: None`.
//!
//! # Alias resolution
//!
//! [`Definition::resolve_alias`] checks all entity type aliases before falling
//! back to a direct name lookup, mirroring `Cortex::Ontology::Definition#resolve_alias`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Property definitions
// ─────────────────────────────────────────────────────────────────────────────

/// Validation rules for a property.
///
/// Both fields are optional and independently applied during validation.
/// An empty `ValidationRules` is always valid.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ValidationRules {
    /// A regular-expression pattern the value must match (string properties only).
    pub pattern: Option<String>,
    /// The exhaustive set of allowed values. Empty means unconstrained.
    pub allowed_values: Vec<String>,
}

impl ValidationRules {
    /// Returns `true` if no constraints are set.
    pub fn is_empty(&self) -> bool {
        self.pattern.is_none() && self.allowed_values.is_empty()
    }
}

/// The definition of a single property on an entity or relation type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PropertyDefinition {
    /// Property name (snake_case recommended, e.g. `"cfr_citation"`).
    pub name: String,
    /// Declared data type: `"string"`, `"integer"`, `"float"`, `"boolean"`,
    /// `"date"`, `"datetime"`, `"array"`, or `"object"`.
    pub data_type: String,
    /// Whether the property must be present on every instance of its owner type.
    pub required: bool,
    /// Optional validation constraints applied during extraction validation.
    #[serde(default)]
    pub validation_rules: ValidationRules,
}

impl PropertyDefinition {
    /// Construct a minimal required property with no validation rules.
    pub fn required(name: impl Into<String>, data_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            data_type: data_type.into(),
            required: true,
            validation_rules: ValidationRules::default(),
        }
    }

    /// Construct a minimal optional property with no validation rules.
    pub fn optional(name: impl Into<String>, data_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            data_type: data_type.into(),
            required: false,
            validation_rules: ValidationRules::default(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Extraction patterns
// ─────────────────────────────────────────────────────────────────────────────

/// A named regex-like pattern used to guide entity extraction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExtractionPattern {
    /// The pattern string (regex or descriptive heuristic).
    pub pattern: String,
    /// A human-readable description of what this pattern matches.
    pub description: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Entity type definition
// ─────────────────────────────────────────────────────────────────────────────

/// The definition of an entity type within an ontology.
///
/// An entity type describes what a node in the knowledge graph can be.
/// It carries guidance for LLM-based extraction (description, strategy,
/// patterns, aliases) and validation constraints (properties).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EntityTypeDefinition {
    /// Canonical type name (e.g. `"Regulation"`, `"Agency"`).
    pub name: String,
    /// Human-readable description, included in LLM extraction prompts.
    pub description: Option<String>,
    /// Preferred extraction strategy: `"llm"`, `"regex"`, or `"hybrid"`.
    pub extraction_strategy: Option<String>,
    /// Regex-like patterns that help identify instances of this type in text.
    #[serde(default)]
    pub extraction_patterns: Vec<ExtractionPattern>,
    /// Alternative names or surface forms (e.g. `"Provision"` for `"Regulation"`).
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Property constraints for this type.
    #[serde(default)]
    pub properties: Vec<PropertyDefinition>,
}

impl EntityTypeDefinition {
    /// Returns `true` if `candidate` matches the canonical name or any alias
    /// (case-insensitive).
    pub fn matches_name(&self, candidate: &str) -> bool {
        if self.name.eq_ignore_ascii_case(candidate) {
            return true;
        }
        self.aliases.iter().any(|a| a.eq_ignore_ascii_case(candidate))
    }

    /// Iterator over the canonical name followed by all aliases.
    pub fn all_names(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.name.as_str())
            .chain(self.aliases.iter().map(String::as_str))
    }

    /// Look up a property by name (case-insensitive).
    pub fn property(&self, name: &str) -> Option<&PropertyDefinition> {
        self.properties
            .iter()
            .find(|p| p.name.eq_ignore_ascii_case(name))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Relation type definition
// ─────────────────────────────────────────────────────────────────────────────

/// The definition of a relation type within an ontology.
///
/// A relation type describes what a directed edge in the knowledge graph
/// can be. Source and target type constraints, when non-empty, are enforced
/// during validation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RelationTypeDefinition {
    /// Canonical type name in SCREAMING_SNAKE_CASE (e.g. `"ISSUED_BY"`).
    pub name: String,
    /// Human-readable description, included in LLM extraction prompts.
    pub description: Option<String>,
    /// Entity type names allowed as the source (domain) of this relation.
    /// Empty means unconstrained.
    #[serde(default)]
    pub source_types: Vec<String>,
    /// Entity type names allowed as the target (range) of this relation.
    /// Empty means unconstrained.
    #[serde(default)]
    pub target_types: Vec<String>,
    /// Property constraints for this relation type.
    #[serde(default)]
    pub properties: Vec<PropertyDefinition>,
}

impl RelationTypeDefinition {
    /// Returns `true` if `entity_type` is an allowed source type, or if no
    /// source constraint is set.
    pub fn valid_source(&self, entity_type: &str) -> bool {
        self.source_types.is_empty()
            || self.source_types.iter().any(|t| t.eq_ignore_ascii_case(entity_type))
    }

    /// Returns `true` if `entity_type` is an allowed target type, or if no
    /// target constraint is set.
    pub fn valid_target(&self, entity_type: &str) -> bool {
        self.target_types.is_empty()
            || self.target_types.iter().any(|t| t.eq_ignore_ascii_case(entity_type))
    }

    /// Look up a property by name (case-insensitive).
    pub fn property(&self, name: &str) -> Option<&PropertyDefinition> {
        self.properties
            .iter()
            .find(|p| p.name.eq_ignore_ascii_case(name))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Import provenance
// ─────────────────────────────────────────────────────────────────────────────

/// Provenance captured when an ontology is imported from an external source.
///
/// Authored or derived ontologies have `provenance: None` on their
/// [`Definition`]. The provenance cannot be reconstructed after import —
/// it is captured at import time and persisted alongside the definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ImportProvenance {
    /// The canonical URI of the source ontology (e.g. `owl:versionIRI`).
    pub source_uri: Option<String>,
    /// The import format: `"turtle"`, `"rdf_xml"`, `"skos"`, `"jsonld"`,
    /// `"yaml"`, `"json"`.
    pub source_format: Option<String>,
    /// The source ontology version string (from `owl:versionInfo`).
    pub source_version: Option<String>,
    /// ISO 8601 timestamp of the import.
    pub imported_at: Option<String>,
    /// Number of OWL/SKOS classes successfully imported as entity types.
    pub imported_classes: usize,
    /// Number of properties imported.
    pub imported_properties: usize,
    /// Number of axioms that had no `Definition` equivalent and were skipped.
    pub skipped_axioms: usize,
    /// Human-readable descriptions of each skipped axiom.
    #[serde(default)]
    pub skipped_axioms_detail: Vec<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Definition
// ─────────────────────────────────────────────────────────────────────────────

/// A complete, versioned ontology definition.
///
/// `Definition` is the central type in the ontology module. It is fully
/// serialisable and can be round-tripped through JSON for the Ruby–Rust
/// boundary. Import provenance is optional — authored or derived definitions
/// carry `provenance: None`.
///
/// # Version source of truth
///
/// `Definition::version` is the canonical version. The registry's `version`
/// parameter is an override/fallback only. See the Plans document for the
/// full version source-of-truth rules.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Definition {
    /// The canonical name used to identify this ontology in the registry.
    pub name: String,
    /// The canonical version string. `None` until explicitly assigned.
    pub version: Option<String>,
    /// The domain this ontology covers (freeform, e.g. `"regulatory"`).
    pub domain: Option<String>,
    /// A brief description of this ontology's scope and purpose.
    pub description: Option<String>,
    /// Entity type definitions.
    #[serde(default)]
    pub entity_types: Vec<EntityTypeDefinition>,
    /// Relation type definitions.
    #[serde(default)]
    pub relation_types: Vec<RelationTypeDefinition>,
    /// Import provenance. `None` for authored or derived ontologies.
    pub provenance: Option<ImportProvenance>,
    /// Arbitrary extra metadata (e.g. materialization info).
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Definition {
    /// Construct a minimal named definition with no types and no version.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: None,
            domain: None,
            description: None,
            entity_types: Vec::new(),
            relation_types: Vec::new(),
            provenance: None,
            metadata: HashMap::new(),
        }
    }

    /// Look up an entity type definition by canonical name (case-insensitive).
    ///
    /// Does **not** search aliases — use [`resolve_alias`] for that.
    pub fn entity_type(&self, name: &str) -> Option<&EntityTypeDefinition> {
        self.entity_types
            .iter()
            .find(|t| t.name.eq_ignore_ascii_case(name))
    }

    /// Look up a relation type definition by canonical name (case-insensitive).
    pub fn relation_type(&self, name: &str) -> Option<&RelationTypeDefinition> {
        self.relation_types
            .iter()
            .find(|t| t.name.eq_ignore_ascii_case(name))
    }

    /// Resolve an alias or canonical name to the canonical
    /// [`EntityTypeDefinition`].
    ///
    /// Checks aliases first, then falls back to canonical name lookup.
    /// Returns `None` if neither matches.
    pub fn resolve_alias(&self, candidate: &str) -> Option<&EntityTypeDefinition> {
        // Alias search first
        self.entity_types
            .iter()
            .find(|t| {
                t.aliases
                    .iter()
                    .any(|a| a.eq_ignore_ascii_case(candidate))
            })
            // Canonical name fallback
            .or_else(|| self.entity_type(candidate))
    }

    /// Iterator over all canonical entity type names.
    pub fn entity_type_names(&self) -> impl Iterator<Item = &str> {
        self.entity_types.iter().map(|t| t.name.as_str())
    }

    /// Iterator over all canonical relation type names.
    pub fn relation_type_names(&self) -> impl Iterator<Item = &str> {
        self.relation_types.iter().map(|t| t.name.as_str())
    }

    /// Serialise to a compact JSON string for the Ruby boundary.
    pub fn to_json(&self) -> crate::ontology::error::Result<String> {
        serde_json::to_string(self)
            .map_err(|e| crate::ontology::error::Error::serialisation(e.to_string()))
    }

    /// Deserialise from a JSON string produced by [`to_json`] or the Ruby layer.
    pub fn from_json(json: &str) -> crate::ontology::error::Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| crate::ontology::error::Error::serialisation(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regulatory_definition() -> Definition {
        Definition {
            name: "regulatory".to_string(),
            version: Some("1.0.0".to_string()),
            domain: Some("regulatory".to_string()),
            description: Some("US federal regulatory ontology".to_string()),
            entity_types: vec![
                EntityTypeDefinition {
                    name: "Regulation".to_string(),
                    description: Some("A codified rule".to_string()),
                    extraction_strategy: Some("llm".to_string()),
                    extraction_patterns: vec![ExtractionPattern {
                        pattern: r"\d+ CFR".to_string(),
                        description: Some("CFR citation".to_string()),
                    }],
                    aliases: vec!["Provision".to_string(), "Rule".to_string()],
                    properties: vec![
                        PropertyDefinition::required("cfr_citation", "string"),
                        PropertyDefinition::optional("effective_date", "date"),
                    ],
                },
                EntityTypeDefinition {
                    name: "Agency".to_string(),
                    description: Some("A federal agency".to_string()),
                    extraction_strategy: None,
                    extraction_patterns: Vec::new(),
                    aliases: Vec::new(),
                    properties: vec![PropertyDefinition::optional("acronym", "string")],
                },
            ],
            relation_types: vec![RelationTypeDefinition {
                name: "ISSUED_BY".to_string(),
                description: Some("Regulation issued by Agency".to_string()),
                source_types: vec!["Regulation".to_string()],
                target_types: vec!["Agency".to_string()],
                properties: vec![PropertyDefinition::optional("authority_citation", "string")],
            }],
            provenance: None,
            metadata: HashMap::new(),
        }
    }

    // ── ValidationRules ──

    #[test]
    fn validation_rules_default_is_empty() {
        assert!(ValidationRules::default().is_empty());
    }

    #[test]
    fn validation_rules_with_pattern_is_not_empty() {
        let r = ValidationRules { pattern: Some(".*".to_string()), allowed_values: Vec::new() };
        assert!(!r.is_empty());
    }

    #[test]
    fn validation_rules_with_allowed_values_is_not_empty() {
        let r = ValidationRules { pattern: None, allowed_values: vec!["a".to_string()] };
        assert!(!r.is_empty());
    }

    // ── PropertyDefinition ──

    #[test]
    fn required_property_is_required() {
        let p = PropertyDefinition::required("cfr_citation", "string");
        assert!(p.required);
        assert_eq!(p.name, "cfr_citation");
        assert_eq!(p.data_type, "string");
    }

    #[test]
    fn optional_property_is_not_required() {
        let p = PropertyDefinition::optional("effective_date", "date");
        assert!(!p.required);
    }

    // ── EntityTypeDefinition ──

    #[test]
    fn entity_type_matches_canonical_name() {
        let def = regulatory_definition();
        let e = def.entity_type("Regulation").unwrap();
        assert!(e.matches_name("Regulation"));
    }

    #[test]
    fn entity_type_matches_name_case_insensitively() {
        let def = regulatory_definition();
        let e = def.entity_type("Regulation").unwrap();
        assert!(e.matches_name("regulation"));
        assert!(e.matches_name("REGULATION"));
    }

    #[test]
    fn entity_type_matches_alias() {
        let def = regulatory_definition();
        let e = def.entity_type("Regulation").unwrap();
        assert!(e.matches_name("Provision"));
        assert!(e.matches_name("provision"));
    }

    #[test]
    fn entity_type_does_not_match_unrelated() {
        let def = regulatory_definition();
        let e = def.entity_type("Regulation").unwrap();
        assert!(!e.matches_name("Agency"));
    }

    #[test]
    fn all_names_yields_canonical_then_aliases() {
        let def = regulatory_definition();
        let e = def.entity_type("Regulation").unwrap();
        let names: Vec<&str> = e.all_names().collect();
        assert_eq!(names[0], "Regulation");
        assert!(names.contains(&"Provision"));
        assert!(names.contains(&"Rule"));
    }

    #[test]
    fn entity_type_property_lookup_case_insensitive() {
        let def = regulatory_definition();
        let e = def.entity_type("Regulation").unwrap();
        assert!(e.property("cfr_citation").is_some());
        assert!(e.property("CFR_CITATION").is_some());
        assert!(e.property("nonexistent").is_none());
    }

    // ── RelationTypeDefinition ──

    #[test]
    fn valid_source_empty_is_unconstrained() {
        let r = RelationTypeDefinition {
            name: "RELATED_TO".to_string(),
            description: None,
            source_types: Vec::new(),
            target_types: Vec::new(),
            properties: Vec::new(),
        };
        assert!(r.valid_source("AnythingAtAll"));
        assert!(r.valid_target("AnythingAtAll"));
    }

    #[test]
    fn valid_source_constrained() {
        let def = regulatory_definition();
        let r = def.relation_type("ISSUED_BY").unwrap();
        assert!(r.valid_source("Regulation"));
        assert!(!r.valid_source("Agency"));
    }

    #[test]
    fn valid_target_constrained() {
        let def = regulatory_definition();
        let r = def.relation_type("ISSUED_BY").unwrap();
        assert!(r.valid_target("Agency"));
        assert!(!r.valid_target("Regulation"));
    }

    #[test]
    fn valid_source_case_insensitive() {
        let def = regulatory_definition();
        let r = def.relation_type("ISSUED_BY").unwrap();
        assert!(r.valid_source("regulation"));
        assert!(r.valid_source("REGULATION"));
    }

    // ── Definition lookup ──

    #[test]
    fn entity_type_found_by_canonical_name() {
        let def = regulatory_definition();
        assert!(def.entity_type("Regulation").is_some());
        assert!(def.entity_type("Agency").is_some());
    }

    #[test]
    fn entity_type_not_found() {
        let def = regulatory_definition();
        assert!(def.entity_type("Nonexistent").is_none());
    }

    #[test]
    fn entity_type_lookup_case_insensitive() {
        let def = regulatory_definition();
        assert!(def.entity_type("regulation").is_some());
        assert!(def.entity_type("AGENCY").is_some());
    }

    #[test]
    fn relation_type_found() {
        let def = regulatory_definition();
        assert!(def.relation_type("ISSUED_BY").is_some());
    }

    #[test]
    fn relation_type_not_found() {
        let def = regulatory_definition();
        assert!(def.relation_type("NONEXISTENT").is_none());
    }

    // ── Alias resolution ──

    #[test]
    fn resolve_alias_returns_canonical_for_alias() {
        let def = regulatory_definition();
        let resolved = def.resolve_alias("Provision");
        assert!(resolved.is_some());
        assert_eq!(resolved.unwrap().name, "Regulation");
    }

    #[test]
    fn resolve_alias_case_insensitive() {
        let def = regulatory_definition();
        assert!(def.resolve_alias("provision").is_some());
        assert!(def.resolve_alias("PROVISION").is_some());
    }

    #[test]
    fn resolve_alias_falls_back_to_canonical() {
        let def = regulatory_definition();
        let resolved = def.resolve_alias("Regulation");
        assert!(resolved.is_some());
        assert_eq!(resolved.unwrap().name, "Regulation");
    }

    #[test]
    fn resolve_alias_returns_none_for_unknown() {
        let def = regulatory_definition();
        assert!(def.resolve_alias("Completely Unknown").is_none());
    }

    // ── Iterators ──

    #[test]
    fn entity_type_names_iterator() {
        let def = regulatory_definition();
        let names: Vec<&str> = def.entity_type_names().collect();
        assert!(names.contains(&"Regulation"));
        assert!(names.contains(&"Agency"));
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn relation_type_names_iterator() {
        let def = regulatory_definition();
        let names: Vec<&str> = def.relation_type_names().collect();
        assert!(names.contains(&"ISSUED_BY"));
        assert_eq!(names.len(), 1);
    }

    // ── JSON round-trip ──

    #[test]
    fn json_round_trip() {
        let def = regulatory_definition();
        let json = def.to_json().unwrap();
        let restored = Definition::from_json(&json).unwrap();
        assert_eq!(def, restored);
    }

    #[test]
    fn from_json_invalid_returns_error() {
        let result = Definition::from_json("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn json_preserves_version() {
        let def = regulatory_definition();
        let json = def.to_json().unwrap();
        let restored = Definition::from_json(&json).unwrap();
        assert_eq!(restored.version, Some("1.0.0".to_string()));
    }

    #[test]
    fn json_preserves_provenance() {
        let mut def = regulatory_definition();
        def.provenance = Some(ImportProvenance {
            source_uri: Some("http://example.com/onto".to_string()),
            source_format: Some("turtle".to_string()),
            source_version: Some("1.0".to_string()),
            imported_at: Some("2026-04-01T00:00:00Z".to_string()),
            imported_classes: 2,
            imported_properties: 3,
            skipped_axioms: 1,
            skipped_axioms_detail: vec!["rdfs:subClassOf (not modeled)".to_string()],
        });
        let json = def.to_json().unwrap();
        let restored = Definition::from_json(&json).unwrap();
        let prov = restored.provenance.unwrap();
        assert_eq!(prov.source_uri, Some("http://example.com/onto".to_string()));
        assert_eq!(prov.skipped_axioms, 1);
        assert_eq!(prov.skipped_axioms_detail.len(), 1);
    }

    // ── Definition::new ──

    #[test]
    fn new_definition_has_no_types() {
        let def = Definition::new("test");
        assert!(def.entity_types.is_empty());
        assert!(def.relation_types.is_empty());
        assert!(def.version.is_none());
        assert!(def.provenance.is_none());
    }
}
