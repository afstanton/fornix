//! Ontology validation for extracted entities and relations.
//!
//! [`OntologyValidator`] checks extracted data against a [`Definition`] and
//! returns a [`ValidationResult`] collecting all errors rather than failing
//! fast. This mirrors `Cortex::Ontology::Validator` in Ruby.
//!
//! # Usage
//!
//! ```rust
//! use fornix::ontology::types::{Definition, EntityTypeDefinition, PropertyDefinition};
//! use fornix::ontology::validator::OntologyValidator;
//! use fornix::common::metadata::Metadata;
//!
//! let mut def = Definition::new("regulatory");
//! def.entity_types.push(EntityTypeDefinition {
//!     name: "Agency".to_string(),
//!     description: None,
//!     extraction_strategy: None,
//!     extraction_patterns: Vec::new(),
//!     aliases: Vec::new(),
//!     properties: vec![PropertyDefinition::required("acronym", "string")],
//! });
//! def.version = Some("1.0".to_string());
//!
//! let validator = OntologyValidator::new(&def);
//! let mut props = Metadata::new();
//! props.insert("acronym".to_string(), serde_json::json!("EPA"));
//!
//! let result = validator.validate_entity("Agency", &props);
//! assert!(result.is_valid());
//! ```

use crate::common::metadata::Metadata;
use crate::ontology::types::Definition;

// ─────────────────────────────────────────────────────────────────────────────
// Validation errors
// ─────────────────────────────────────────────────────────────────────────────

/// A single ontology validation failure.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// The entity type is not defined in the ontology.
    UnknownEntityType(String),
    /// The relation type is not defined in the ontology.
    UnknownRelationType(String),
    /// The source entity type is not allowed for this relation.
    InvalidSourceType { relation: String, got: String },
    /// The target entity type is not allowed for this relation.
    InvalidTargetType { relation: String, got: String },
    /// A required property is absent from the extracted properties.
    MissingRequiredProperty { owner: String, property: String },
    /// A property value failed a pattern constraint.
    PatternMismatch { property: String, pattern: String },
    /// A property value is not in the allowed-values set.
    AllowedValuesMismatch { property: String, allowed: Vec<String> },
}

impl ValidationError {
    /// A human-readable description of this error.
    pub fn message(&self) -> String {
        match self {
            Self::UnknownEntityType(t) => format!("unknown entity type: {}", t),
            Self::UnknownRelationType(t) => format!("unknown relation type: {}", t),
            Self::InvalidSourceType { relation, got } => {
                format!("relation {} rejects source type {}", relation, got)
            }
            Self::InvalidTargetType { relation, got } => {
                format!("relation {} rejects target type {}", relation, got)
            }
            Self::MissingRequiredProperty { owner, property } => {
                format!("missing required property {} on {}", property, owner)
            }
            Self::PatternMismatch { property, pattern } => {
                format!("property {} does not match pattern {}", property, pattern)
            }
            Self::AllowedValuesMismatch { property, allowed } => {
                format!(
                    "property {} value not in allowed set: [{}]",
                    property,
                    allowed.join(", ")
                )
            }
        }
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation result
// ─────────────────────────────────────────────────────────────────────────────

/// The result of validating an entity or relation against an ontology.
///
/// Collects all validation errors rather than failing at the first one.
#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    pub errors: Vec<ValidationError>,
}

impl ValidationResult {
    /// Returns `true` if no validation errors were found.
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Collect all error messages as strings.
    pub fn error_messages(&self) -> Vec<String> {
        self.errors.iter().map(|e| e.message()).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Validator
// ─────────────────────────────────────────────────────────────────────────────

/// Validates extracted entities and relations against an ontology definition.
///
/// Borrows the [`Definition`] for the duration of validation calls. Construct
/// a new validator per-call or hold one alongside an `Arc<Definition>`.
pub struct OntologyValidator<'a> {
    ontology: &'a Definition,
}

impl<'a> OntologyValidator<'a> {
    /// Construct a validator for `ontology`.
    pub fn new(ontology: &'a Definition) -> Self {
        Self { ontology }
    }

    /// Returns `true` if `entity_type` is a known type (by canonical name or alias).
    pub fn known_entity_type(&self, entity_type: &str) -> bool {
        self.ontology.resolve_alias(entity_type).is_some()
    }

    /// Returns `true` if `relation_type` is a known type.
    pub fn known_relation_type(&self, relation_type: &str) -> bool {
        self.ontology.relation_type(relation_type).is_some()
    }

    /// Resolve an alias or canonical name to the canonical type name.
    ///
    /// Returns `None` if the type is not in the ontology.
    pub fn canonical_entity_type(&self, candidate: &str) -> Option<&str> {
        self.ontology.resolve_alias(candidate).map(|t| t.name.as_str())
    }

    /// Validate an extracted entity against the ontology.
    ///
    /// Checks:
    /// 1. The entity type is known (by canonical name or alias).
    /// 2. All required properties are present in `properties`.
    /// 3. Property values satisfy validation rules (pattern, allowed_values).
    pub fn validate_entity(
        &self,
        entity_type: &str,
        properties: &Metadata,
    ) -> ValidationResult {
        let mut result = ValidationResult::default();

        let type_def = match self.ontology.resolve_alias(entity_type) {
            Some(t) => t,
            None => {
                result.errors.push(ValidationError::UnknownEntityType(entity_type.to_string()));
                return result;
            }
        };

        self.check_properties(&type_def.name, &type_def.properties, properties, &mut result);
        result
    }

    /// Validate an extracted relation against the ontology.
    ///
    /// Checks:
    /// 1. The relation type is known.
    /// 2. `source_type` is an allowed source (if the relation defines source constraints).
    /// 3. `target_type` is an allowed target (if the relation defines target constraints).
    /// 4. All required properties are present.
    /// 5. Property values satisfy validation rules.
    pub fn validate_relation(
        &self,
        relation_type: &str,
        source_type: &str,
        target_type: &str,
        properties: &Metadata,
    ) -> ValidationResult {
        let mut result = ValidationResult::default();

        let rel_def = match self.ontology.relation_type(relation_type) {
            Some(r) => r,
            None => {
                result.errors.push(ValidationError::UnknownRelationType(relation_type.to_string()));
                return result;
            }
        };

        if !rel_def.valid_source(source_type) {
            result.errors.push(ValidationError::InvalidSourceType {
                relation: relation_type.to_string(),
                got: source_type.to_string(),
            });
        }

        if !rel_def.valid_target(target_type) {
            result.errors.push(ValidationError::InvalidTargetType {
                relation: relation_type.to_string(),
                got: target_type.to_string(),
            });
        }

        self.check_properties(&rel_def.name, &rel_def.properties, properties, &mut result);
        result
    }

    // ── private ──────────────────────────────────────────────────────────────

    fn check_properties(
        &self,
        owner: &str,
        property_defs: &[crate::ontology::types::PropertyDefinition],
        properties: &Metadata,
        result: &mut ValidationResult,
    ) {
        for prop_def in property_defs {
            let value = properties.get(&prop_def.name);

            // Required presence check
            if prop_def.required && value.is_none() {
                result.errors.push(ValidationError::MissingRequiredProperty {
                    owner: owner.to_string(),
                    property: prop_def.name.clone(),
                });
                continue; // can't validate value if absent
            }

            let Some(value) = value else { continue };

            // Allowed values check
            if !prop_def.validation_rules.allowed_values.is_empty() {
                let str_value = value.as_str().unwrap_or("");
                if !prop_def
                    .validation_rules
                    .allowed_values
                    .iter()
                    .any(|av| av == str_value)
                {
                    result.errors.push(ValidationError::AllowedValuesMismatch {
                        property: prop_def.name.clone(),
                        allowed: prop_def.validation_rules.allowed_values.clone(),
                    });
                }
            }

            // Pattern check (string values only)
            if let Some(pattern) = &prop_def.validation_rules.pattern
                && let Some(str_value) = value.as_str()
                && let Ok(re) = regex::Regex::new(pattern)
                && !re.is_match(str_value)
            {
                result.errors.push(ValidationError::PatternMismatch {
                    property: prop_def.name.clone(),
                    pattern: pattern.clone(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::types::{
        Definition, EntityTypeDefinition, ExtractionPattern, PropertyDefinition,
        RelationTypeDefinition, ValidationRules,
    };

    fn regulatory_definition() -> Definition {
        let mut def = Definition::new("regulatory");
        def.version = Some("1.0.0".to_string());
        def.entity_types = vec![
            EntityTypeDefinition {
                name: "Regulation".to_string(),
                description: None,
                extraction_strategy: None,
                extraction_patterns: vec![ExtractionPattern {
                    pattern: r"\d+ CFR".to_string(),
                    description: None,
                }],
                aliases: vec!["Provision".to_string()],
                properties: vec![
                    PropertyDefinition {
                        name: "cfr_citation".to_string(),
                        data_type: "string".to_string(),
                        required: true,
                        validation_rules: ValidationRules {
                            pattern: Some(r"^\d+ CFR".to_string()),
                            allowed_values: Vec::new(),
                        },
                    },
                    PropertyDefinition::optional("status", "string"),
                ],
            },
            EntityTypeDefinition {
                name: "Agency".to_string(),
                description: None,
                extraction_strategy: None,
                extraction_patterns: Vec::new(),
                aliases: Vec::new(),
                properties: vec![PropertyDefinition {
                    name: "kind".to_string(),
                    data_type: "string".to_string(),
                    required: false,
                    validation_rules: ValidationRules {
                        pattern: None,
                        allowed_values: vec!["federal".to_string(), "state".to_string()],
                    },
                }],
            },
        ];
        def.relation_types = vec![RelationTypeDefinition {
            name: "ISSUED_BY".to_string(),
            description: None,
            source_types: vec!["Regulation".to_string()],
            target_types: vec!["Agency".to_string()],
            properties: vec![PropertyDefinition::required("authority_citation", "string")],
        }];
        def
    }

    fn props(pairs: &[(&str, serde_json::Value)]) -> Metadata {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    // ── known_entity_type / known_relation_type ──

    #[test]
    fn known_entity_type_canonical() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        assert!(v.known_entity_type("Regulation"));
        assert!(v.known_entity_type("Agency"));
        assert!(!v.known_entity_type("Unknown"));
    }

    #[test]
    fn known_entity_type_via_alias() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        assert!(v.known_entity_type("Provision")); // alias of Regulation
    }

    #[test]
    fn known_relation_type() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        assert!(v.known_relation_type("ISSUED_BY"));
        assert!(!v.known_relation_type("NONEXISTENT"));
    }

    // ── canonical_entity_type ──

    #[test]
    fn canonical_resolves_alias() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        assert_eq!(v.canonical_entity_type("Provision"), Some("Regulation"));
    }

    #[test]
    fn canonical_resolves_name() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        assert_eq!(v.canonical_entity_type("Regulation"), Some("Regulation"));
    }

    #[test]
    fn canonical_returns_none_for_unknown() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        assert!(v.canonical_entity_type("Nonexistent").is_none());
    }

    // ── validate_entity — type check ──

    #[test]
    fn entity_unknown_type_produces_error() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        let result = v.validate_entity("Unicorn", &Metadata::new());
        assert!(!result.is_valid());
        assert!(matches!(result.errors[0], ValidationError::UnknownEntityType(_)));
    }

    #[test]
    fn entity_known_type_with_all_required_props() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        let p = props(&[("cfr_citation", serde_json::json!("42 CFR 11.1"))]);
        let result = v.validate_entity("Regulation", &p);
        assert!(result.is_valid(), "errors: {:?}", result.error_messages());
    }

    #[test]
    fn entity_alias_is_accepted() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        let p = props(&[("cfr_citation", serde_json::json!("42 CFR 11.1"))]);
        let result = v.validate_entity("Provision", &p);
        assert!(result.is_valid(), "errors: {:?}", result.error_messages());
    }

    // ── validate_entity — missing required property ──

    #[test]
    fn entity_missing_required_property() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        let result = v.validate_entity("Regulation", &Metadata::new());
        assert!(!result.is_valid());
        assert!(result.errors.iter().any(|e| matches!(
            e,
            ValidationError::MissingRequiredProperty { property, .. } if property == "cfr_citation"
        )));
    }

    // ── validate_entity — pattern mismatch ──

    #[test]
    fn entity_pattern_mismatch() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        let p = props(&[("cfr_citation", serde_json::json!("not-a-cfr"))]);
        let result = v.validate_entity("Regulation", &p);
        assert!(!result.is_valid());
        assert!(result.errors.iter().any(|e| matches!(
            e,
            ValidationError::PatternMismatch { property, .. } if property == "cfr_citation"
        )));
    }

    #[test]
    fn entity_pattern_match_passes() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        let p = props(&[("cfr_citation", serde_json::json!("42 CFR 11.1"))]);
        let result = v.validate_entity("Regulation", &p);
        assert!(result.is_valid(), "errors: {:?}", result.error_messages());
    }

    // ── validate_entity — allowed values ──

    #[test]
    fn entity_allowed_value_passes() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        let p = props(&[("kind", serde_json::json!("federal"))]);
        let result = v.validate_entity("Agency", &p);
        assert!(result.is_valid(), "errors: {:?}", result.error_messages());
    }

    #[test]
    fn entity_disallowed_value_fails() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        let p = props(&[("kind", serde_json::json!("galactic"))]);
        let result = v.validate_entity("Agency", &p);
        assert!(!result.is_valid());
        assert!(result.errors.iter().any(|e| matches!(
            e,
            ValidationError::AllowedValuesMismatch { property, .. } if property == "kind"
        )));
    }

    // ── validate_relation ──

    #[test]
    fn relation_unknown_type_produces_error() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        let result = v.validate_relation("INVENTED_BY", "Regulation", "Agency", &Metadata::new());
        assert!(!result.is_valid());
        assert!(matches!(result.errors[0], ValidationError::UnknownRelationType(_)));
    }

    #[test]
    fn relation_valid_endpoints_and_required_prop() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        let p = props(&[("authority_citation", serde_json::json!("42 USC 7401"))]);
        let result = v.validate_relation("ISSUED_BY", "Regulation", "Agency", &p);
        assert!(result.is_valid(), "errors: {:?}", result.error_messages());
    }

    #[test]
    fn relation_wrong_source_type() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        let p = props(&[("authority_citation", serde_json::json!("x"))]);
        let result = v.validate_relation("ISSUED_BY", "Agency", "Agency", &p);
        assert!(!result.is_valid());
        assert!(result.errors.iter().any(|e| matches!(e, ValidationError::InvalidSourceType { .. })));
    }

    #[test]
    fn relation_wrong_target_type() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        let p = props(&[("authority_citation", serde_json::json!("x"))]);
        let result = v.validate_relation("ISSUED_BY", "Regulation", "Regulation", &p);
        assert!(!result.is_valid());
        assert!(result.errors.iter().any(|e| matches!(e, ValidationError::InvalidTargetType { .. })));
    }

    #[test]
    fn relation_missing_required_property() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        let result = v.validate_relation("ISSUED_BY", "Regulation", "Agency", &Metadata::new());
        assert!(!result.is_valid());
        assert!(result.errors.iter().any(|e| matches!(
            e,
            ValidationError::MissingRequiredProperty { property, .. } if property == "authority_citation"
        )));
    }

    // ── multiple errors collected ──

    #[test]
    fn multiple_errors_collected() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        // Wrong source, wrong target, missing required prop — all three should appear
        let result = v.validate_relation("ISSUED_BY", "Agency", "Regulation", &Metadata::new());
        assert!(result.errors.len() >= 3);
    }

    // ── ValidationResult helpers ──

    #[test]
    fn error_messages_nonempty_on_failure() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        let result = v.validate_entity("Ghost", &Metadata::new());
        assert!(!result.error_messages().is_empty());
    }

    #[test]
    fn valid_result_has_no_messages() {
        let def = regulatory_definition();
        let v = OntologyValidator::new(&def);
        let p = props(&[("cfr_citation", serde_json::json!("42 CFR 11.1"))]);
        let result = v.validate_entity("Regulation", &p);
        assert!(result.error_messages().is_empty());
    }

    // ── ValidationError display ──

    #[test]
    fn validation_error_display_unknown_entity_type() {
        let e = ValidationError::UnknownEntityType("Foo".to_string());
        assert!(e.to_string().contains("unknown entity type"));
        assert!(e.to_string().contains("Foo"));
    }
}
