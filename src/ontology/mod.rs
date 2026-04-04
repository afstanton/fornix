//! Versioned ontology schemas for domain-aware knowledge graph extraction.
//!
//! This module is the Rust equivalent of the `cortex-ontology` Ruby gem.
//! It provides the core types needed on the hot extraction path — type
//! lookup, alias resolution, validation, and prompt construction — without
//! the import/export machinery (OWL, SKOS, JSON-LD parsers), which stays
//! in the Ruby layer.
//!
//! # What lives here
//!
//! - [`types::Definition`] — complete, versioned ontology schema
//! - [`types::EntityTypeDefinition`] / [`types::RelationTypeDefinition`] — type specs
//! - [`types::PropertyDefinition`] — property constraints
//! - [`registry::OntologyRegistry`] trait + [`registry::MemoryOntologyRegistry`]
//! - [`validator::OntologyValidator`] — entity/relation validation
//! - [`prompt::OntologyPrompt`] — LLM extraction prompt construction
//! - [`alignment`] — cross-ontology alignment types (SKOS vocabulary, match records)
//!
//! # What stays in Ruby
//!
//! Import/export (OWL, SKOS, JSON-LD, YAML), storage adapters (Postgres, File),
//! the evolution/proposal workflow, and the alignment computation algorithm
//! all remain in `cortex-ontology`. Those paths are not on the extraction hot
//! path and depend on Ruby gems with no useful Rust equivalent.
//!
//! # Ruby native extension boundary
//!
//! [`Definition`] serialises to/from JSON via [`Definition::to_json`] and
//! [`Definition::from_json`], so it can be marshalled across the Magnus
//! boundary. The Ruby gem checks `Cortex::Ontology::Fornix.available?` and
//! delegates hot-path calls to the native extension when present, falling
//! back to the pure-Ruby implementation otherwise.
//!
//! # Integration with `graph` and `graphrag`
//!
//! When the `graph` feature is also enabled, [`crate::graph::GraphConfig`]
//! accepts an `ontology: Option<Arc<Definition>>` and an `ontology_strict`
//! flag. Violations raise [`crate::graph::Error::OntologyViolation`] in strict
//! mode or log a warning in soft mode.
//!
//! The `graphrag` module always depends on `ontology`. [`crate::graphrag::GraphRagConfig`]
//! accepts `ontology: Option<Arc<Definition>>`; when set, the flat
//! `entity_types` and `relation_types` lists are derived from it.
//!
//! # Quick start
//!
//! ```rust
//! use fornix::ontology::types::{Definition, EntityTypeDefinition, PropertyDefinition};
//! use fornix::ontology::validator::OntologyValidator;
//! use fornix::ontology::prompt::OntologyPrompt;
//! use fornix::common::metadata::Metadata;
//!
//! let mut def = Definition::new("regulatory");
//! def.version = Some("1.0.0".to_string());
//! def.entity_types.push(EntityTypeDefinition {
//!     name: "Agency".to_string(),
//!     description: Some("A regulatory agency.".to_string()),
//!     extraction_strategy: None,
//!     extraction_patterns: Vec::new(),
//!     aliases: vec!["Bureau".to_string()],
//!     properties: vec![PropertyDefinition::optional("acronym", "string")],
//! });
//!
//! // Validate
//! let v = OntologyValidator::new(&def);
//! assert!(v.known_entity_type("Agency"));
//! assert!(v.known_entity_type("Bureau")); // alias
//! assert_eq!(v.canonical_entity_type("Bureau"), Some("Agency"));
//!
//! // Build a prompt fragment
//! let prompt = OntologyPrompt::build_entity_prompt(&def, "Agency").unwrap();
//! assert!(prompt.contains("Agency"));
//! assert!(prompt.contains("Bureau"));
//! ```

pub mod alignment;
pub mod config;
pub mod error;
pub mod prompt;
pub mod registry;
pub mod types;
pub mod validator;

pub use alignment::{AlignmentKind, AlignmentMatch, SkosRelation};
pub use config::{MaterializationStrategy, OntologyConfig};
pub use error::{Error, Result};
pub use prompt::OntologyPrompt;
pub use registry::{MemoryOntologyRegistry, OntologyRegistry};
pub use types::{
    Definition, EntityTypeDefinition, ExtractionPattern, ImportProvenance,
    PropertyDefinition, RelationTypeDefinition, ValidationRules,
};
pub use validator::{OntologyValidator, ValidationError, ValidationResult};
