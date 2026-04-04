//! Configuration for the ontology module.

use crate::store::{
    config::AdapterConfig,
    error::Result as StoreResult,
};

/// Configuration for ontology-aware graph validation.
///
/// When an ontology is active on a [`GraphAdapter`], this configuration
/// controls how violations are handled when `create_entity` or
/// `create_relation` is called with a type not in the schema.
#[derive(Debug, Clone, Default)]
pub struct OntologyConfig {
    /// When `true`, a type violation on a graph write raises an error.
    /// When `false` (the default), a violation is logged as a warning and
    /// the write proceeds.
    pub strict: bool,
}

impl OntologyConfig {
    /// A strict config that rejects type violations at write time.
    pub fn strict() -> Self {
        Self { strict: true }
    }

    /// A soft config that logs violations but does not block writes.
    pub fn soft() -> Self {
        Self { strict: false }
    }
}

impl AdapterConfig for OntologyConfig {
    fn adapter_name(&self) -> &'static str {
        "ontology"
    }

    fn validate(&self) -> StoreResult<()> {
        Ok(())
    }
}

/// Configuration for the materialization strategy when combining ontologies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MaterializationStrategy {
    /// Union of all types from all sources; same-name types are merged.
    #[default]
    Union,
    /// Only types present in every source ontology.
    Intersection,
    /// Union of all types; first-listed source wins for same-name conflicts
    /// without property merging.
    Precedence,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_not_strict() {
        assert!(!OntologyConfig::default().strict);
    }

    #[test]
    fn strict_constructor() {
        assert!(OntologyConfig::strict().strict);
    }

    #[test]
    fn soft_constructor() {
        assert!(!OntologyConfig::soft().strict);
    }

    #[test]
    fn adapter_name() {
        assert_eq!(OntologyConfig::default().adapter_name(), "ontology");
    }

    #[test]
    fn validate_always_passes() {
        assert!(OntologyConfig::default().validate().is_ok());
        assert!(OntologyConfig::strict().validate().is_ok());
    }

    #[test]
    fn materialization_strategy_default_is_union() {
        assert_eq!(MaterializationStrategy::default(), MaterializationStrategy::Union);
    }

    #[test]
    fn materialization_strategies_are_distinct() {
        assert_ne!(MaterializationStrategy::Union, MaterializationStrategy::Intersection);
        assert_ne!(MaterializationStrategy::Union, MaterializationStrategy::Precedence);
        assert_ne!(MaterializationStrategy::Intersection, MaterializationStrategy::Precedence);
    }
}
