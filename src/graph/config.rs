//! Graph adapter configuration.

use crate::store::{
    config::AdapterConfig,
    error::{Error as StoreError, Result as StoreResult},
};

#[cfg(feature = "ontology")]
use std::sync::Arc;

/// Which community detection algorithm to use by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CommunityAlgorithm {
    /// Simple DFS connected-components (fast, no dependencies).
    #[default]
    ConnectedComponents,
    /// Leiden algorithm (higher quality, requires native extension).
    Leiden,
}

/// Configuration for graph adapters.
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Whether temporal versioning (valid-time / system-time) is enabled.
    /// Default: true.
    pub temporal_enabled: bool,

    /// Whether `valid_from` defaults to the current time when not specified.
    /// Default: false (open-ended valid_from).
    pub default_valid_from_now: bool,

    /// Whether to record a changelog of all state transitions.
    /// Default: true.
    pub changelog_enabled: bool,

    /// When true, read operations filter to currently-active records by default.
    /// When false, all records including retracted/superseded ones are returned
    /// unless the caller explicitly requests current-only reads.
    /// Default: false.
    pub current_state_reads: bool,

    /// Default community detection algorithm.
    pub default_community_algorithm: CommunityAlgorithm,

    /// Maximum depth for causal path traversal.
    /// Default: 7.
    pub max_causal_depth: usize,

    /// Maximum paths returned by causal traversal.
    /// Default: 20.
    pub max_causal_paths: usize,

    /// The active ontology definition used for type validation on writes.
    ///
    /// When `Some`, `create_entity` and `create_relation` validate the
    /// supplied type against the ontology. When `None`, all type names are
    /// accepted (current behaviour unchanged).
    ///
    /// Only available when the `ontology` feature is enabled.
    #[cfg(feature = "ontology")]
    pub ontology: Option<Arc<crate::ontology::Definition>>,

    /// When `true` and an ontology is set, type violations on graph writes
    /// raise [`crate::graph::Error::OntologyViolation`]. When `false`
    /// (the default), violations are logged as warnings and writes proceed.
    ///
    /// Only meaningful when `ontology` is `Some` and the `ontology` feature
    /// is enabled.
    #[cfg(feature = "ontology")]
    pub ontology_strict: bool,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            temporal_enabled: true,
            default_valid_from_now: false,
            changelog_enabled: true,
            current_state_reads: false,
            default_community_algorithm: CommunityAlgorithm::ConnectedComponents,
            max_causal_depth: 7,
            max_causal_paths: 20,
            #[cfg(feature = "ontology")]
            ontology: None,
            #[cfg(feature = "ontology")]
            ontology_strict: false,
        }
    }
}

impl GraphConfig {
    /// Construct a minimal config with temporal features enabled.
    pub fn temporal() -> Self {
        Self::default()
    }

    /// Construct a config with temporal features disabled
    /// (simpler writes, no valid-time filtering).
    pub fn without_temporal() -> Self {
        Self {
            temporal_enabled: false,
            ..Default::default()
        }
    }

    /// Validate an entity type against the configured ontology, if any.
    ///
    /// Returns `Ok(canonical_type)` — the canonical type name after alias
    /// resolution. If no ontology is configured, `entity_type` is returned
    /// unchanged.
    ///
    /// When `ontology_strict` is `true` a type violation returns
    /// `Err(OntologyViolation)`. When soft, the violation is logged via
    /// `tracing::warn!` and `Ok(entity_type)` is returned so the write
    /// can proceed.
    ///
    /// Only compiled when the `ontology` feature is enabled.
    #[cfg(feature = "ontology")]
    pub fn validate_entity_type<'a>(
        &self,
        entity_type: &'a str,
    ) -> crate::graph::error::Result<std::borrow::Cow<'a, str>> {
        let Some(ontology) = &self.ontology else {
            return Ok(std::borrow::Cow::Borrowed(entity_type));
        };

        let validator = crate::ontology::OntologyValidator::new(ontology);
        if let Some(canonical) = validator.canonical_entity_type(entity_type) {
            return Ok(std::borrow::Cow::Owned(canonical.to_string()));
        }

        let msg = format!("unknown ontology entity type: {}", entity_type);
        if self.ontology_strict {
            Err(crate::graph::Error::ontology_violation(msg))
        } else {
            tracing::warn!("[fornix::graph] {}", msg);
            Ok(std::borrow::Cow::Borrowed(entity_type))
        }
    }

    /// Validate a relation type against the configured ontology, if any.
    ///
    /// Returns `Ok(())` on success. Raises or warns on violation per
    /// `ontology_strict`.
    ///
    /// Only compiled when the `ontology` feature is enabled.
    #[cfg(feature = "ontology")]
    pub fn validate_relation_type(
        &self,
        relation_type: &str,
        source_entity_type: &str,
        target_entity_type: &str,
    ) -> crate::graph::error::Result<()> {
        let Some(ontology) = &self.ontology else {
            return Ok(());
        };

        let validator = crate::ontology::OntologyValidator::new(ontology);
        let result =
            validator.validate_relation(relation_type, source_entity_type, target_entity_type, &Default::default());

        if result.is_valid() {
            return Ok(());
        }

        let msg = result.error_messages().join("; ");
        if self.ontology_strict {
            Err(crate::graph::Error::ontology_violation(msg))
        } else {
            tracing::warn!("[fornix::graph] {}", msg);
            Ok(())
        }
    }
}

impl AdapterConfig for GraphConfig {
    fn adapter_name(&self) -> &'static str {
        "graph"
    }

    fn validate(&self) -> StoreResult<()> {
        if self.max_causal_depth == 0 {
            return Err(StoreError::config("max_causal_depth must be greater than zero"));
        }
        if self.max_causal_paths == 0 {
            return Err(StoreError::config("max_causal_paths must be greater than zero"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_temporal_enabled() {
        assert!(GraphConfig::default().temporal_enabled);
    }

    #[test]
    fn default_changelog_enabled() {
        assert!(GraphConfig::default().changelog_enabled);
    }

    #[test]
    fn default_current_state_reads_is_false() {
        assert!(!GraphConfig::default().current_state_reads);
    }

    #[test]
    fn default_valid_from_now_is_false() {
        assert!(!GraphConfig::default().default_valid_from_now);
    }

    #[test]
    fn default_max_causal_depth() {
        assert_eq!(GraphConfig::default().max_causal_depth, 7);
    }

    #[test]
    fn default_max_causal_paths() {
        assert_eq!(GraphConfig::default().max_causal_paths, 20);
    }

    #[test]
    fn adapter_name_is_graph() {
        assert_eq!(GraphConfig::default().adapter_name(), "graph");
    }

    #[test]
    fn validate_passes_for_defaults() {
        assert!(GraphConfig::default().validate().is_ok());
    }

    #[test]
    fn validate_fails_for_zero_causal_depth() {
        let c = GraphConfig { max_causal_depth: 0, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_fails_for_zero_causal_paths() {
        let c = GraphConfig { max_causal_paths: 0, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn without_temporal_disables_temporal() {
        assert!(!GraphConfig::without_temporal().temporal_enabled);
    }

    #[test]
    fn community_algorithm_default_is_connected_components() {
        assert_eq!(
            GraphConfig::default().default_community_algorithm,
            CommunityAlgorithm::ConnectedComponents
        );
    }

    #[cfg(feature = "ontology")]
    #[test]
    fn ontology_default_is_none() {
        assert!(GraphConfig::default().ontology.is_none());
    }

    #[cfg(feature = "ontology")]
    #[test]
    fn ontology_strict_default_is_false() {
        assert!(!GraphConfig::default().ontology_strict);
    }

    #[cfg(feature = "ontology")]
    #[test]
    fn validate_entity_type_no_ontology_passes_through() {
        let config = GraphConfig::default(); // ontology: None
        let result = config.validate_entity_type("Anything").unwrap();
        assert_eq!(result.as_ref(), "Anything");
    }

    #[cfg(feature = "ontology")]
    #[test]
    fn validate_entity_type_known_type_returns_canonical() {
        use crate::ontology::types::{Definition, EntityTypeDefinition};
        use std::sync::Arc;

        let mut def = Definition::new("test");
        def.version = Some("1.0".to_string());
        def.entity_types.push(EntityTypeDefinition {
            name: "Regulation".to_string(),
            description: None,
            extraction_strategy: None,
            extraction_patterns: Vec::new(),
            aliases: vec!["Provision".to_string()],
            properties: Vec::new(),
        });

        let config = GraphConfig {
            ontology: Some(Arc::new(def)),
            ontology_strict: false,
            ..Default::default()
        };

        let canonical = config.validate_entity_type("Provision").unwrap();
        assert_eq!(canonical.as_ref(), "Regulation");
    }

    #[cfg(feature = "ontology")]
    #[test]
    fn validate_entity_type_unknown_strict_raises() {
        use crate::ontology::types::Definition;
        use crate::graph::Error;
        use std::sync::Arc;

        let mut def = Definition::new("test");
        def.version = Some("1.0".to_string());

        let config = GraphConfig {
            ontology: Some(Arc::new(def)),
            ontology_strict: true,
            ..Default::default()
        };

        let err = config.validate_entity_type("Unknown").unwrap_err();
        assert!(matches!(err, Error::OntologyViolation(_)));
    }

    #[cfg(feature = "ontology")]
    #[test]
    fn validate_entity_type_unknown_soft_passes_through() {
        use crate::ontology::types::Definition;
        use std::sync::Arc;

        let mut def = Definition::new("test");
        def.version = Some("1.0".to_string());

        let config = GraphConfig {
            ontology: Some(Arc::new(def)),
            ontology_strict: false,
            ..Default::default()
        };

        // Soft mode: unknown type logs a warning but write proceeds
        let result = config.validate_entity_type("Unknown").unwrap();
        assert_eq!(result.as_ref(), "Unknown");
    }
}
