//! Vector adapter configuration.

use crate::common::namespace::Namespace;
use crate::store::{
    config::AdapterConfig,
    error::{Error as StoreError, Result as StoreResult},
};

/// Configuration for a vector storage adapter.
#[derive(Debug, Clone)]
pub struct VectorConfig {
    /// The namespace used when the caller does not specify one.
    pub default_namespace: Namespace,

    /// Expected vector dimension. When set, all upserts and queries
    /// are validated against this value and an error is returned on mismatch.
    /// `None` disables dimension checking.
    pub dimension: Option<usize>,

    /// Name of the table or collection used to store vectors.
    /// Defaults to `"fornix_vectors"`.
    pub table_name: String,
}

impl Default for VectorConfig {
    fn default() -> Self {
        Self {
            default_namespace: Namespace::named("default"),
            dimension: None,
            table_name: "fornix_vectors".to_string(),
        }
    }
}

impl VectorConfig {
    /// Construct a config for a known embedding dimension.
    pub fn with_dimension(dim: usize) -> Self {
        Self {
            dimension: Some(dim),
            ..Default::default()
        }
    }

    /// Construct a config with a specific default namespace.
    pub fn with_namespace(ns: impl Into<String>) -> Self {
        Self {
            default_namespace: Namespace::named(ns),
            ..Default::default()
        }
    }

    /// Resolve the effective namespace: use the provided one if given,
    /// otherwise fall back to the configured default.
    pub fn resolve_namespace<'a>(&'a self, ns: Option<&'a Namespace>) -> &'a Namespace {
        ns.unwrap_or(&self.default_namespace)
    }

    /// Validate that a vector's dimension matches the configured dimension,
    /// if one has been set. Returns an error on mismatch.
    pub fn check_dimension(&self, actual: usize) -> StoreResult<()> {
        if let Some(expected) = self.dimension {
            if actual != expected {
                return Err(StoreError::DimensionMismatch { expected, actual });
            }
        }
        Ok(())
    }
}

impl AdapterConfig for VectorConfig {
    fn adapter_name(&self) -> &'static str {
        "vector"
    }

    fn validate(&self) -> StoreResult<()> {
        if self.default_namespace.is_default() {
            return Err(StoreError::config(
                "vector default_namespace must be a named namespace",
            ));
        }
        if let Some(dim) = self.dimension {
            if dim == 0 {
                return Err(StoreError::config("dimension must be greater than zero if set"));
            }
        }
        if self.table_name.is_empty() {
            return Err(StoreError::config("table_name must not be empty"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::error::Error as StoreError;

    #[test]
    fn default_has_named_namespace() {
        let c = VectorConfig::default();
        assert_eq!(c.default_namespace.as_deref(), Some("default"));
    }

    #[test]
    fn default_has_no_dimension() {
        assert!(VectorConfig::default().dimension.is_none());
    }

    #[test]
    fn default_table_name() {
        assert_eq!(VectorConfig::default().table_name, "fornix_vectors");
    }

    #[test]
    fn with_dimension_sets_dimension() {
        let c = VectorConfig::with_dimension(384);
        assert_eq!(c.dimension, Some(384));
    }

    #[test]
    fn with_namespace_sets_namespace() {
        let c = VectorConfig::with_namespace("docs");
        assert_eq!(c.default_namespace.as_deref(), Some("docs"));
    }

    #[test]
    fn adapter_name_is_vector() {
        assert_eq!(VectorConfig::default().adapter_name(), "vector");
    }

    #[test]
    fn validate_passes_for_valid_config() {
        assert!(VectorConfig::default().validate().is_ok());
    }

    #[test]
    fn validate_fails_for_bare_default_namespace() {
        let c = VectorConfig {
            default_namespace: Namespace::default_ns(),
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_fails_for_zero_dimension() {
        let c = VectorConfig {
            dimension: Some(0),
            ..Default::default()
        };
        let err = c.validate().unwrap_err();
        assert!(err.to_string().contains("dimension"));
    }

    #[test]
    fn validate_fails_for_empty_table_name() {
        let c = VectorConfig {
            table_name: String::new(),
            ..Default::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn resolve_namespace_returns_provided() {
        let c = VectorConfig::default();
        let ns = Namespace::named("override");
        assert_eq!(c.resolve_namespace(Some(&ns)), &ns);
    }

    #[test]
    fn resolve_namespace_falls_back_to_default() {
        let c = VectorConfig::with_namespace("default-ns");
        assert_eq!(c.resolve_namespace(None).as_deref(), Some("default-ns"));
    }

    #[test]
    fn check_dimension_passes_when_not_configured() {
        let c = VectorConfig::default();
        assert!(c.check_dimension(999).is_ok());
    }

    #[test]
    fn check_dimension_passes_when_matching() {
        let c = VectorConfig::with_dimension(384);
        assert!(c.check_dimension(384).is_ok());
    }

    #[test]
    fn check_dimension_fails_on_mismatch() {
        let c = VectorConfig::with_dimension(384);
        let err = c.check_dimension(512).unwrap_err();
        assert!(matches!(err, StoreError::DimensionMismatch { expected: 384, actual: 512 }));
    }
}
