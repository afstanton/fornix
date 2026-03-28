//! Cache configuration.

use crate::store::{config::AdapterConfig, error::{Error as StoreError, Result as StoreResult}};
use crate::common::namespace::Namespace;

/// Configuration for a cache adapter.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// The namespace used when the caller does not specify one explicitly.
    pub default_namespace: Namespace,

    /// Default TTL applied when the caller does not supply one.
    /// `None` means entries do not expire by default.
    pub default_ttl: Option<std::time::Duration>,

    /// Maximum number of entries allowed in the cache.
    /// `None` means no limit (only meaningful for in-process adapters).
    pub max_entries: Option<usize>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            default_namespace: Namespace::named("default"),
            default_ttl: None,
            max_entries: None,
        }
    }
}

impl CacheConfig {
    /// Construct a config with the given default namespace.
    pub fn with_namespace(ns: impl Into<String>) -> Self {
        Self {
            default_namespace: Namespace::named(ns),
            ..Default::default()
        }
    }
}

impl AdapterConfig for CacheConfig {
    fn adapter_name(&self) -> &'static str {
        "cache"
    }

    fn validate(&self) -> StoreResult<()> {
        if self.default_namespace.is_default() {
            return Err(StoreError::config(
                "cache default_namespace must be a named namespace",
            ));
        }
        if let Some(max) = self.max_entries {
            if max == 0 {
                return Err(StoreError::config("max_entries must be greater than zero if set"));
            }
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
        let c = CacheConfig::default();
        assert!(!c.default_namespace.is_default());
        assert_eq!(c.default_namespace.as_deref(), Some("default"));
    }

    #[test]
    fn default_has_no_ttl() {
        assert!(CacheConfig::default().default_ttl.is_none());
    }

    #[test]
    fn default_has_no_max_entries() {
        assert!(CacheConfig::default().max_entries.is_none());
    }

    #[test]
    fn with_namespace_sets_namespace() {
        let c = CacheConfig::with_namespace("embeddings");
        assert_eq!(c.default_namespace.as_deref(), Some("embeddings"));
    }

    #[test]
    fn adapter_name_is_cache() {
        assert_eq!(CacheConfig::default().adapter_name(), "cache");
    }

    #[test]
    fn validate_passes_for_valid_config() {
        assert!(CacheConfig::default().validate().is_ok());
    }

    #[test]
    fn validate_fails_for_bare_default_namespace() {
        let c = CacheConfig {
            default_namespace: Namespace::default_ns(),
            ..Default::default()
        };
        let err = c.validate().unwrap_err();
        assert!(matches!(err, StoreError::Configuration(_)));
        assert!(err.to_string().contains("named namespace"));
    }

    #[test]
    fn validate_fails_for_zero_max_entries() {
        let c = CacheConfig {
            max_entries: Some(0),
            ..Default::default()
        };
        let err = c.validate().unwrap_err();
        assert!(matches!(err, StoreError::Configuration(_)));
        assert!(err.to_string().contains("max_entries"));
    }

    #[test]
    fn validate_passes_with_explicit_max_entries() {
        let c = CacheConfig { max_entries: Some(1000), ..Default::default() };
        assert!(c.validate().is_ok());
    }

    #[test]
    fn validate_passes_with_ttl() {
        let c = CacheConfig {
            default_ttl: Some(std::time::Duration::from_secs(300)),
            ..Default::default()
        };
        assert!(c.validate().is_ok());
    }
}
