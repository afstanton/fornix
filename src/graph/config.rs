//! Graph adapter configuration.

use crate::store::{
    config::AdapterConfig,
    error::{Error as StoreError, Result as StoreResult},
};

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
}
