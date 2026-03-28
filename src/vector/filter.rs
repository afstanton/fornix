//! Metadata filter type for vector search queries.

use crate::common::metadata::Metadata;

/// A set of key-value constraints applied to record metadata during search.
///
/// A record matches a filter when *all* constraints hold — i.e. filters
/// are conjunctive (AND). A value of `None` or an empty map matches everything.
///
/// ```rust
/// use fornix::vector::filter::MetadataFilter;
///
/// let filter = MetadataFilter::new()
///     .with("source", "ingest")
///     .with("verified", true);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MetadataFilter(Metadata);

impl MetadataFilter {
    /// Create an empty filter (matches everything).
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a constraint. `value` must be serialisable to `serde_json::Value`.
    pub fn with(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.0.insert(key.into(), value.into());
        self
    }

    /// Returns `true` if the filter has no constraints.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns `true` if the given metadata satisfies all constraints.
    pub fn matches(&self, metadata: &Metadata) -> bool {
        if self.0.is_empty() {
            return true;
        }
        self.0.iter().all(|(k, v)| metadata.get(k) == Some(v))
    }

    /// Borrow the inner metadata map.
    pub fn as_metadata(&self) -> &Metadata {
        &self.0
    }

    /// Consume the filter into its inner metadata map.
    pub fn into_metadata(self) -> Metadata {
        self.0
    }
}

impl From<Metadata> for MetadataFilter {
    fn from(m: Metadata) -> Self {
        Self(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn meta(pairs: &[(&str, serde_json::Value)]) -> Metadata {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn empty_filter_matches_anything() {
        let f = MetadataFilter::new();
        assert!(f.is_empty());
        assert!(f.matches(&meta(&[("key", json!("value"))])));
        assert!(f.matches(&Metadata::new()));
    }

    #[test]
    fn single_constraint_matches() {
        let f = MetadataFilter::new().with("source", json!("ingest"));
        assert!(f.matches(&meta(&[("source", json!("ingest"))])));
    }

    #[test]
    fn single_constraint_does_not_match_wrong_value() {
        let f = MetadataFilter::new().with("source", json!("ingest"));
        assert!(!f.matches(&meta(&[("source", json!("other"))])));
    }

    #[test]
    fn single_constraint_does_not_match_missing_key() {
        let f = MetadataFilter::new().with("source", json!("ingest"));
        assert!(!f.matches(&Metadata::new()));
    }

    #[test]
    fn multiple_constraints_all_must_match() {
        let f = MetadataFilter::new()
            .with("source", json!("ingest"))
            .with("verified", json!(true));

        let full = meta(&[("source", json!("ingest")), ("verified", json!(true))]);
        let partial = meta(&[("source", json!("ingest"))]);
        let wrong = meta(&[("source", json!("ingest")), ("verified", json!(false))]);

        assert!(f.matches(&full));
        assert!(!f.matches(&partial));
        assert!(!f.matches(&wrong));
    }

    #[test]
    fn extra_metadata_keys_are_ignored() {
        let f = MetadataFilter::new().with("a", json!(1));
        let m = meta(&[("a", json!(1)), ("b", json!(2)), ("c", json!(3))]);
        assert!(f.matches(&m));
    }

    #[test]
    fn from_metadata_map() {
        let m: Metadata = [("k".to_string(), json!("v"))].into_iter().collect();
        let f = MetadataFilter::from(m.clone());
        assert!(f.matches(&m));
    }

    #[test]
    fn as_metadata_borrows_inner() {
        let f = MetadataFilter::new().with("x", json!(1));
        assert_eq!(f.as_metadata().get("x"), Some(&json!(1)));
    }

    #[test]
    fn into_metadata_consumes() {
        let f = MetadataFilter::new().with("y", json!(2));
        let m = f.into_metadata();
        assert_eq!(m.get("y"), Some(&json!(2)));
    }

    #[test]
    fn is_empty_false_when_has_constraints() {
        let f = MetadataFilter::new().with("k", json!("v"));
        assert!(!f.is_empty());
    }
}
