//! Arbitrary key-value metadata attached to stored records.

use std::collections::HashMap;

/// Arbitrary JSON-typed metadata attached to a stored record.
///
/// Keys are always strings. Values are arbitrary JSON — strings, numbers,
/// booleans, arrays, objects, or null. This matches the open-hash pattern
/// used throughout the Ruby layer, while remaining fully serialisable.
pub type Metadata = HashMap<String, serde_json::Value>;

/// Construct a [`Metadata`] map from key-value pairs.
///
/// ```rust
/// use fornix::common::metadata::{Metadata, metadata};
///
/// let m: Metadata = metadata! {
///     "source" => "ingest",
///     "confidence" => 0.95_f64,
/// };
/// ```
#[macro_export]
macro_rules! metadata {
    ($($key:expr => $val:expr),* $(,)?) => {{
        #[allow(unused_mut)]
        let mut m = $crate::common::metadata::Metadata::new();
        $(m.insert($key.to_string(), serde_json::json!($val));)*
        m
    }};
}

pub use metadata;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_macro_produces_empty_map() {
        let m = metadata! {};
        assert!(m.is_empty());
    }

    #[test]
    fn macro_inserts_string_value() {
        let m = metadata! { "key" => "value" };
        assert_eq!(m.get("key"), Some(&json!("value")));
    }

    #[test]
    fn macro_inserts_numeric_value() {
        let m = metadata! { "score" => 0.95_f64 };
        assert_eq!(m.get("score"), Some(&json!(0.95_f64)));
    }

    #[test]
    fn macro_inserts_boolean_value() {
        let m = metadata! { "active" => true };
        assert_eq!(m.get("active"), Some(&json!(true)));
    }

    #[test]
    fn macro_inserts_multiple_values() {
        let m = metadata! {
            "source" => "ingest",
            "version" => 2_i64,
            "verified" => false,
        };
        assert_eq!(m.len(), 3);
        assert_eq!(m.get("source"), Some(&json!("ingest")));
        assert_eq!(m.get("version"), Some(&json!(2_i64)));
        assert_eq!(m.get("verified"), Some(&json!(false)));
    }

    #[test]
    fn macro_later_value_overwrites_earlier_for_duplicate_key() {
        let m = metadata! {
            "key" => "first",
            "key" => "second",
        };
        assert_eq!(m.get("key"), Some(&json!("second")));
    }

    #[test]
    fn metadata_is_a_standard_hashmap() {
        let mut m: Metadata = HashMap::new();
        m.insert("x".to_string(), json!(1));
        assert_eq!(m["x"], json!(1));
    }
}
