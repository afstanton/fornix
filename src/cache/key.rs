//! Cache key construction via SHA-256 hashing.
//!
//! A `CacheKey` is a stable, deterministic hex string derived from an
//! operation name, a model identifier, a list of inputs, and an optional
//! set of parameters. Identical inputs always produce the same key;
//! any difference in inputs produces a different key.

use sha2::{Digest, Sha256};

/// A computed cache key represented as a hex-encoded SHA-256 digest.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey(String);

impl CacheKey {
    /// Build a cache key from structured inputs.
    ///
    /// - `operation` — the kind of operation being cached (e.g. `"embed"`, `"completion"`)
    /// - `model` — the model or provider identifier (e.g. `"qwen3-4b"`)
    /// - `inputs` — the primary input strings (e.g. the texts being embedded)
    /// - `params` — optional key-value parameters that affect the output
    ///   (e.g. `[("temperature", "0.7")]`). Params are sorted by key before
    ///   hashing so that order does not affect the key.
    pub fn build(
        operation: &str,
        model: &str,
        inputs: &[impl AsRef<str>],
        params: &[(impl AsRef<str>, impl AsRef<str>)],
    ) -> Self {
        let mut hasher = Sha256::new();

        // Use a null byte as separator to prevent collisions between
        // adjacent fields (e.g. "ab" + "c" vs "a" + "bc").
        hasher.update(operation.as_bytes());
        hasher.update(b"\x00");
        hasher.update(model.as_bytes());
        hasher.update(b"\x00");

        for input in inputs {
            hasher.update(input.as_ref().as_bytes());
            hasher.update(b"\x00");
        }

        // Sort params by key for deterministic ordering.
        let mut sorted_params: Vec<(&str, &str)> = params
            .iter()
            .map(|(k, v)| (k.as_ref(), v.as_ref()))
            .collect();
        sorted_params.sort_by_key(|(k, _)| *k);

        for (k, v) in sorted_params {
            hasher.update(k.as_bytes());
            hasher.update(b"=");
            hasher.update(v.as_bytes());
            hasher.update(b"\x00");
        }

        let digest = hasher.finalize();
        Self(hex::encode(digest))
    }

    /// Returns the key as a hex string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the key and returns the inner hex string.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl std::fmt::Display for CacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for CacheKey {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(op: &str, model: &str, inputs: &[&str], params: &[(&str, &str)]) -> CacheKey {
        CacheKey::build(op, model, inputs, params)
    }

    #[test]
    fn same_inputs_produce_same_key() {
        let a = key("embed", "qwen3", &["hello world"], &[]);
        let b = key("embed", "qwen3", &["hello world"], &[]);
        assert_eq!(a, b);
    }

    #[test]
    fn different_input_text_produces_different_key() {
        let a = key("embed", "qwen3", &["hello"], &[]);
        let b = key("embed", "qwen3", &["world"], &[]);
        assert_ne!(a, b);
    }

    #[test]
    fn different_operation_produces_different_key() {
        let a = key("embed", "qwen3", &["text"], &[]);
        let b = key("completion", "qwen3", &["text"], &[]);
        assert_ne!(a, b);
    }

    #[test]
    fn different_model_produces_different_key() {
        let a = key("embed", "qwen3", &["text"], &[]);
        let b = key("embed", "phi3", &["text"], &[]);
        assert_ne!(a, b);
    }

    #[test]
    fn params_order_does_not_affect_key() {
        let a = key("embed", "m", &["t"], &[("b", "2"), ("a", "1")]);
        let b = key("embed", "m", &["t"], &[("a", "1"), ("b", "2")]);
        assert_eq!(a, b);
    }

    #[test]
    fn different_param_value_produces_different_key() {
        let a = key("embed", "m", &["t"], &[("temp", "0.7")]);
        let b = key("embed", "m", &["t"], &[("temp", "0.9")]);
        assert_ne!(a, b);
    }

    #[test]
    fn empty_inputs_is_valid() {
        let k = key("embed", "m", &[], &[]);
        assert!(!k.as_str().is_empty());
    }

    #[test]
    fn multiple_inputs_are_order_sensitive() {
        let a = key("embed", "m", &["a", "b"], &[]);
        let b = key("embed", "m", &["b", "a"], &[]);
        assert_ne!(a, b);
    }

    #[test]
    fn key_is_64_hex_chars_sha256() {
        let k = key("op", "model", &["input"], &[]);
        assert_eq!(k.as_str().len(), 64);
        assert!(k.as_str().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn display_matches_as_str() {
        let k = key("op", "m", &["x"], &[]);
        assert_eq!(k.to_string(), k.as_str());
    }

    #[test]
    fn as_ref_str() {
        let k = key("op", "m", &["x"], &[]);
        let s: &str = k.as_ref();
        assert_eq!(s, k.as_str());
    }

    #[test]
    fn into_string_consumes_key() {
        let k = key("op", "m", &["x"], &[]);
        let expected = k.as_str().to_string();
        let s = k.into_string();
        assert_eq!(s, expected);
    }

    #[test]
    fn adjacent_field_collision_prevention() {
        // "ab" + "c" vs "a" + "bc" should produce different keys
        // because we use null byte separators between fields.
        let a = key("ab", "c", &["input"], &[]);
        let b = key("a", "bc", &["input"], &[]);
        assert_ne!(a, b);
    }
}
