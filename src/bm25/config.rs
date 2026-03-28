//! BM25 configuration.

use crate::store::{
    config::AdapterConfig,
    error::{Error as StoreError, Result as StoreResult},
};

/// English stop words filtered during tokenisation.
pub const DEFAULT_STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with",
];

/// Configuration for BM25 adapters.
#[derive(Debug, Clone)]
pub struct Bm25Config {
    /// BM25 k1 parameter — controls term frequency saturation.
    /// Typical values: 1.2–2.0. Default: 1.2.
    pub k1: f32,

    /// BM25 b parameter — controls document length normalisation.
    /// 0.0 = no normalisation, 1.0 = full normalisation. Default: 0.75.
    pub b: f32,

    /// Stop words removed during tokenisation.
    /// `None` disables stop word filtering.
    pub stop_words: Option<Vec<String>>,

    /// Minimum token length (inclusive). Tokens shorter than this are discarded.
    /// Default: 2.
    pub token_min_length: usize,

    /// Maximum token length (inclusive). Tokens longer than this are discarded.
    /// Default: 50.
    pub token_max_length: usize,

    /// HMAC-SHA256 key used to hash tokens before storage for privacy.
    ///
    /// When `Some`, all tokens are stored as `HMAC-SHA256(key, token)` hex
    /// digests rather than plaintext. The same key must be used consistently
    /// for indexing and querying.
    ///
    /// When `None`, tokens are stored as plaintext (simpler but less private).
    pub blind_index_key: Option<Vec<u8>>,

    /// Document fields indexed and searched. Empty means all text fields.
    /// For structured documents, this restricts indexing to named fields.
    pub fields: Vec<String>,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            stop_words: Some(DEFAULT_STOP_WORDS.iter().map(|s| s.to_string()).collect()),
            token_min_length: 2,
            token_max_length: 50,
            blind_index_key: None,
            fields: Vec::new(),
        }
    }
}

impl Bm25Config {
    /// Construct a config with English stop words and the given field list.
    pub fn with_fields(fields: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            fields: fields.into_iter().map(Into::into).collect(),
            ..Default::default()
        }
    }

    /// Disable stop word filtering entirely.
    pub fn without_stop_words(mut self) -> Self {
        self.stop_words = None;
        self
    }

    /// Set the blind index key for token privacy.
    pub fn with_blind_index_key(mut self, key: impl Into<Vec<u8>>) -> Self {
        self.blind_index_key = Some(key.into());
        self
    }
}

impl AdapterConfig for Bm25Config {
    fn adapter_name(&self) -> &'static str {
        "bm25"
    }

    fn validate(&self) -> StoreResult<()> {
        if self.k1 <= 0.0 {
            return Err(StoreError::config("k1 must be greater than zero"));
        }
        if !(0.0..=1.0).contains(&self.b) {
            return Err(StoreError::config("b must be in [0.0, 1.0]"));
        }
        if self.token_min_length == 0 {
            return Err(StoreError::config("token_min_length must be at least 1"));
        }
        if self.token_max_length < self.token_min_length {
            return Err(StoreError::config(
                "token_max_length must be >= token_min_length",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::error::Error as StoreError;

    #[test]
    fn default_k1_is_1_2() {
        assert!((Bm25Config::default().k1 - 1.2).abs() < 1e-6);
    }

    #[test]
    fn default_b_is_0_75() {
        assert!((Bm25Config::default().b - 0.75).abs() < 1e-6);
    }

    #[test]
    fn default_min_length_is_2() {
        assert_eq!(Bm25Config::default().token_min_length, 2);
    }

    #[test]
    fn default_max_length_is_50() {
        assert_eq!(Bm25Config::default().token_max_length, 50);
    }

    #[test]
    fn default_has_english_stop_words() {
        let words = Bm25Config::default().stop_words.unwrap();
        assert!(words.contains(&"the".to_string()));
        assert!(words.contains(&"and".to_string()));
    }

    #[test]
    fn default_has_no_blind_index_key() {
        assert!(Bm25Config::default().blind_index_key.is_none());
    }

    #[test]
    fn default_has_no_fields() {
        assert!(Bm25Config::default().fields.is_empty());
    }

    #[test]
    fn with_fields_sets_fields() {
        let c = Bm25Config::with_fields(["title", "body"]);
        assert_eq!(c.fields, vec!["title", "body"]);
    }

    #[test]
    fn without_stop_words_clears_them() {
        let c = Bm25Config::default().without_stop_words();
        assert!(c.stop_words.is_none());
    }

    #[test]
    fn with_blind_index_key_sets_key() {
        let c = Bm25Config::default().with_blind_index_key(b"secret".to_vec());
        assert!(c.blind_index_key.is_some());
    }

    #[test]
    fn adapter_name_is_bm25() {
        assert_eq!(Bm25Config::default().adapter_name(), "bm25");
    }

    #[test]
    fn validate_passes_for_valid_config() {
        assert!(Bm25Config::default().validate().is_ok());
    }

    #[test]
    fn validate_fails_for_zero_k1() {
        let c = Bm25Config { k1: 0.0, ..Default::default() };
        let err = c.validate().unwrap_err();
        assert!(matches!(err, StoreError::Configuration(_)));
        assert!(err.to_string().contains("k1"));
    }

    #[test]
    fn validate_fails_for_negative_k1() {
        let c = Bm25Config { k1: -1.0, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_fails_for_b_above_one() {
        let c = Bm25Config { b: 1.1, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_fails_for_b_below_zero() {
        let c = Bm25Config { b: -0.1, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_fails_for_zero_min_length() {
        let c = Bm25Config { token_min_length: 0, ..Default::default() };
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_fails_when_max_less_than_min() {
        let c = Bm25Config { token_min_length: 5, token_max_length: 3, ..Default::default() };
        assert!(c.validate().is_err());
    }
}
