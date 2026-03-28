//! BM25 scoring function (Okapi BM25).
//!
//! Pure mathematics — no I/O, no allocator dependency beyond the call stack.
//! The scorer is constructed once from configuration and reused across
//! all documents in a search pass.
//!
//! # Algorithm
//!
//! For each query term `t` present in document `d`:
//!
//! ```text
//! score += IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (|d| / avgdl)))
//! ```
//!
//! where:
//! - `tf`     = term frequency of `t` in `d`
//! - `|d|`    = length of `d` in tokens
//! - `avgdl`  = average document length in the corpus
//! - `IDF(t)` = ln((N - df + 0.5) / (df + 0.5) + 1)
//! - `N`      = corpus size (number of documents)
//! - `df`     = number of documents containing `t`

use crate::bm25::config::Bm25Config;

/// BM25 scorer parameterised by k1 and b.
#[derive(Debug, Clone, Copy)]
pub struct Scorer {
    /// Term frequency saturation parameter.
    k1: f32,
    /// Document length normalisation parameter.
    b: f32,
}

impl Scorer {
    /// Construct a scorer from a [`Bm25Config`].
    pub fn from_config(config: &Bm25Config) -> Self {
        Self { k1: config.k1, b: config.b }
    }

    /// Construct a scorer with explicit k1 and b values.
    pub fn new(k1: f32, b: f32) -> Self {
        Self { k1, b }
    }

    /// Compute the BM25 score for a single document.
    ///
    /// # Parameters
    /// - `query_terms` — tokens extracted from the query (duplicates are deduplicated internally)
    /// - `term_frequencies` — for each token: how many times it appears in the document
    /// - `doc_length` — total token count of the document
    /// - `avg_doc_length` — average token count across the corpus
    /// - `doc_frequencies` — for each token: how many documents in the corpus contain it
    /// - `corpus_size` — total number of documents in the corpus
    pub fn score(
        &self,
        query_terms: &[String],
        term_frequencies: &std::collections::HashMap<String, u32>,
        doc_length: u32,
        avg_doc_length: f32,
        doc_frequencies: &std::collections::HashMap<String, u32>,
        corpus_size: u32,
    ) -> f32 {
        if avg_doc_length < f32::EPSILON {
            return 0.0;
        }

        let mut seen = std::collections::HashSet::new();
        let mut total = 0.0_f32;

        for term in query_terms {
            if !seen.insert(term) {
                continue; // deduplicate query terms
            }
            let tf = match term_frequencies.get(term) {
                Some(&f) if f > 0 => f as f32,
                _ => continue,
            };
            let df = doc_frequencies.get(term).copied().unwrap_or(0) as f32;
            let idf = self.idf(df, corpus_size as f32);
            let norm = doc_length as f32 / avg_doc_length;
            let numerator = tf * (self.k1 + 1.0);
            let denominator = tf + self.k1 * (1.0 - self.b + self.b * norm);
            total += idf * (numerator / denominator);
        }

        total
    }

    /// Inverse document frequency (smooth IDF).
    ///
    /// `IDF(t) = ln((N - df + 0.5) / (df + 0.5) + 1.0)`
    ///
    /// Always returns a non-negative value. When `corpus_size` is zero, returns 0.0.
    pub fn idf(&self, doc_frequency: f32, corpus_size: f32) -> f32 {
        if corpus_size < f32::EPSILON {
            return 0.0;
        }
        let numerator = corpus_size - doc_frequency + 0.5;
        let denominator = doc_frequency + 0.5;
        ((numerator / denominator) + 1.0).ln().max(0.0)
    }
}

impl Default for Scorer {
    fn default() -> Self {
        Self::from_config(&Bm25Config::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn scorer() -> Scorer {
        Scorer::default()
    }

    fn tf(pairs: &[(&str, u32)]) -> HashMap<String, u32> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn df(pairs: &[(&str, u32)]) -> HashMap<String, u32> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    // --- idf ---

    #[test]
    fn idf_zero_corpus_is_zero() {
        assert_eq!(scorer().idf(0.0, 0.0), 0.0);
    }

    #[test]
    fn idf_term_in_all_docs_approaches_zero() {
        // df == corpus_size → numerator ≈ 0.5, denominator ≈ corpus + 0.5
        // result approaches ln(1) = 0 as corpus grows
        let idf = scorer().idf(100.0, 100.0);
        assert!(idf >= 0.0);
        assert!(idf < 0.01);
    }

    #[test]
    fn idf_term_in_one_doc_of_large_corpus_is_high() {
        let idf = scorer().idf(1.0, 1000.0);
        assert!(idf > 5.0);
    }

    #[test]
    fn idf_is_non_negative() {
        for df_val in [0.0, 0.5, 1.0, 10.0, 100.0] {
            assert!(scorer().idf(df_val, 100.0) >= 0.0);
        }
    }

    #[test]
    fn idf_increases_as_doc_frequency_decreases() {
        let rare = scorer().idf(1.0, 100.0);
        let common = scorer().idf(50.0, 100.0);
        assert!(rare > common);
    }

    // --- score: zero / trivial cases ---

    #[test]
    fn score_zero_when_avg_doc_length_is_zero() {
        let s = scorer().score(
            &["rust".to_string()],
            &tf(&[("rust", 3)]),
            10, 0.0,
            &df(&[("rust", 5)]),
            100,
        );
        assert_eq!(s, 0.0);
    }

    #[test]
    fn score_zero_when_term_not_in_document() {
        let s = scorer().score(
            &["absent".to_string()],
            &tf(&[]),
            10, 10.0,
            &df(&[("absent", 2)]),
            100,
        );
        assert_eq!(s, 0.0);
    }

    #[test]
    fn score_zero_when_query_empty() {
        let s = scorer().score(
            &[],
            &tf(&[("rust", 3)]),
            10, 10.0,
            &df(&[("rust", 5)]),
            100,
        );
        assert_eq!(s, 0.0);
    }

    // --- score: positive cases ---

    #[test]
    fn score_is_positive_for_matching_term() {
        let s = scorer().score(
            &["rust".to_string()],
            &tf(&[("rust", 3)]),
            10, 10.0,
            &df(&[("rust", 5)]),
            100,
        );
        assert!(s > 0.0);
    }

    #[test]
    fn score_increases_with_term_frequency() {
        let low = scorer().score(
            &["rust".to_string()],
            &tf(&[("rust", 1)]),
            10, 10.0,
            &df(&[("rust", 5)]),
            100,
        );
        let high = scorer().score(
            &["rust".to_string()],
            &tf(&[("rust", 10)]),
            10, 10.0,
            &df(&[("rust", 5)]),
            100,
        );
        assert!(high > low);
    }

    #[test]
    fn score_saturates_at_high_term_frequency() {
        // BM25 should not grow unboundedly with tf
        let s10 = scorer().score(
            &["rust".to_string()], &tf(&[("rust", 10)]),
            10, 10.0, &df(&[("rust", 5)]), 100,
        );
        let s1000 = scorer().score(
            &["rust".to_string()], &tf(&[("rust", 1000)]),
            10, 10.0, &df(&[("rust", 5)]), 100,
        );
        // The ratio should be well below 10x even though tf is 100x
        assert!(s1000 / s10 < 5.0);
    }

    #[test]
    fn score_decreases_for_longer_documents() {
        let short = scorer().score(
            &["rust".to_string()], &tf(&[("rust", 2)]),
            5, 10.0, &df(&[("rust", 5)]), 100,
        );
        let long = scorer().score(
            &["rust".to_string()], &tf(&[("rust", 2)]),
            50, 10.0, &df(&[("rust", 5)]), 100,
        );
        assert!(short > long);
    }

    #[test]
    fn score_accumulates_across_multiple_terms() {
        let single = scorer().score(
            &["rust".to_string()],
            &tf(&[("rust", 2), ("programming", 1)]),
            10, 10.0,
            &df(&[("rust", 5), ("programming", 3)]),
            100,
        );
        let multi = scorer().score(
            &["rust".to_string(), "programming".to_string()],
            &tf(&[("rust", 2), ("programming", 1)]),
            10, 10.0,
            &df(&[("rust", 5), ("programming", 3)]),
            100,
        );
        assert!(multi > single);
    }

    #[test]
    fn duplicate_query_terms_are_counted_once() {
        let single = scorer().score(
            &["rust".to_string()],
            &tf(&[("rust", 2)]),
            10, 10.0, &df(&[("rust", 5)]), 100,
        );
        let doubled = scorer().score(
            &["rust".to_string(), "rust".to_string()],
            &tf(&[("rust", 2)]),
            10, 10.0, &df(&[("rust", 5)]), 100,
        );
        assert!((single - doubled).abs() < 1e-5);
    }

    #[test]
    fn b_zero_disables_length_normalisation() {
        let scorer_no_norm = Scorer::new(1.2, 0.0);
        let short = scorer_no_norm.score(
            &["rust".to_string()], &tf(&[("rust", 2)]),
            5, 10.0, &df(&[("rust", 5)]), 100,
        );
        let long = scorer_no_norm.score(
            &["rust".to_string()], &tf(&[("rust", 2)]),
            100, 10.0, &df(&[("rust", 5)]), 100,
        );
        // With b=0, document length has no effect
        assert!((short - long).abs() < 1e-5);
    }
}
