//! Query gap tracker.
//!
//! Observes RAG results and records queries that produced low-similarity
//! results ("misses"), enabling the corpus maintainer to identify content
//! gaps that should be filled.

use std::collections::HashMap;

const DEFAULT_THRESHOLD: f32 = 0.45;

/// Stop words stripped from queries before normalisation.
const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "of", "on", "or", "that", "the", "these", "this", "to",
    "what", "when", "where", "which", "who", "why", "with",
];

/// Per-pattern statistics tracked by the gap tracker.
#[derive(Debug, Clone, Default)]
struct PatternEntry {
    miss_count: usize,
    similarities: Vec<f32>,
}

impl PatternEntry {
    fn avg_similarity(&self) -> Option<f32> {
        if self.similarities.is_empty() {
            return None;
        }
        Some(self.similarities.iter().sum::<f32>() / self.similarities.len() as f32)
    }
}

/// A summary of a frequently-missed query pattern.
#[derive(Debug, Clone)]
pub struct MissedPattern {
    pub pattern: String,
    pub miss_count: usize,
    pub avg_similarity: Option<f32>,
}

/// Tracks queries that consistently fail to retrieve high-similarity results.
#[derive(Debug)]
pub struct QueryGapTracker {
    miss_threshold: f32,
    patterns: HashMap<String, PatternEntry>,
}

impl QueryGapTracker {
    pub fn new(miss_threshold: f32) -> Self {
        Self { miss_threshold, patterns: HashMap::new() }
    }

    /// Observe a query and its maximum retrieved similarity.
    ///
    /// A query is recorded as a miss when no context exceeds
    /// `miss_threshold` similarity.
    pub fn observe(&mut self, query: &str, max_similarity: Option<f32>) {
        let pattern = self.normalise(query);
        if pattern.is_empty() {
            return;
        }

        let is_miss = max_similarity.is_none_or(|s| s < self.miss_threshold);
        if !is_miss {
            return;
        }

        let entry = self.patterns.entry(pattern).or_default();
        entry.miss_count += 1;
        if let Some(s) = max_similarity {
            entry.similarities.push(s);
        }
    }

    /// Return the `limit` most frequently missed patterns, sorted by miss
    /// count descending.
    pub fn most_missed(&self, limit: usize) -> Vec<MissedPattern> {
        let mut rows: Vec<MissedPattern> = self
            .patterns
            .iter()
            .map(|(pattern, entry)| MissedPattern {
                pattern: pattern.clone(),
                miss_count: entry.miss_count,
                avg_similarity: entry.avg_similarity(),
            })
            .collect();

        rows.sort_by(|a, b| {
            b.miss_count
                .cmp(&a.miss_count)
                .then_with(|| a.pattern.cmp(&b.pattern))
        });
        rows.truncate(limit);
        rows
    }

    /// Total number of distinct missed patterns recorded.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    fn normalise(&self, query: &str) -> String {
        let lower = query.to_lowercase();
        let owned: Vec<String> = lower
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|t| !t.is_empty())
            .filter(|t| !STOP_WORDS.contains(t))
            .map(|t| t.to_string())
            .collect();

        if owned.is_empty() {
            return query.trim().to_lowercase();
        }
        owned.join(" ")
    }
}

impl Default for QueryGapTracker {
    fn default() -> Self {
        Self::new(DEFAULT_THRESHOLD)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_misses_initially() {
        let tracker = QueryGapTracker::default();
        assert!(tracker.most_missed(10).is_empty());
    }

    #[test]
    fn below_threshold_is_recorded() {
        let mut t = QueryGapTracker::new(0.5);
        t.observe("machine learning", Some(0.3));
        assert_eq!(t.pattern_count(), 1);
    }

    #[test]
    fn above_threshold_is_not_recorded() {
        let mut t = QueryGapTracker::new(0.5);
        t.observe("machine learning", Some(0.8));
        assert_eq!(t.pattern_count(), 0);
    }

    #[test]
    fn none_similarity_is_recorded_as_miss() {
        let mut t = QueryGapTracker::default();
        t.observe("what is rag", None);
        assert_eq!(t.pattern_count(), 1);
    }

    #[test]
    fn stop_words_removed_from_pattern() {
        let mut t = QueryGapTracker::default();
        t.observe("what is the meaning of rag", Some(0.1));
        let missed = t.most_missed(1);
        assert!(!missed[0].pattern.contains("what"));
        assert!(!missed[0].pattern.contains("the"));
        assert!(!missed[0].pattern.contains("of"));
        assert!(missed[0].pattern.contains("meaning"));
    }

    #[test]
    fn miss_count_increments_on_repeated_similar_query() {
        let mut t = QueryGapTracker::default();
        t.observe("neural networks", Some(0.1));
        t.observe("neural networks", Some(0.2));
        let missed = t.most_missed(1);
        assert_eq!(missed[0].miss_count, 2);
    }

    #[test]
    fn most_missed_sorted_by_count_descending() {
        let mut t = QueryGapTracker::default();
        for _ in 0..3 {
            t.observe("frequent miss", Some(0.1));
        }
        t.observe("rare miss", Some(0.1));
        let missed = t.most_missed(10);
        assert_eq!(missed[0].pattern, "frequent miss");
        assert_eq!(missed[0].miss_count, 3);
    }

    #[test]
    fn most_missed_respects_limit() {
        let mut t = QueryGapTracker::default();
        for i in 0..5 {
            t.observe(&format!("query {}", i), Some(0.1));
        }
        assert_eq!(t.most_missed(3).len(), 3);
    }

    #[test]
    fn avg_similarity_computed_correctly() {
        let mut t = QueryGapTracker::default();
        t.observe("gap query", Some(0.2));
        t.observe("gap query", Some(0.4));
        let missed = t.most_missed(1);
        let avg = missed[0].avg_similarity.unwrap();
        assert!((avg - 0.3).abs() < 1e-5);
    }

    #[test]
    fn empty_query_is_ignored() {
        let mut t = QueryGapTracker::default();
        t.observe("", Some(0.1));
        assert_eq!(t.pattern_count(), 0);
    }
}
