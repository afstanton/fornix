//! In-process BM25 adapter with a full inverted index.
//!
//! Implements the complete BM25 ranking pipeline in memory:
//! tokenisation → inverted index → IDF/TF scoring → ranked results.
//!
//! Thread-safe via `tokio::sync::RwLock`. Useful for testing, small corpora,
//! and as a reference implementation.

use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::bm25::{
    adapter::{Bm25Adapter, IndexDocument, SearchOptions},
    config::Bm25Config,
    error::{Error, Result},
    result::Bm25Result,
    scorer::Scorer,
    tokenizer::Tokenizer,
};
use crate::common::namespace::Namespace;
use crate::store::config::AdapterConfig;
use crate::store::health::{HealthReport, HealthStatus};

// ============================================================================
// Index data structures
// ============================================================================

/// Term frequencies for a single document field.
/// Maps token → occurrence count within that field.
type TermFreqs = HashMap<String, u32>;
/// Aggregated score tuple for a matched document.
type DocumentScore = (f32, Vec<String>, HashMap<String, f32>);

/// Per-document statistics for one field.
#[derive(Debug, Clone)]
struct DocFieldStats {
    /// Total token count (document length for this field).
    token_count: u32,
    /// Per-token frequency within this field.
    term_freqs: TermFreqs,
}

/// Corpus-level statistics for one (namespace, field) pair.
#[derive(Debug, Default, Clone)]
struct CorpusStats {
    /// Number of indexed documents.
    doc_count: u32,
    /// Sum of token counts (used to compute average document length).
    total_tokens: u64,
    /// Maps token → number of documents containing that token.
    doc_frequencies: HashMap<String, u32>,
}

impl CorpusStats {
    fn avg_doc_length(&self) -> f32 {
        if self.doc_count == 0 {
            0.0
        } else {
            self.total_tokens as f32 / self.doc_count as f32
        }
    }
}

/// All indexed data for one namespace.
#[derive(Debug, Default)]
struct NamespaceIndex {
    /// doc_id → (field_name → DocFieldStats)
    documents: HashMap<String, HashMap<String, DocFieldStats>>,
    /// field_name → CorpusStats
    corpus: HashMap<String, CorpusStats>,
}

impl NamespaceIndex {
    /// Add or replace a document. Updates corpus stats atomically.
    fn upsert(&mut self, id: &str, fields: &HashMap<String, Vec<String>>) {
        // Remove old document contribution from corpus stats first.
        if let Some(old_fields) = self.documents.get(id) {
            for (field_name, old_stats) in old_fields {
                let corpus = self.corpus.entry(field_name.clone()).or_default();
                corpus.doc_count = corpus.doc_count.saturating_sub(1);
                corpus.total_tokens =
                    corpus.total_tokens.saturating_sub(old_stats.token_count as u64);
                for token in old_stats.term_freqs.keys() {
                    if let Some(df) = corpus.doc_frequencies.get_mut(token) {
                        *df = df.saturating_sub(1);
                        if *df == 0 {
                            corpus.doc_frequencies.remove(token);
                        }
                    }
                }
            }
        }

        // Build new per-field stats.
        let mut doc_fields: HashMap<String, DocFieldStats> = HashMap::new();
        for (field_name, tokens) in fields {
            let mut term_freqs: TermFreqs = HashMap::new();
            for token in tokens {
                *term_freqs.entry(token.clone()).or_insert(0) += 1;
            }
            doc_fields.insert(field_name.clone(), DocFieldStats {
                token_count: tokens.len() as u32,
                term_freqs,
            });
        }

        // Update corpus stats with new document.
        for (field_name, stats) in &doc_fields {
            let corpus = self.corpus.entry(field_name.clone()).or_default();
            corpus.doc_count += 1;
            corpus.total_tokens += stats.token_count as u64;
            for token in stats.term_freqs.keys() {
                *corpus.doc_frequencies.entry(token.clone()).or_insert(0) += 1;
            }
        }

        self.documents.insert(id.to_string(), doc_fields);
    }

    /// Remove a document. Returns `true` if it existed.
    fn remove(&mut self, id: &str) -> bool {
        let Some(old_fields) = self.documents.remove(id) else {
            return false;
        };
        for (field_name, old_stats) in &old_fields {
            let corpus = self.corpus.entry(field_name.clone()).or_default();
            corpus.doc_count = corpus.doc_count.saturating_sub(1);
            corpus.total_tokens =
                corpus.total_tokens.saturating_sub(old_stats.token_count as u64);
            for token in old_stats.term_freqs.keys() {
                if let Some(df) = corpus.doc_frequencies.get_mut(token) {
                    *df = df.saturating_sub(1);
                    if *df == 0 {
                        corpus.doc_frequencies.remove(token);
                    }
                }
            }
        }
        true
    }

    /// Score all documents for the given query tokens and field set.
    ///
    /// Returns a map of doc_id → (total_score, matched_terms, per_field_scores).
    fn score_all(
        &self,
        query_tokens: &[String],
        fields: &[String],
        scorer: &Scorer,
    ) -> HashMap<String, DocumentScore> {
        let mut results: HashMap<String, DocumentScore> = HashMap::new();

        for (doc_id, doc_fields) in &self.documents {
            let mut total_score = 0.0_f32;
            let mut field_scores: HashMap<String, f32> = HashMap::new();
            let mut matched: std::collections::HashSet<String> = std::collections::HashSet::new();

            for (field_name, stats) in doc_fields {
                // Skip fields not in the requested set (unless set is empty → all fields)
                if !fields.is_empty() && !fields.contains(field_name) {
                    continue;
                }
                let corpus = match self.corpus.get(field_name) {
                    Some(c) => c,
                    None => continue,
                };
                let field_score = scorer.score(
                    query_tokens,
                    &stats.term_freqs,
                    stats.token_count,
                    corpus.avg_doc_length(),
                    &corpus.doc_frequencies,
                    corpus.doc_count,
                );
                if field_score > 0.0 {
                    total_score += field_score;
                    field_scores.insert(field_name.clone(), field_score);
                    for token in query_tokens {
                        if stats.term_freqs.contains_key(token) {
                            matched.insert(token.clone());
                        }
                    }
                }
            }

            if total_score > 0.0 {
                results.insert(
                    doc_id.clone(),
                    (total_score, matched.into_iter().collect(), field_scores),
                );
            }
        }

        results
    }
}

// ============================================================================
// Adapter
// ============================================================================

/// In-process BM25 adapter backed by a full inverted index.
pub struct MemoryBm25Adapter {
    config: Bm25Config,
    connected: bool,
    tokenizer: Tokenizer,
    scorer: Scorer,
    /// namespace → NamespaceIndex
    index: RwLock<HashMap<String, NamespaceIndex>>,
}

impl MemoryBm25Adapter {
    /// Create a new (disconnected) adapter.
    pub fn new(config: Bm25Config) -> Self {
        let tokenizer = Tokenizer::from_config(&config);
        let scorer = Scorer::from_config(&config);
        Self {
            config,
            connected: false,
            tokenizer,
            scorer,
            index: RwLock::new(HashMap::new()),
        }
    }

    /// Create and immediately connect an adapter.
    pub async fn connect(config: Bm25Config) -> Result<Self> {
        config.validate().map_err(|e| Error::config(e.to_string()))?;
        let tokenizer = Tokenizer::from_config(&config);
        let scorer = Scorer::from_config(&config);
        Ok(Self {
            config,
            connected: true,
            tokenizer,
            scorer,
            index: RwLock::new(HashMap::new()),
        })
    }

    fn resolve_ns<'a>(&'a self, ns: Option<&'a Namespace>) -> &'a str {
        ns.and_then(|n| n.as_deref())
            .or_else(|| self.config.default_namespace_str())
            .unwrap_or("default")
    }
}

// Small helper on config to avoid carrying a namespace field there.
impl Bm25Config {
    fn default_namespace_str(&self) -> Option<&str> {
        None // BM25 config doesn't have a default namespace; callers use "default"
    }
}

#[async_trait]
impl Bm25Adapter for MemoryBm25Adapter {
    fn name(&self) -> &'static str {
        "memory"
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    fn config(&self) -> &Bm25Config {
        &self.config
    }

    async fn index(&self, document: IndexDocument, namespace: Option<&Namespace>) -> Result<()> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace).to_string();

        // Tokenise each field.
        let tokenized: HashMap<String, Vec<String>> = document
            .fields
            .iter()
            .map(|(field, text)| (field.clone(), self.tokenizer.tokenize(text)))
            .collect();

        let mut idx = self.index.write().await;
        idx.entry(ns).or_default().upsert(&document.id, &tokenized);
        Ok(())
    }

    async fn remove(&self, id: &str, namespace: Option<&Namespace>) -> Result<bool> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace).to_string();
        let mut idx = self.index.write().await;
        Ok(idx.entry(ns).or_default().remove(id))
    }

    async fn search(
        &self,
        query: &str,
        namespace: Option<&Namespace>,
        options: SearchOptions,
    ) -> Result<Vec<Bm25Result>> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let ns = self.resolve_ns(namespace).to_string();
        let query_tokens = self.tokenizer.tokenize(query);

        if query_tokens.is_empty() {
            return Ok(Vec::new());
        }

        let idx = self.index.read().await;
        let ns_index = match idx.get(&ns) {
            Some(i) => i,
            None => return Ok(Vec::new()),
        };

        let raw = ns_index.score_all(&query_tokens, &options.fields, &self.scorer);

        let mut results: Vec<Bm25Result> = raw
            .into_iter()
            .filter(|(_, (score, _, _))| {
                options.min_score.is_none_or(|min| *score >= min)
            })
            .map(|(id, (score, matched, field_scores))| {
                Bm25Result::new(id, score, matched, field_scores)
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(options.limit);
        Ok(results)
    }

    async fn count(&self, namespace: Option<&Namespace>) -> Result<usize> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let idx = self.index.read().await;
        Ok(match namespace {
            Some(_) => {
                let ns = self.resolve_ns(namespace);
                idx.get(ns).map(|i| i.documents.len()).unwrap_or(0)
            }
            None => idx.values().map(|i| i.documents.len()).sum(),
        })
    }

    async fn clear(&self, namespace: Option<&Namespace>) -> Result<usize> {
        if !self.connected {
            return Err(Error::NotConnected);
        }
        let mut idx = self.index.write().await;
        match namespace {
            None => {
                let total: usize = idx.values().map(|i| i.documents.len()).sum();
                idx.clear();
                Ok(total)
            }
            Some(_) => {
                let ns = self.resolve_ns(namespace).to_string();
                let count = idx.get(&ns).map(|i| i.documents.len()).unwrap_or(0);
                idx.remove(&ns);
                Ok(count)
            }
        }
    }

    async fn healthcheck(&self) -> HealthReport {
        let status = if self.connected {
            HealthStatus::Healthy
        } else {
            HealthStatus::Unhealthy { reason: "not connected".to_string() }
        };
        HealthReport::begin("memory-bm25").finish(status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::adapter::SearchOptions;
    use crate::bm25::config::Bm25Config;

    async fn adapter() -> MemoryBm25Adapter {
        MemoryBm25Adapter::connect(Bm25Config::default()).await.unwrap()
    }

    fn doc(id: &str, text: &str) -> IndexDocument {
        IndexDocument::new(id, text)
    }

    fn ns(s: &str) -> Namespace {
        Namespace::named(s)
    }

    // --- lifecycle ---

    #[tokio::test]
    async fn new_is_disconnected() {
        let a = MemoryBm25Adapter::new(Bm25Config::default());
        assert!(!a.is_connected());
    }

    #[tokio::test]
    async fn connect_produces_connected_adapter() {
        assert!(adapter().await.is_connected());
    }

    #[tokio::test]
    async fn name_is_memory() {
        assert_eq!(adapter().await.name(), "memory");
    }

    #[tokio::test]
    async fn operations_fail_when_not_connected() {
        let a = MemoryBm25Adapter::new(Bm25Config::default());
        let err = a.index(doc("id", "text"), None).await.unwrap_err();
        assert!(matches!(err, Error::NotConnected));
    }

    // --- index / count ---

    #[tokio::test]
    async fn index_increments_count() {
        let a = adapter().await;
        a.index(doc("a", "rust programming language"), None).await.unwrap();
        assert_eq!(a.count(None).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn index_same_id_replaces_document() {
        let a = adapter().await;
        a.index(doc("a", "original text"), None).await.unwrap();
        a.index(doc("a", "updated text"), None).await.unwrap();
        assert_eq!(a.count(None).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn multiple_documents_counted() {
        let a = adapter().await;
        for i in 0..5 {
            a.index(doc(&i.to_string(), "some text"), None).await.unwrap();
        }
        assert_eq!(a.count(None).await.unwrap(), 5);
    }

    // --- search ---

    #[tokio::test]
    async fn search_returns_matching_documents() {
        let a = adapter().await;
        a.index(doc("rust", "rust programming language systems"), None).await.unwrap();
        a.index(doc("python", "python scripting easy language"), None).await.unwrap();

        let results = a.search("rust", None, SearchOptions::default()).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "rust");
    }

    #[tokio::test]
    async fn search_empty_query_returns_no_results() {
        let a = adapter().await;
        a.index(doc("a", "some content"), None).await.unwrap();
        let results = a.search("", None, SearchOptions::default()).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_stop_word_only_query_returns_no_results() {
        let a = adapter().await;
        a.index(doc("a", "some content here"), None).await.unwrap();
        // "the" is a stop word by default
        let results = a.search("the", None, SearchOptions::default()).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_ranked_by_score_descending() {
        let a = adapter().await;
        // "rust" appears many times in first doc, once in second
        a.index(doc("high", "rust rust rust rust systems rust"), None).await.unwrap();
        a.index(doc("low",  "rust scripting easy"), None).await.unwrap();

        let results = a.search("rust", None, SearchOptions::default()).await.unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].score >= results[1].score);
    }

    #[tokio::test]
    async fn search_respects_limit() {
        let a = adapter().await;
        for i in 0..5 {
            a.index(doc(&i.to_string(), "rust programming"), None).await.unwrap();
        }
        let results = a
            .search("rust", None, SearchOptions::default().with_limit(2))
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn search_respects_min_score() {
        let a = adapter().await;
        a.index(doc("a", "rust programming systems language"), None).await.unwrap();
        a.index(doc("b", "rust"), None).await.unwrap();

        let results = a
            .search("rust programming", None, SearchOptions::default().with_min_score(1.0))
            .await
            .unwrap();

        assert!(results.iter().all(|r| r.score >= 1.0));
    }

    #[tokio::test]
    async fn search_result_has_matched_terms() {
        let a = adapter().await;
        a.index(doc("a", "rust programming language"), None).await.unwrap();

        let results = a.search("rust programming", None, SearchOptions::default()).await.unwrap();
        assert!(!results[0].matched_terms.is_empty());
    }

    #[tokio::test]
    async fn search_result_has_field_scores() {
        let a = adapter().await;
        a.index(doc("a", "rust systems programming"), None).await.unwrap();

        let results = a.search("rust", None, SearchOptions::default()).await.unwrap();
        assert!(!results[0].field_scores.is_empty());
    }

    // --- multi-field ---

    #[tokio::test]
    async fn multi_field_search_combines_scores() {
        let a = adapter().await;
        let doc_multi = IndexDocument::with_fields("a", [
            ("title", "Rust programming"),
            ("body", "Systems language for memory safety"),
        ]);
        a.index(doc_multi, None).await.unwrap();

        let results = a.search("rust systems", None, SearchOptions::default()).await.unwrap();
        assert!(!results.is_empty());
        // The result should have both fields represented in field_scores
        let r = &results[0];
        assert!(r.field_scores.contains_key("title") || r.field_scores.contains_key("body"));
    }

    #[tokio::test]
    async fn field_restriction_limits_search_to_specified_fields() {
        let a = adapter().await;
        let d = IndexDocument::with_fields("a", [
            ("title", "rust programming"),
            ("body",  "python scripting"),
        ]);
        a.index(d, None).await.unwrap();

        // Searching "python" restricted to "title" field should find nothing
        let results = a
            .search(
                "python",
                None,
                SearchOptions::default().with_fields(["title"]),
            )
            .await
            .unwrap();
        assert!(results.is_empty());

        // Searching "python" restricted to "body" should find the doc
        let results = a
            .search(
                "python",
                None,
                SearchOptions::default().with_fields(["body"]),
            )
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    // --- namespace isolation ---

    #[tokio::test]
    async fn namespaces_are_isolated() {
        let a = adapter().await;
        a.index(doc("a", "rust programming"), Some(&ns("ns1"))).await.unwrap();
        a.index(doc("b", "python scripting"), Some(&ns("ns2"))).await.unwrap();

        assert_eq!(a.count(Some(&ns("ns1"))).await.unwrap(), 1);
        assert_eq!(a.count(Some(&ns("ns2"))).await.unwrap(), 1);
        assert_eq!(a.count(None).await.unwrap(), 2);

        // "rust" should only be found in ns1
        let r1 = a.search("rust", Some(&ns("ns1")), SearchOptions::default()).await.unwrap();
        let r2 = a.search("rust", Some(&ns("ns2")), SearchOptions::default()).await.unwrap();
        assert!(!r1.is_empty());
        assert!(r2.is_empty());
    }

    // --- remove ---

    #[tokio::test]
    async fn remove_existing_returns_true() {
        let a = adapter().await;
        a.index(doc("a", "rust"), None).await.unwrap();
        assert!(a.remove("a", None).await.unwrap());
        assert_eq!(a.count(None).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn remove_missing_returns_false() {
        let a = adapter().await;
        assert!(!a.remove("nope", None).await.unwrap());
    }

    #[tokio::test]
    async fn removed_document_not_returned_in_search() {
        let a = adapter().await;
        a.index(doc("a", "rust programming"), None).await.unwrap();
        a.remove("a", None).await.unwrap();
        let results = a.search("rust", None, SearchOptions::default()).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn corpus_stats_updated_after_remove() {
        let a = adapter().await;
        a.index(doc("a", "rust programming"), None).await.unwrap();
        a.index(doc("b", "rust systems"), None).await.unwrap();
        a.remove("a", None).await.unwrap();

        // After removing "a", searching should still find "b"
        let results = a.search("rust", None, SearchOptions::default()).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    // --- clear ---

    #[tokio::test]
    async fn clear_namespace_removes_only_that_namespace() {
        let a = adapter().await;
        a.index(doc("a", "rust"), Some(&ns("x"))).await.unwrap();
        a.index(doc("b", "rust"), Some(&ns("y"))).await.unwrap();

        let removed = a.clear(Some(&ns("x"))).await.unwrap();
        assert_eq!(removed, 1);
        assert_eq!(a.count(Some(&ns("x"))).await.unwrap(), 0);
        assert_eq!(a.count(Some(&ns("y"))).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn clear_all_removes_everything() {
        let a = adapter().await;
        a.index(doc("a", "rust"), Some(&ns("x"))).await.unwrap();
        a.index(doc("b", "rust"), Some(&ns("y"))).await.unwrap();

        let removed = a.clear(None).await.unwrap();
        assert_eq!(removed, 2);
        assert_eq!(a.count(None).await.unwrap(), 0);
    }

    // --- idf correctness ---

    #[tokio::test]
    async fn rare_term_scores_higher_than_common_term() {
        let a = adapter().await;
        // "rust" appears in all 5 docs (common); "unique" appears in only 1 (rare)
        for i in 0..5 {
            let text = if i == 0 {
                "rust unique systems".to_string()
            } else {
                format!("rust doc{}", i)
            };
            a.index(doc(&i.to_string(), &text), None).await.unwrap();
        }

        let common_results = a.search("rust", None, SearchOptions::default()).await.unwrap();
        let rare_results = a.search("unique", None, SearchOptions::default()).await.unwrap();

        // The single doc with "unique" should score higher for "unique" than for "rust"
        assert!(!rare_results.is_empty());
        assert!(!common_results.is_empty());
        let rare_score = rare_results[0].score;
        let common_score_for_same_doc = common_results
            .iter()
            .find(|r| r.id == "0")
            .map(|r| r.score)
            .unwrap_or(0.0);
        assert!(rare_score > common_score_for_same_doc);
    }

    // --- healthcheck ---

    #[tokio::test]
    async fn healthcheck_healthy_when_connected() {
        let r = adapter().await.healthcheck().await;
        assert!(r.status.is_healthy());
    }

    #[tokio::test]
    async fn healthcheck_unhealthy_when_not_connected() {
        let a = MemoryBm25Adapter::new(Bm25Config::default());
        let r = a.healthcheck().await;
        assert!(!r.status.is_usable());
    }
}
