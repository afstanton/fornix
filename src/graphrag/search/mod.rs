//! GraphRAG search modes: Local, Global, and Hybrid.
//!
//! All three implement [`GraphRagSearch`], the common search trait.
//! The underlying graph and community data are injected at construction.

use crate::graph::{
    adapter::{CausalOptions, EntitySearchOptions, GraphAdapter},
    types::{CausalPath, Community, Entity},
};
use crate::graphrag::{
    config::GraphRagConfig,
    error::{Error, Result},
    types::{SearchContext, SearchResult},
};

/// Common search interface shared by Local, Global, and Hybrid.
pub trait GraphRagSearch: Send + Sync {
    fn name(&self) -> &'static str;

    /// Execute a search and return a [`SearchResult`].
    fn search(&self, query: &str) -> Result<SearchResult>;
}

// ─────────────────────────────────────────────────────────────────
// Causal intent detection (lightweight heuristic)
// ─────────────────────────────────────────────────────────────────

/// Terms that suggest the user wants causal information.
const CAUSAL_SIGNALS: &[&str] = &[
    "cause", "caused", "causes", "why", "because", "lead", "leads", "result",
    "resulted", "effect", "affect", "affects", "trigger", "triggers", "prevent",
    "prevents", "enable", "enables", "require", "requires",
];

const ANCESTOR_SIGNALS: &[&str] = &["why", "because", "cause", "caused", "what led", "root"];

/// Returns `true` when the query likely intends causal traversal.
pub fn has_causal_intent(query: &str) -> bool {
    let lower = query.to_lowercase();
    CAUSAL_SIGNALS.iter().any(|s| lower.contains(s))
}

/// Returns `true` when the causal traversal should go backwards (ancestors).
pub fn wants_ancestors(query: &str) -> bool {
    let lower = query.to_lowercase();
    ANCESTOR_SIGNALS.iter().any(|s| lower.contains(s))
}

// ─────────────────────────────────────────────────────────────────
// Local search
// ─────────────────────────────────────────────────────────────────

/// Seed-entity-based neighbourhood search.
///
/// 1. Resolves a seed entity from the query by exact name, fuzzy name, or
///    token match.
/// 2. Expands to depth-N neighbours.
/// 3. Optionally follows causal paths when the query implies causation.
pub struct LocalSearch<G: GraphAdapter> {
    graph: G,
    config: GraphRagConfig,
}

impl<G: GraphAdapter> LocalSearch<G> {
    pub fn new(graph: G, config: GraphRagConfig) -> Self {
        Self { graph, config }
    }

    async fn resolve_seed(&self, query: &str) -> Result<Option<Entity>> {
        // 1. Exact name match
        if let Ok(Some(e)) = self.graph.find_entity_by_name(query, None).await {
            return Ok(Some(e));
        }
        // 2. Type-agnostic search (limit 1)
        let opts = EntitySearchOptions::new().with_query(query).with_limit(1);
        if let Ok(results) = self.graph.search_entities(opts, None).await {
            if let Some(e) = results.into_iter().next() {
                return Ok(Some(e));
            }
        }
        // 3. Token-level fallback — try individual meaningful tokens
        let tokens: Vec<&str> = query
            .split_whitespace()
            .filter(|t| t.len() >= 3)
            .collect();
        for token in &tokens {
            let opts = EntitySearchOptions::new().with_query(token).with_limit(5);
            if let Ok(results) = self.graph.search_entities(opts, None).await {
                if let Some(e) = results.into_iter().next() {
                    return Ok(Some(e));
                }
            }
        }
        Ok(None)
    }

    pub async fn search_async(&self, query: &str) -> Result<SearchResult> {
        let seed = self.resolve_seed(query).await?;
        let seed = match seed {
            Some(e) => e,
            None => return Ok(SearchResult::empty()),
        };

        // Neighbour traversal
        let neighbours = self
            .graph
            .neighbors(seed.id, self.config.local_search_depth, Default::default(), None)
            .await
            .unwrap_or_default();

        // Causal paths when the query implies causation
        let causal_paths: Vec<CausalPath> = if self.config.causal_extraction_enabled
            && has_causal_intent(query)
        {
            let opts = CausalOptions {
                max_depth: self.config.causal_max_depth,
                ..Default::default()
            };
            if wants_ancestors(query) {
                self.graph
                    .causal_ancestors(seed.id, opts, None)
                    .await
                    .unwrap_or_default()
            } else {
                self.graph
                    .causal_descendants(seed.id, opts, None)
                    .await
                    .unwrap_or_default()
            }
        } else {
            Vec::new()
        };

        // Build contexts from neighbours
        let contexts: Vec<SearchContext> = neighbours
            .iter()
            .map(|entity| SearchContext {
                entity: Some(entity.clone()),
                relations: Vec::new(),
                text: None,
                score: entity.confidence.overall,
                metadata: Default::default(),
            })
            .collect();

        let mut all_entities = vec![seed];
        all_entities.extend(neighbours);

        let (avg_confidence, min_confidence) =
            SearchResult::compute_confidence_metrics(&all_entities);

        Ok(SearchResult {
            entities: all_entities,
            contexts,
            communities: Vec::new(),
            paths: causal_paths,
            provenance: Vec::new(),
            answer: None,
            avg_confidence,
            min_confidence,
        })
    }
}

impl<G: GraphAdapter + 'static> GraphRagSearch for LocalSearch<G> {
    fn name(&self) -> &'static str { "local" }

    fn search(&self, _query: &str) -> Result<SearchResult> {
        // Synchronous shim — callers with a Tokio runtime should use search_async.
        Err(Error::search("LocalSearch requires an async runtime; use search_async"))
    }
}

// ─────────────────────────────────────────────────────────────────
// Global search
// ─────────────────────────────────────────────────────────────────

/// Community-summary-based global search.
///
/// Ranks community summaries by lexical overlap with the query and
/// returns the top-N contexts. An optional LLM can synthesise a final
/// answer from the summaries; without one the raw summaries are returned.
pub struct GlobalSearch {
    communities: Vec<Community>,
    /// community key → pre-generated summary text
    summaries: std::collections::HashMap<String, String>,
    config: GraphRagConfig,
}

impl GlobalSearch {
    pub fn new(
        communities: Vec<Community>,
        summaries: std::collections::HashMap<String, String>,
        config: GraphRagConfig,
    ) -> Self {
        Self { communities, summaries, config }
    }

    /// Lexical overlap score: fraction of query tokens found in the summary.
    fn relevance_score(query: &str, summary: &str) -> f32 {
        let query_tokens: Vec<&str> = query.split_whitespace().collect();
        if query_tokens.is_empty() { return 0.0; }
        let lower_summary = summary.to_lowercase();
        let matched = query_tokens
            .iter()
            .filter(|t| lower_summary.contains(&t.to_lowercase()))
            .count();
        matched as f32 / query_tokens.len() as f32
    }

    fn community_key(community: &Community) -> String {
        let mut ids: Vec<u64> = community.entities.iter().map(|e| e.id).collect();
        ids.sort_unstable();
        ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(":")
    }

    pub fn search_communities(&self, query: &str) -> SearchResult {
        let limit = self.config.max_community_summaries;

        let mut scored: Vec<(&Community, f32, String)> = self
            .communities
            .iter()
            .filter_map(|community| {
                let key = Self::community_key(community);
                let summary = self.summaries.get(&key)?;
                let score = Self::relevance_score(query, summary);
                Some((community, score, summary.clone()))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        let contexts: Vec<SearchContext> = scored
            .iter()
            .map(|(_, score, summary)| SearchContext {
                entity: None,
                relations: Vec::new(),
                text: Some(summary.clone()),
                score: Some(*score),
                metadata: Default::default(),
            })
            .collect();

        let communities: Vec<Community> = scored
            .into_iter()
            .map(|(c, _, _)| c.clone())
            .collect();

        SearchResult {
            entities: Vec::new(),
            contexts,
            communities,
            paths: Vec::new(),
            provenance: Vec::new(),
            answer: None,
            avg_confidence: None,
            min_confidence: None,
        }
    }
}

impl GraphRagSearch for GlobalSearch {
    fn name(&self) -> &'static str { "global" }

    fn search(&self, query: &str) -> Result<SearchResult> {
        Ok(self.search_communities(query))
    }
}

// ─────────────────────────────────────────────────────────────────
// Hybrid search
// ─────────────────────────────────────────────────────────────────

/// Combines local (entity-seeded) and global (community) results.
pub struct HybridSearch<L: GraphRagSearch, G: GraphRagSearch> {
    local: L,
    global: G,
}

impl<L: GraphRagSearch, G: GraphRagSearch> HybridSearch<L, G> {
    pub fn new(local: L, global: G) -> Self {
        Self { local, global }
    }
}

impl<L: GraphRagSearch, G: GraphRagSearch> GraphRagSearch for HybridSearch<L, G> {
    fn name(&self) -> &'static str { "hybrid" }

    fn search(&self, query: &str) -> Result<SearchResult> {
        let local_result = self.local.search(query)?;
        let global_result = self.global.search(query)?;

        let mut contexts = local_result.contexts;
        contexts.extend(global_result.contexts);

        Ok(SearchResult {
            entities: local_result.entities,
            contexts,
            communities: global_result.communities,
            paths: local_result.paths,
            provenance: local_result.provenance,
            answer: global_result.answer,
            avg_confidence: local_result.avg_confidence,
            min_confidence: local_result.min_confidence,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::graph::types::Community;

    // ── Causal intent detection ──

    #[test]
    fn causal_intent_detected_in_why_query() {
        assert!(has_causal_intent("why did the market crash?"));
    }

    #[test]
    fn causal_intent_detected_in_causes_query() {
        assert!(has_causal_intent("what causes inflation?"));
    }

    #[test]
    fn no_causal_intent_in_factual_query() {
        assert!(!has_causal_intent("what is the capital of France?"));
    }

    #[test]
    fn ancestor_intent_detected() {
        assert!(wants_ancestors("why did this happen?"));
    }

    #[test]
    fn descendant_intent_not_ancestor() {
        assert!(!wants_ancestors("what does this lead to?"));
    }

    // ── GlobalSearch ──

    fn community(ids: &[u64]) -> Community {
        use crate::graph::types::{AssertionState, ConfidenceScores, Entity};
        let entities = ids.iter().map(|&id| Entity {
            id,
            name: format!("e{}", id),
            entity_type: "T".to_string(),
            properties: Default::default(),
            valid_from: None,
            valid_to: None,
            system_from: None,
            system_to: None,
            superseded_by: None,
            assertion_state: AssertionState::Active,
            confidence: ConfidenceScores::default(),
        }).collect();
        Community { entities, density: 0.5, central_entity: None }
    }

    fn global_search() -> GlobalSearch {
        let c1 = community(&[1, 2]);
        let c2 = community(&[3, 4]);
        let key1 = GlobalSearch::community_key(&c1);
        let key2 = GlobalSearch::community_key(&c2);
        let mut summaries = HashMap::new();
        summaries.insert(key1, "Rust programming language systems".to_string());
        summaries.insert(key2, "Python scripting data science".to_string());
        GlobalSearch::new(vec![c1, c2], summaries, GraphRagConfig::default())
    }

    #[test]
    fn global_search_returns_relevant_community_first() {
        let gs = global_search();
        let result = gs.search("rust systems").unwrap();
        assert!(!result.communities.is_empty());
        // Rust-related community should score higher
        let top_score = result.contexts[0].score.unwrap_or(0.0);
        assert!(top_score > 0.0);
    }

    #[test]
    fn global_search_empty_query_returns_result() {
        let gs = global_search();
        // Empty query matches nothing, but shouldn't error
        let result = gs.search("").unwrap();
        // Scores should be 0 for empty query
        assert!(result.contexts.iter().all(|c| c.score.unwrap_or(0.0) == 0.0));
    }

    #[test]
    fn relevance_score_full_match_is_one() {
        let score = GlobalSearch::relevance_score("rust systems", "rust systems programming");
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn relevance_score_no_match_is_zero() {
        let score = GlobalSearch::relevance_score("haskell functional", "python data science");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn relevance_score_partial_match() {
        let score = GlobalSearch::relevance_score("rust python", "rust programming");
        assert!((score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn global_search_respects_max_community_summaries_limit() {
        let config = GraphRagConfig { max_community_summaries: 1, ..Default::default() };
        let c1 = community(&[1, 2]);
        let c2 = community(&[3, 4]);
        let key1 = GlobalSearch::community_key(&c1);
        let key2 = GlobalSearch::community_key(&c2);
        let mut summaries = HashMap::new();
        summaries.insert(key1, "alpha".to_string());
        summaries.insert(key2, "beta".to_string());
        let gs = GlobalSearch::new(vec![c1, c2], summaries, config);
        let result = gs.search("alpha").unwrap();
        assert!(result.contexts.len() <= 1);
    }
}
