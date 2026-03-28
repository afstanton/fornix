//! GraphRAG configuration.

/// Which duplicate-handling strategy to use during ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnDuplicate { #[default] Skip, Update, Merge }

/// How to handle a content change on re-ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnContentChange { #[default] Update, Supersede, Skip }

/// Entity resolution strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResolutionStrategy { #[default] EmbeddingSimilarity, NameNormalization }

/// Confidence weights for composite scoring.
#[derive(Debug, Clone)]
pub struct ConfidenceWeights {
    pub llm:  f32,
    pub freq: f32,
    pub corr: f32,
}

impl Default for ConfidenceWeights {
    fn default() -> Self {
        Self { llm: 0.5, freq: 0.3, corr: 0.2 }
    }
}

/// Coverage score weights.
#[derive(Debug, Clone)]
pub struct CoverageWeights {
    pub description: f32,
    pub property:    f32,
    pub relation:    f32,
    pub source:      f32,
}

impl Default for CoverageWeights {
    fn default() -> Self {
        Self { description: 0.35, property: 0.25, relation: 0.25, source: 0.15 }
    }
}

/// Full GraphRAG configuration.
#[derive(Debug, Clone)]
pub struct GraphRagConfig {
    // ── Extraction ─────────────────────────────────────────────────────────
    pub entity_types:            Vec<String>,
    pub relation_types:          Vec<String>,
    pub extraction_chunk_size:   usize,
    pub extraction_max_entities: usize,
    pub max_gleanings:           usize,
    pub extraction_temperature:  f32,

    // ── Resolution ─────────────────────────────────────────────────────────
    pub resolution_strategy:       ResolutionStrategy,
    pub resolution_threshold:      f32,
    pub resolution_candidate_limit: usize,

    // ── Community ──────────────────────────────────────────────────────────
    pub min_community_size:     usize,
    pub max_community_summaries: usize,
    pub summary_concurrency:    usize,

    // ── Search ─────────────────────────────────────────────────────────────
    pub local_search_depth:     usize,
    pub auto_global_min_terms:  usize,
    pub auto_global_min_chars:  usize,

    // ── Causal ─────────────────────────────────────────────────────────────
    pub causal_extraction_enabled: bool,
    pub causal_max_depth:          usize,
    pub chain_confidence_enabled:  bool,
    pub chain_confidence_decay:    f32,
    pub min_chain_confidence:      Option<f32>,

    // ── Confidence ─────────────────────────────────────────────────────────
    pub confidence_enabled: bool,
    pub confidence_weights: ConfidenceWeights,

    // ── Coverage ───────────────────────────────────────────────────────────
    pub coverage_target_description_length: usize,
    pub coverage_target_relation_types:     usize,
    pub coverage_target_source_count:       usize,
    pub coverage_weights:                   CoverageWeights,
    pub coverage_low_score_threshold:       f32,

    // ── Ingestion ──────────────────────────────────────────────────────────
    pub batch_size:           usize,
    pub default_on_duplicate: OnDuplicate,
    pub default_on_content_change: OnContentChange,

    // ── Antifragility ──────────────────────────────────────────────────────
    pub scout_enabled:             bool,
    pub scout_orphan_min_mentions: usize,
    pub scout_confidence_threshold: f32,
}

impl Default for GraphRagConfig {
    fn default() -> Self {
        Self {
            entity_types: crate::graphrag::schema::DEFAULT_ENTITY_TYPES
                .iter().map(|s| s.to_string()).collect(),
            relation_types: crate::graphrag::schema::DEFAULT_RELATION_TYPES
                .iter().map(|s| s.to_string()).collect(),
            extraction_chunk_size:   2000,
            extraction_max_entities: 50,
            max_gleanings:           1,
            extraction_temperature:  0.0,
            resolution_strategy:     ResolutionStrategy::EmbeddingSimilarity,
            resolution_threshold:    0.85,
            resolution_candidate_limit: 200,
            min_community_size:      3,
            max_community_summaries: 10,
            summary_concurrency:     4,
            local_search_depth:      2,
            auto_global_min_terms:   8,
            auto_global_min_chars:   80,
            causal_extraction_enabled: false,
            causal_max_depth:        5,
            chain_confidence_enabled: true,
            chain_confidence_decay:  0.9,
            min_chain_confidence:    None,
            confidence_enabled:      true,
            confidence_weights:      ConfidenceWeights::default(),
            coverage_target_description_length: 300,
            coverage_target_relation_types:     5,
            coverage_target_source_count:       3,
            coverage_weights:        CoverageWeights::default(),
            coverage_low_score_threshold: 0.35,
            batch_size:              50,
            default_on_duplicate:    OnDuplicate::Skip,
            default_on_content_change: OnContentChange::Update,
            scout_enabled:           false,
            scout_orphan_min_mentions: 3,
            scout_confidence_threshold: 0.35,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_entity_types_populated() {
        let c = GraphRagConfig::default();
        assert!(c.entity_types.contains(&"Person".to_string()));
    }

    #[test]
    fn default_resolution_threshold() {
        assert!((GraphRagConfig::default().resolution_threshold - 0.85).abs() < 1e-6);
    }

    #[test]
    fn confidence_weights_sum_to_one() {
        let w = ConfidenceWeights::default();
        assert!((w.llm + w.freq + w.corr - 1.0).abs() < 1e-6);
    }

    #[test]
    fn coverage_weights_sum_to_one() {
        let w = CoverageWeights::default();
        assert!((w.description + w.property + w.relation + w.source - 1.0).abs() < 1e-6);
    }
}
