//! GraphRAG configuration.

use std::sync::Arc;
use crate::ontology::Definition;

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
    // ── Ontology ───────────────────────────────────────────────────────────
    /// An optional ontology definition for constrained extraction.
    ///
    /// When `Some`, [`effective_entity_types`] and [`effective_relation_types`]
    /// are derived from the ontology rather than the fallback lists below.
    /// The [`crate::ontology::OntologyValidator`] and
    /// [`crate::ontology::OntologyPrompt`] are used during extraction to
    /// constrain type lists and build per-type guidance blocks.
    ///
    /// When `None`, the `entity_types` and `relation_types` flat lists are
    /// used directly (existing behaviour, unchanged).
    pub ontology: Option<Arc<Definition>>,

    // ── Extraction ─────────────────────────────────────────────────────────
    /// Fallback entity type list used when `ontology` is `None`.
    /// When `ontology` is `Some`, this list is ignored by
    /// [`effective_entity_types`] but retained for callers that want to
    /// override type scoping explicitly.
    pub entity_types:            Vec<String>,
    /// Fallback relation type list used when `ontology` is `None`.
    pub relation_types:          Vec<String>,
    pub extraction_chunk_size:   usize,
    pub extraction_max_entities: usize,
    pub max_gleanings:           usize,
    pub extraction_temperature:  f32,

    // ── Resolution ─────────────────────────────────────────────────────────
    pub resolution_strategy:        ResolutionStrategy,
    pub resolution_threshold:       f32,
    pub resolution_candidate_limit: usize,

    // ── Community ──────────────────────────────────────────────────────────
    pub min_community_size:      usize,
    pub max_community_summaries: usize,
    pub summary_concurrency:     usize,

    // ── Search ─────────────────────────────────────────────────────────────
    pub local_search_depth:    usize,
    pub auto_global_min_terms: usize,
    pub auto_global_min_chars: usize,

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
    pub batch_size:                usize,
    pub default_on_duplicate:      OnDuplicate,
    pub default_on_content_change: OnContentChange,

    // ── Antifragility ──────────────────────────────────────────────────────
    pub scout_enabled:              bool,
    pub scout_orphan_min_mentions:  usize,
    pub scout_confidence_threshold: f32,
}

impl Default for GraphRagConfig {
    fn default() -> Self {
        Self {
            ontology: None,
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

impl GraphRagConfig {
    /// The effective entity type list for extraction prompt construction.
    ///
    /// When an `ontology` is configured, returns every entity type name
    /// defined in the ontology. When no ontology is configured, returns
    /// the flat `entity_types` fallback list.
    ///
    /// Callers that want a scoped subset (e.g. only types relevant to a
    /// particular text chunk) should use
    /// [`crate::ontology::OntologyPrompt::scoped_entity_types`] instead.
    pub fn effective_entity_types(&self) -> Vec<String> {
        match &self.ontology {
            Some(ont) => ont.entity_type_names().map(String::from).collect(),
            None => self.entity_types.clone(),
        }
    }

    /// The effective relation type list for extraction prompt construction.
    ///
    /// Mirrors [`effective_entity_types`] for relation types.
    pub fn effective_relation_types(&self) -> Vec<String> {
        match &self.ontology {
            Some(ont) => ont.relation_type_names().map(String::from).collect(),
            None => self.relation_types.clone(),
        }
    }

    /// Returns `true` if an ontology is configured.
    pub fn has_ontology(&self) -> bool {
        self.ontology.is_some()
    }

    /// Normalise an extracted entity type name via the configured ontology.
    ///
    /// If an ontology is configured and the candidate matches a canonical
    /// type name or alias, returns the canonical name. If the candidate is
    /// not in the ontology, returns `None`. If no ontology is configured,
    /// returns `None` (callers should use the fallback list).
    pub fn normalize_entity_type(&self, candidate: &str) -> Option<&str> {
        let ont = self.ontology.as_ref()?;
        ont.resolve_alias(candidate).map(|t| t.name.as_str())
    }

    /// Returns `true` if `entity_type` is a known type in the configured
    /// ontology (by canonical name or alias).
    ///
    /// Returns `false` when no ontology is configured, deferring to the
    /// caller's fallback logic.
    pub fn known_entity_type(&self, entity_type: &str) -> bool {
        self.ontology
            .as_ref()
            .map(|ont| ont.resolve_alias(entity_type).is_some())
            .unwrap_or(false)
    }

    /// Returns `true` if `relation_type` is a known type in the configured
    /// ontology.
    pub fn known_relation_type(&self, relation_type: &str) -> bool {
        self.ontology
            .as_ref()
            .map(|ont| ont.relation_type(relation_type).is_some())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::types::{
        Definition, EntityTypeDefinition, RelationTypeDefinition,
    };

    fn make_ontology() -> Arc<Definition> {
        let mut def = Definition::new("regulatory");
        def.version = Some("1.0.0".to_string());
        def.entity_types = vec![
            EntityTypeDefinition {
                name: "Regulation".to_string(),
                description: None,
                extraction_strategy: None,
                extraction_patterns: Vec::new(),
                aliases: vec!["Provision".to_string()],
                properties: Vec::new(),
            },
            EntityTypeDefinition {
                name: "Agency".to_string(),
                description: None,
                extraction_strategy: None,
                extraction_patterns: Vec::new(),
                aliases: Vec::new(),
                properties: Vec::new(),
            },
        ];
        def.relation_types = vec![RelationTypeDefinition {
            name: "ISSUED_BY".to_string(),
            description: None,
            source_types: vec!["Regulation".to_string()],
            target_types: vec!["Agency".to_string()],
            properties: Vec::new(),
        }];
        Arc::new(def)
    }

    // ── Default ──

    #[test]
    fn default_ontology_is_none() {
        assert!(GraphRagConfig::default().ontology.is_none());
    }

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

    // ── has_ontology ──

    #[test]
    fn has_ontology_false_when_none() {
        assert!(!GraphRagConfig::default().has_ontology());
    }

    #[test]
    fn has_ontology_true_when_set() {
        let c = GraphRagConfig { ontology: Some(make_ontology()), ..Default::default() };
        assert!(c.has_ontology());
    }

    // ── effective_entity_types ──

    #[test]
    fn effective_entity_types_uses_fallback_without_ontology() {
        let c = GraphRagConfig::default();
        let types = c.effective_entity_types();
        assert!(types.contains(&"Person".to_string()));
        assert!(!types.is_empty());
    }

    #[test]
    fn effective_entity_types_derives_from_ontology_when_set() {
        let c = GraphRagConfig { ontology: Some(make_ontology()), ..Default::default() };
        let types = c.effective_entity_types();
        assert!(types.contains(&"Regulation".to_string()));
        assert!(types.contains(&"Agency".to_string()));
        // Schema defaults like "Person" are NOT in the ontology
        assert!(!types.contains(&"Person".to_string()));
        assert_eq!(types.len(), 2);
    }

    // ── effective_relation_types ──

    #[test]
    fn effective_relation_types_uses_fallback_without_ontology() {
        let c = GraphRagConfig::default();
        let types = c.effective_relation_types();
        assert!(types.contains(&"RELATED_TO".to_string()));
    }

    #[test]
    fn effective_relation_types_derives_from_ontology_when_set() {
        let c = GraphRagConfig { ontology: Some(make_ontology()), ..Default::default() };
        let types = c.effective_relation_types();
        assert_eq!(types, vec!["ISSUED_BY".to_string()]);
    }

    // ── normalize_entity_type ──

    #[test]
    fn normalize_entity_type_no_ontology_returns_none() {
        let c = GraphRagConfig::default();
        assert!(c.normalize_entity_type("Regulation").is_none());
    }

    #[test]
    fn normalize_entity_type_canonical_name() {
        let c = GraphRagConfig { ontology: Some(make_ontology()), ..Default::default() };
        assert_eq!(c.normalize_entity_type("Regulation"), Some("Regulation"));
    }

    #[test]
    fn normalize_entity_type_resolves_alias() {
        let c = GraphRagConfig { ontology: Some(make_ontology()), ..Default::default() };
        assert_eq!(c.normalize_entity_type("Provision"), Some("Regulation"));
    }

    #[test]
    fn normalize_entity_type_unknown_returns_none() {
        let c = GraphRagConfig { ontology: Some(make_ontology()), ..Default::default() };
        assert!(c.normalize_entity_type("Unicorn").is_none());
    }

    // ── known_entity_type ──

    #[test]
    fn known_entity_type_no_ontology_returns_false() {
        assert!(!GraphRagConfig::default().known_entity_type("Regulation"));
    }

    #[test]
    fn known_entity_type_canonical() {
        let c = GraphRagConfig { ontology: Some(make_ontology()), ..Default::default() };
        assert!(c.known_entity_type("Regulation"));
        assert!(c.known_entity_type("Agency"));
    }

    #[test]
    fn known_entity_type_via_alias() {
        let c = GraphRagConfig { ontology: Some(make_ontology()), ..Default::default() };
        assert!(c.known_entity_type("Provision"));
    }

    #[test]
    fn known_entity_type_unknown_returns_false() {
        let c = GraphRagConfig { ontology: Some(make_ontology()), ..Default::default() };
        assert!(!c.known_entity_type("Unicorn"));
    }

    // ── known_relation_type ──

    #[test]
    fn known_relation_type_no_ontology_returns_false() {
        assert!(!GraphRagConfig::default().known_relation_type("ISSUED_BY"));
    }

    #[test]
    fn known_relation_type_known() {
        let c = GraphRagConfig { ontology: Some(make_ontology()), ..Default::default() };
        assert!(c.known_relation_type("ISSUED_BY"));
    }

    #[test]
    fn known_relation_type_unknown_returns_false() {
        let c = GraphRagConfig { ontology: Some(make_ontology()), ..Default::default() };
        assert!(!c.known_relation_type("INVENTED_BY"));
    }

    // ── Arc sharing / Clone ──

    #[test]
    fn config_clone_shares_ontology_arc() {
        let ont = make_ontology();
        let c = GraphRagConfig { ontology: Some(Arc::clone(&ont)), ..Default::default() };
        let cloned = c.clone();
        // Both configs share the same Arc allocation
        assert!(Arc::ptr_eq(
            c.ontology.as_ref().unwrap(),
            cloned.ontology.as_ref().unwrap()
        ));
    }
}
