//! GraphRAG domain types: extracted entities, search results, ingest
//! observations, and the LLM extraction interfaces.

use std::collections::HashMap;
use crate::common::metadata::Metadata;
use crate::graph::types::{CausalPath, Community, Entity, Relation};

// ─────────────────────────────────────────────────────────────────
// Extraction types
// ─────────────────────────────────────────────────────────────────

/// An entity extracted from raw text by the extraction pipeline.
#[derive(Debug, Clone)]
pub struct ExtractedEntity {
    pub name: String,
    pub entity_type: String,
    pub description: Option<String>,
    pub properties: Metadata,
}

/// A relation extracted from raw text.
#[derive(Debug, Clone)]
pub struct ExtractedRelation {
    pub from_name: String,
    pub to_name: String,
    pub relation_type: String,
    pub description: Option<String>,
    pub confidence: Option<f32>,
    pub properties: Metadata,
}

/// Combined extraction result for a single text chunk.
#[derive(Debug, Clone, Default)]
pub struct ExtractionResult {
    pub entities: Vec<ExtractedEntity>,
    pub relations: Vec<ExtractedRelation>,
}

// ─────────────────────────────────────────────────────────────────
// LLM interfaces
// ─────────────────────────────────────────────────────────────────

/// Minimal LLM interface used by the extraction and summarisation pipeline.
/// The implementation provides the network call; fornix supplies the prompts.
pub trait LlmClient: Send + Sync {
    /// Call the LLM with a single prompt and return its text output.
    fn complete(&self, prompt: &str) -> crate::graphrag::error::Result<String>;
}

/// An entity extractor converts raw text into [`ExtractedEntity`] values.
pub trait EntityExtractor: Send + Sync {
    fn extract(&self, text: &str) -> crate::graphrag::error::Result<Vec<ExtractedEntity>>;
}

/// A relation extractor converts raw text and known entities into
/// [`ExtractedRelation`] values.
pub trait RelationExtractor: Send + Sync {
    fn extract(
        &self,
        text: &str,
        entities: &[ExtractedEntity],
    ) -> crate::graphrag::error::Result<Vec<ExtractedRelation>>;
}

// ─────────────────────────────────────────────────────────────────
// Search result
// ─────────────────────────────────────────────────────────────────

/// A context item returned by a search operation.
#[derive(Debug, Clone)]
pub struct SearchContext {
    /// The entity this context is about.
    pub entity: Option<Entity>,
    /// Relations in the neighbourhood of this entity.
    pub relations: Vec<Relation>,
    /// Free-text snippet (e.g. community summary).
    pub text: Option<String>,
    /// Relevance or confidence score for this context item.
    pub score: Option<f32>,
    /// Provenance metadata.
    pub metadata: Metadata,
}

/// The result of a GraphRAG search operation.
#[derive(Debug, Clone, Default)]
pub struct SearchResult {
    /// Entities surfaced by the search.
    pub entities: Vec<Entity>,
    /// Context items (entity neighbourhoods + community summaries).
    pub contexts: Vec<SearchContext>,
    /// Communities relevant to the query.
    pub communities: Vec<Community>,
    /// Causal paths discovered during local search.
    pub paths: Vec<CausalPath>,
    /// Source provenance records.
    pub provenance: Vec<HashMap<String, serde_json::Value>>,
    /// Pre-computed LLM answer (global search only).
    pub answer: Option<String>,
    /// Mean confidence across entities in this result.
    pub avg_confidence: Option<f32>,
    /// Minimum confidence across entities in this result.
    pub min_confidence: Option<f32>,
}

impl SearchResult {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty() && self.contexts.is_empty()
    }

    /// Compute avg and min confidence from an entity slice.
    pub fn compute_confidence_metrics(entities: &[Entity]) -> (Option<f32>, Option<f32>) {
        let confidences: Vec<f32> = entities
            .iter()
            .filter_map(|e| e.confidence.overall)
            .collect();
        if confidences.is_empty() {
            return (None, None);
        }
        let avg = confidences.iter().sum::<f32>() / confidences.len() as f32;
        let min = confidences.iter().cloned().fold(f32::INFINITY, f32::min);
        (Some(avg), Some(min))
    }
}

// ─────────────────────────────────────────────────────────────────
// IngestObservation — antifragility signal
// ─────────────────────────────────────────────────────────────────

/// Default information-gain weights for antifragility tracking.
#[derive(Debug, Clone)]
pub struct InformationGainWeights {
    pub expansion:     f32,
    pub confidence:    f32,
    pub contradiction: f32,
}

impl Default for InformationGainWeights {
    fn default() -> Self {
        Self { expansion: 0.7, confidence: 0.3, contradiction: 0.0 }
    }
}

/// A recorded observation from one ingestion batch, used to track
/// antifragility signals.
#[derive(Debug, Clone)]
pub struct IngestObservation {
    pub batch_id:              Option<String>,
    pub observed_at:           std::time::SystemTime,
    pub surprisal:             Option<f32>,
    pub entities_created:      usize,
    pub relations_created:     usize,
    pub entities_merged:       usize,
    pub confidence_delta:      f32,
    pub contradictions_flagged: usize,
    pub is_stressor:           bool,
}

impl IngestObservation {
    /// Compute an information-gain score for this observation.
    ///
    /// Combines expansion (log of new entity/relation count), confidence
    /// delta, and a contradiction indicator with configurable weights.
    pub fn information_gain(&self, weights: &InformationGainWeights) -> f32 {
        let expansion = ((1 + self.entities_created + self.relations_created) as f32).ln();
        let confidence_gain = self.confidence_delta.max(0.0);
        let contradiction_signal = if self.contradictions_flagged > 0 { 1.0 } else { 0.0 };

        (weights.expansion * expansion
            + weights.confidence * confidence_gain
            + weights.contradiction * contradiction_signal)
            .clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    fn observation(created: usize, conf_delta: f32, contradictions: usize) -> IngestObservation {
        IngestObservation {
            batch_id: None,
            observed_at: SystemTime::now(),
            surprisal: None,
            entities_created: created,
            relations_created: 0,
            entities_merged: 0,
            confidence_delta: conf_delta,
            contradictions_flagged: contradictions,
            is_stressor: false,
        }
    }

    // ── SearchResult ──

    #[test]
    fn empty_result_is_empty() {
        assert!(SearchResult::empty().is_empty());
    }

    #[test]
    fn confidence_metrics_empty_entities() {
        let (avg, min) = SearchResult::compute_confidence_metrics(&[]);
        assert!(avg.is_none());
        assert!(min.is_none());
    }

    // ── IngestObservation ──

    #[test]
    fn information_gain_zero_creation_is_low() {
        let obs = observation(0, 0.0, 0);
        let ig = obs.information_gain(&InformationGainWeights::default());
        // ln(1) = 0, so expansion = 0; whole score is 0
        assert!(ig.abs() < 1e-5);
    }

    #[test]
    fn information_gain_positive_creation_is_positive() {
        let obs = observation(5, 0.0, 0);
        let ig = obs.information_gain(&InformationGainWeights::default());
        assert!(ig > 0.0, "ig={}", ig);
    }

    #[test]
    fn information_gain_is_clamped_to_one() {
        // Extreme creation count — gain should not exceed 1
        let obs = observation(10_000, 1.0, 1);
        let ig = obs.information_gain(&InformationGainWeights::default());
        assert!(ig <= 1.0 + 1e-6);
    }

    #[test]
    fn confidence_delta_contributes_positively() {
        let low = observation(1, 0.0, 0).information_gain(&InformationGainWeights::default());
        let high = observation(1, 0.5, 0).information_gain(&InformationGainWeights::default());
        assert!(high > low, "high={} low={}", high, low);
    }

    #[test]
    fn negative_confidence_delta_is_ignored() {
        let obs = observation(1, -0.5, 0);
        let ig = obs.information_gain(&InformationGainWeights::default());
        // Same as confidence_delta=0
        let baseline = observation(1, 0.0, 0).information_gain(&InformationGainWeights::default());
        assert!((ig - baseline).abs() < 1e-5);
    }

    #[test]
    fn contradiction_signal_contributes_with_nonzero_weight() {
        let weights = InformationGainWeights { expansion: 0.5, confidence: 0.3, contradiction: 0.2 };
        let without = observation(1, 0.0, 0).information_gain(&weights);
        let with = observation(1, 0.0, 1).information_gain(&weights);
        assert!(with > without, "with={} without={}", with, without);
    }

    // ── InformationGainWeights ──

    #[test]
    fn default_weights_sum_to_one() {
        let w = InformationGainWeights::default();
        assert!((w.expansion + w.confidence + w.contradiction - 1.0).abs() < 1e-6);
    }
}
