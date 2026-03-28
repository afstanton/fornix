//! In-memory routing metrics collector.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::router::types::RoutingDecision;

/// A recorded routing event.
#[derive(Debug, Clone)]
pub struct RoutingRecord {
    pub model: String,
    pub provider: String,
    pub confidence: Option<f32>,
    pub strategy: Option<String>,
    pub estimated_cost: Option<f64>,
}

impl From<&RoutingDecision> for RoutingRecord {
    fn from(d: &RoutingDecision) -> Self {
        Self {
            model: d.model.clone(),
            provider: d.provider.clone(),
            confidence: d.confidence,
            strategy: d.metadata.get("strategy").and_then(|v| v.as_str()).map(|s| s.to_string()),
            estimated_cost: d.metadata.get("estimated_cost").and_then(|v| v.as_f64()),
        }
    }
}

/// Summary of all recorded routing decisions.
#[derive(Debug, Clone)]
pub struct RoutingSummary {
    /// Total number of routing decisions recorded.
    pub count: usize,
    /// How many times each model was chosen.
    pub model_distribution: HashMap<String, usize>,
    /// Mean confidence across all decisions that had one.
    pub avg_confidence: Option<f32>,
    /// Mean estimated cost across all decisions that had one.
    pub avg_estimated_cost: Option<f64>,
}

/// Thread-safe in-memory store for routing decisions.
#[derive(Default)]
pub struct MetricsCollector {
    records: Mutex<Vec<RoutingRecord>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a routing decision.
    pub fn record(&self, decision: &RoutingDecision) {
        if let Ok(mut records) = self.records.lock() {
            records.push(RoutingRecord::from(decision));
        }
    }

    /// Return all recorded entries (cloned).
    pub fn entries(&self) -> Vec<RoutingRecord> {
        self.records.lock().map(|r| r.clone()).unwrap_or_default()
    }

    /// Produce an aggregate summary.
    pub fn summary(&self) -> RoutingSummary {
        let records = self.entries();
        if records.is_empty() {
            return RoutingSummary {
                count: 0,
                model_distribution: HashMap::new(),
                avg_confidence: None,
                avg_estimated_cost: None,
            };
        }

        let mut distribution: HashMap<String, usize> = HashMap::new();
        let mut confidences: Vec<f32> = Vec::new();
        let mut costs: Vec<f64> = Vec::new();

        for rec in &records {
            *distribution.entry(rec.model.clone()).or_insert(0) += 1;
            if let Some(c) = rec.confidence {
                confidences.push(c);
            }
            if let Some(cost) = rec.estimated_cost {
                costs.push(cost);
            }
        }

        let avg_confidence = if confidences.is_empty() {
            None
        } else {
            Some(confidences.iter().sum::<f32>() / confidences.len() as f32)
        };

        let avg_estimated_cost = if costs.is_empty() {
            None
        } else {
            Some(costs.iter().sum::<f64>() / costs.len() as f64)
        };

        RoutingSummary {
            count: records.len(),
            model_distribution: distribution,
            avg_confidence,
            avg_estimated_cost,
        }
    }

    /// Reset all recorded data.
    pub fn clear(&self) {
        if let Ok(mut records) = self.records.lock() {
            records.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::types::RoutingDecision;

    fn decision(model: &str, confidence: Option<f32>) -> RoutingDecision {
        let mut d = RoutingDecision::new(model, "p");
        if let Some(c) = confidence {
            d = d.with_confidence(c);
        }
        d
    }

    #[test]
    fn empty_collector_summary_has_zero_count() {
        let c = MetricsCollector::new();
        assert_eq!(c.summary().count, 0);
        assert!(c.summary().avg_confidence.is_none());
    }

    #[test]
    fn record_increments_count() {
        let c = MetricsCollector::new();
        c.record(&decision("gpt-4o", None));
        assert_eq!(c.summary().count, 1);
    }

    #[test]
    fn model_distribution_tracks_frequency() {
        let c = MetricsCollector::new();
        c.record(&decision("a", None));
        c.record(&decision("a", None));
        c.record(&decision("b", None));
        let dist = c.summary().model_distribution;
        assert_eq!(dist["a"], 2);
        assert_eq!(dist["b"], 1);
    }

    #[test]
    fn avg_confidence_computed_correctly() {
        let c = MetricsCollector::new();
        c.record(&decision("m", Some(0.8)));
        c.record(&decision("m", Some(0.6)));
        let avg = c.summary().avg_confidence.unwrap();
        assert!((avg - 0.7).abs() < 1e-5);
    }

    #[test]
    fn clear_resets_state() {
        let c = MetricsCollector::new();
        c.record(&decision("m", None));
        c.clear();
        assert_eq!(c.summary().count, 0);
    }

    #[test]
    fn entries_returns_all_records() {
        let c = MetricsCollector::new();
        c.record(&decision("a", None));
        c.record(&decision("b", None));
        assert_eq!(c.entries().len(), 2);
    }
}
