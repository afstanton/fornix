//! RAG evaluation framework.
//!
//! Provides metric traits and an `Evaluator` that scores RAG results against
//! ground-truth samples. Metric implementations that require an LLM judge
//! are stubbed here — they return `None` until an LLM backend is wired in.

use std::collections::HashMap;

/// A ground-truth evaluation sample.
#[derive(Debug, Clone)]
pub struct EvalSample {
    pub question: String,
    pub ground_truth_contexts: Vec<String>,
    pub ground_truth_answer: Option<String>,
}

impl EvalSample {
    pub fn new(question: impl Into<String>) -> Self {
        Self {
            question: question.into(),
            ground_truth_contexts: Vec::new(),
            ground_truth_answer: None,
        }
    }

    pub fn with_ground_truth(mut self, contexts: Vec<String>) -> Self {
        self.ground_truth_contexts = contexts;
        self
    }
}

/// The result of evaluating a single RAG retrieval against a sample.
#[derive(Debug, Clone)]
pub struct EvalResult {
    pub question: String,
    pub retrieved_contexts: Vec<String>,
    pub generated_answer: Option<String>,
    /// Metric name → score in [0.0, 1.0], or None if the metric could not run.
    pub scores: HashMap<String, Option<f32>>,
}

impl EvalResult {
    /// Weighted overall score.
    pub fn overall_score(&self, weights: &HashMap<String, f32>) -> Option<f32> {
        if self.scores.is_empty() {
            return None;
        }
        let mut total = 0.0_f32;
        let mut weight_sum = 0.0_f32;
        for (metric, score) in &self.scores {
            if let Some(s) = score {
                let w = weights.get(metric).copied().unwrap_or(1.0);
                total += s * w;
                weight_sum += w;
            }
        }
        if weight_sum < f32::EPSILON { None } else { Some(total / weight_sum) }
    }
}

/// The evaluation metric interface.
pub trait EvalMetric: Send + Sync {
    fn name(&self) -> &'static str;

    /// Compute a score in [0.0, 1.0], or `None` if the metric cannot run
    /// (e.g. LLM not configured).
    fn score(
        &self,
        question: &str,
        retrieved_contexts: &[String],
        generated_answer: Option<&str>,
        ground_truth_contexts: &[String],
    ) -> Option<f32>;
}

/// Context precision: what fraction of retrieved contexts are relevant?
///
/// Approximated without an LLM by checking for ground-truth string overlap.
pub struct ContextPrecision;

impl EvalMetric for ContextPrecision {
    fn name(&self) -> &'static str {
        "context_precision"
    }

    fn score(
        &self,
        _question: &str,
        retrieved_contexts: &[String],
        _generated_answer: Option<&str>,
        ground_truth_contexts: &[String],
    ) -> Option<f32> {
        if retrieved_contexts.is_empty() {
            return Some(0.0);
        }
        let relevant = retrieved_contexts
            .iter()
            .filter(|rc| {
                ground_truth_contexts
                    .iter()
                    .any(|gt| rc.contains(gt.as_str()) || gt.contains(rc.as_str()))
            })
            .count();
        Some(relevant as f32 / retrieved_contexts.len() as f32)
    }
}

/// Context recall: what fraction of ground-truth contexts were retrieved?
pub struct ContextRecall;

impl EvalMetric for ContextRecall {
    fn name(&self) -> &'static str {
        "context_recall"
    }

    fn score(
        &self,
        _question: &str,
        retrieved_contexts: &[String],
        _generated_answer: Option<&str>,
        ground_truth_contexts: &[String],
    ) -> Option<f32> {
        if ground_truth_contexts.is_empty() {
            return Some(1.0);
        }
        let covered = ground_truth_contexts
            .iter()
            .filter(|gt| {
                retrieved_contexts
                    .iter()
                    .any(|rc| rc.contains(gt.as_str()) || gt.contains(rc.as_str()))
            })
            .count();
        Some(covered as f32 / ground_truth_contexts.len() as f32)
    }
}

/// Faithfulness — requires an LLM judge; always returns `None` until wired.
pub struct Faithfulness;

impl EvalMetric for Faithfulness {
    fn name(&self) -> &'static str {
        "faithfulness"
    }

    fn score(&self, _: &str, _: &[String], _: Option<&str>, _: &[String]) -> Option<f32> {
        None // LLM judge required
    }
}

/// Answer relevance — requires an LLM judge; always returns `None` until wired.
pub struct AnswerRelevance;

impl EvalMetric for AnswerRelevance {
    fn name(&self) -> &'static str {
        "answer_relevance"
    }

    fn score(&self, _: &str, _: &[String], _: Option<&str>, _: &[String]) -> Option<f32> {
        None // LLM judge required
    }
}

/// Batch evaluator that runs a set of metrics over RAG results.
pub struct Evaluator {
    metrics: Vec<Box<dyn EvalMetric>>,
    score_weights: HashMap<String, f32>,
}

impl Evaluator {
    pub fn new(metrics: Vec<Box<dyn EvalMetric>>) -> Self {
        Self { metrics, score_weights: HashMap::new() }
    }

    /// Build an evaluator with the default metrics.
    pub fn default_metrics() -> Self {
        Self::new(vec![
            Box::new(ContextPrecision),
            Box::new(ContextRecall),
            Box::new(Faithfulness),
            Box::new(AnswerRelevance),
        ])
    }

    pub fn with_weights(mut self, weights: HashMap<String, f32>) -> Self {
        self.score_weights = weights;
        self
    }

    /// Evaluate a single RAG result against a ground-truth sample.
    pub fn evaluate(
        &self,
        retrieved_contexts: &[String],
        sample: &EvalSample,
        generated_answer: Option<&str>,
    ) -> EvalResult {
        let scores: HashMap<String, Option<f32>> = self
            .metrics
            .iter()
            .map(|m| {
                let score = m.score(
                    &sample.question,
                    retrieved_contexts,
                    generated_answer,
                    &sample.ground_truth_contexts,
                );
                (m.name().to_string(), score)
            })
            .collect();

        EvalResult {
            question: sample.question.clone(),
            retrieved_contexts: retrieved_contexts.to_vec(),
            generated_answer: generated_answer.map(String::from),
            scores,
        }
    }

    /// Summarise a batch of eval results with mean/median per metric.
    pub fn summarise(&self, results: &[EvalResult]) -> EvalSummary {
        let metric_names: Vec<String> = self.metrics.iter().map(|m| m.name().to_string()).collect();
        let mut metric_stats: HashMap<String, MetricStats> = HashMap::new();

        for name in &metric_names {
            let values: Vec<f32> = results
                .iter()
                .filter_map(|r| r.scores.get(name).and_then(|s| *s))
                .collect();
            metric_stats.insert(name.clone(), MetricStats::from_values(&values));
        }

        let overall_scores: Vec<f32> = results
            .iter()
            .filter_map(|r| r.overall_score(&self.score_weights))
            .collect();

        EvalSummary {
            count: results.len(),
            overall_mean: MetricStats::mean(&overall_scores),
            metrics: metric_stats,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetricStats {
    pub mean: Option<f32>,
    pub median: Option<f32>,
    pub stddev: Option<f32>,
    pub count: usize,
}

impl MetricStats {
    pub fn from_values(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self { mean: None, median: None, stddev: None, count: 0 };
        }
        let mean = Self::mean(values);
        let median = Self::median(values);
        let stddev = mean.map(|m| {
            let variance = values.iter().map(|v| (v - m).powi(2)).sum::<f32>() / values.len() as f32;
            variance.sqrt()
        });
        Self { mean, median, stddev, count: values.len() }
    }

    fn mean(values: &[f32]) -> Option<f32> {
        if values.is_empty() { return None; }
        Some(values.iter().sum::<f32>() / values.len() as f32)
    }

    fn median(values: &[f32]) -> Option<f32> {
        if values.is_empty() { return None; }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            Some((sorted[mid - 1] + sorted[mid]) / 2.0)
        } else {
            Some(sorted[mid])
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvalSummary {
    pub count: usize,
    pub overall_mean: Option<f32>,
    pub metrics: HashMap<String, MetricStats>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(question: &str, gt: &[&str]) -> EvalSample {
        EvalSample::new(question)
            .with_ground_truth(gt.iter().map(|s| s.to_string()).collect())
    }

    // --- ContextPrecision ---

    #[test]
    fn precision_all_relevant() {
        let metric = ContextPrecision;
        let retrieved = vec!["fact about rust".to_string()];
        let gt = vec!["fact about rust".to_string()];
        assert_eq!(metric.score("q", &retrieved, None, &gt), Some(1.0));
    }

    #[test]
    fn precision_none_relevant() {
        let metric = ContextPrecision;
        let retrieved = vec!["unrelated content".to_string()];
        let gt = vec!["rust programming".to_string()];
        assert_eq!(metric.score("q", &retrieved, None, &gt), Some(0.0));
    }

    #[test]
    fn precision_empty_retrieved_is_zero() {
        let metric = ContextPrecision;
        assert_eq!(metric.score("q", &[], None, &["gt".to_string()]), Some(0.0));
    }

    // --- ContextRecall ---

    #[test]
    fn recall_all_covered() {
        let metric = ContextRecall;
        let retrieved = vec!["rust is fast".to_string()];
        let gt = vec!["rust".to_string()];
        assert_eq!(metric.score("q", &retrieved, None, &gt), Some(1.0));
    }

    #[test]
    fn recall_empty_gt_is_one() {
        let metric = ContextRecall;
        assert_eq!(metric.score("q", &["any".to_string()], None, &[]), Some(1.0));
    }

    // --- Evaluator ---

    #[test]
    fn evaluator_produces_scores_for_all_metrics() {
        let eval = Evaluator::default_metrics();
        let s = sample("What is Rust?", &["Rust is a systems language"]);
        let retrieved = vec!["Rust is a systems language".to_string()];
        let result = eval.evaluate(&retrieved, &s, None);
        assert!(result.scores.contains_key("context_precision"));
        assert!(result.scores.contains_key("context_recall"));
        assert!(result.scores.contains_key("faithfulness"));
        assert!(result.scores.contains_key("answer_relevance"));
    }

    #[test]
    fn llm_metrics_return_none() {
        let eval = Evaluator::default_metrics();
        let s = sample("Q", &[]);
        let result = eval.evaluate(&[], &s, None);
        assert!(result.scores["faithfulness"].is_none());
        assert!(result.scores["answer_relevance"].is_none());
    }

    #[test]
    fn overall_score_uses_weights() {
        let mut scores = HashMap::new();
        scores.insert("precision".to_string(), Some(0.8_f32));
        scores.insert("recall".to_string(), Some(0.6_f32));
        let result = EvalResult {
            question: "q".to_string(),
            retrieved_contexts: Vec::new(),
            generated_answer: None,
            scores,
        };
        let mut weights = HashMap::new();
        weights.insert("precision".to_string(), 2.0_f32);
        weights.insert("recall".to_string(), 1.0_f32);
        // (0.8*2 + 0.6*1) / 3 = 2.2/3 ≈ 0.733
        let overall = result.overall_score(&weights).unwrap();
        assert!((overall - 2.2 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn summarise_produces_per_metric_stats() {
        let eval = Evaluator::default_metrics();
        let s = sample("Q", &["Rust"]);
        let r1 = eval.evaluate(&["Rust is fast".to_string()], &s, None);
        let r2 = eval.evaluate(&["Python is easy".to_string()], &s, None);
        let summary = eval.summarise(&[r1, r2]);
        assert_eq!(summary.count, 2);
        assert!(summary.metrics.contains_key("context_precision"));
    }
}
