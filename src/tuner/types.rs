//! Tuner domain types.

use std::collections::HashMap;

/// A single training/evaluation sample: an input paired with an expected output.
#[derive(Debug, Clone)]
pub struct Sample {
    pub input: String,
    pub expected_output: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Sample {
    pub fn new(input: impl Into<String>) -> Self {
        Self { input: input.into(), expected_output: None, metadata: HashMap::new() }
    }
    pub fn with_output(mut self, output: impl Into<String>) -> Self {
        self.expected_output = Some(output.into());
        self
    }
}

/// The result of a tuning run.
#[derive(Debug, Clone)]
pub struct TunerResult {
    /// The optimised prompt.
    pub prompt: String,
    /// Mean score achieved by the optimised prompt on the dataset.
    pub score: Option<f32>,
    /// Strategy-specific metadata (instruction used, demo set, iterations, etc.).
    pub metadata: HashMap<String, serde_json::Value>,
}

impl TunerResult {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self { prompt: prompt.into(), score: None, metadata: HashMap::new() }
    }
    pub fn with_score(mut self, score: f32) -> Self {
        self.score = Some(score);
        self
    }
    pub fn with_meta(mut self, key: impl Into<String>, val: impl Into<serde_json::Value>) -> Self {
        self.metadata.insert(key.into(), val.into());
        self
    }
}

/// An evaluator scores a (prompt, sample, model_output) triple.
/// Returns a score in any range; higher is better.
pub trait Evaluator: Send + Sync {
    fn score(&self, prompt: &str, sample: &Sample, output: &str) -> Option<f32>;
}

/// A simple exact-match evaluator: 1.0 if output matches expected, 0.0 otherwise.
#[derive(Default)]
pub struct ExactMatchEvaluator;

impl Evaluator for ExactMatchEvaluator {
    fn score(&self, _prompt: &str, sample: &Sample, output: &str) -> Option<f32> {
        let expected = sample.expected_output.as_deref()?;
        Some(if output.trim() == expected.trim() { 1.0 } else { 0.0 })
    }
}

/// A substring-match evaluator: 1.0 if output contains the expected string.
#[derive(Default)]
pub struct SubstringEvaluator;

impl Evaluator for SubstringEvaluator {
    fn score(&self, _prompt: &str, sample: &Sample, output: &str) -> Option<f32> {
        let expected = sample.expected_output.as_deref()?;
        Some(if output.contains(expected) { 1.0 } else { 0.0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_builder() {
        let s = Sample::new("input").with_output("expected");
        assert_eq!(s.input, "input");
        assert_eq!(s.expected_output.as_deref(), Some("expected"));
    }

    #[test]
    fn tuner_result_builder() {
        let r = TunerResult::new("prompt text").with_score(0.85);
        assert_eq!(r.prompt, "prompt text");
        assert!((r.score.unwrap() - 0.85).abs() < 1e-6);
    }

    #[test]
    fn exact_match_pass() {
        let e = ExactMatchEvaluator;
        let s = Sample::new("q").with_output("answer");
        assert_eq!(e.score("p", &s, "answer"), Some(1.0));
    }

    #[test]
    fn exact_match_fail() {
        let e = ExactMatchEvaluator;
        let s = Sample::new("q").with_output("answer");
        assert_eq!(e.score("p", &s, "wrong"), Some(0.0));
    }

    #[test]
    fn exact_match_no_expected_returns_none() {
        let e = ExactMatchEvaluator;
        assert!(e.score("p", &Sample::new("q"), "output").is_none());
    }

    #[test]
    fn substring_match() {
        let e = SubstringEvaluator;
        let s = Sample::new("q").with_output("key");
        assert_eq!(e.score("p", &s, "the key answer"), Some(1.0));
        assert_eq!(e.score("p", &s, "no match"), Some(0.0));
    }
}
