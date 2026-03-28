//! Noop strategy — returns the prompt unchanged.

use crate::tuner::{
    error::Result,
    strategies::TuningStrategy,
    types::{Evaluator, Sample, TunerResult},
};

#[derive(Default)]
pub struct Noop;

impl TuningStrategy for Noop {
    fn name(&self) -> &'static str { "noop" }

    fn tune(
        &self,
        prompt: &str,
        _dataset: &[Sample],
        _evaluator: &dyn Evaluator,
        _llm: &dyn Fn(&str) -> Result<String>,
    ) -> Result<TunerResult> {
        Ok(TunerResult::new(prompt).with_meta("strategy", "noop"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuner::types::ExactMatchEvaluator;

    #[test]
    fn noop_returns_original_prompt() {
        let result = Noop
            .tune("my prompt", &[], &ExactMatchEvaluator, &|_| Ok("".to_string()))
            .unwrap();
        assert_eq!(result.prompt, "my prompt");
    }

    #[test]
    fn noop_score_is_none() {
        let result = Noop
            .tune("p", &[], &ExactMatchEvaluator, &|_| Ok("".to_string()))
            .unwrap();
        assert!(result.score.is_none());
    }

    #[test]
    fn noop_name() {
        assert_eq!(Noop.name(), "noop");
    }
}
