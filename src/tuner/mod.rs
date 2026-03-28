//! Prompt optimisation and tuning strategies.
//!
//! Strategies: noop, MIPROv2, GEPA.

/// A prompt to be optimised.
pub struct Prompt {
    pub system: Option<String>,
    pub user: String,
}

/// The result of a tuning pass.
pub struct TuningResult {
    pub optimised_prompt: Prompt,
    pub score: f32,
    pub iterations: usize,
}

/// Interface for evaluating prompt quality.
pub trait TuningEvaluator: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn evaluate(&self, prompt: &Prompt, examples: &[(&str, &str)]) -> Result<f32, Self::Error>;
}

/// Interface for prompt optimisation strategies.
pub trait TuningStrategy: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn name(&self) -> &'static str;
    fn tune(
        &self,
        prompt: Prompt,
        evaluator: &dyn TuningEvaluator<Error = Self::Error>,
    ) -> Result<TuningResult, Self::Error>;
}
