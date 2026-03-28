//! GEPA — Generative Evolutionary Prompt Algorithm.
//!
//! Maintains a population of candidate prompts and iteratively improves them
//! via LLM-guided mutation, keeping the best candidates on a Pareto frontier.

use crate::tuner::{
    error::Result,
    primitives,
    strategies::TuningStrategy,
    types::{Evaluator, Sample, TunerResult},
};

/// GEPA hyperparameters.
#[derive(Debug, Clone)]
pub struct GepaParams {
    pub population_size: usize,
    pub iterations: usize,
    /// Score improvement threshold to count as real progress.
    pub improvement_threshold: f32,
    /// How many top candidates to keep on the Pareto frontier.
    pub pareto_set_size: usize,
    /// Number of feedback samples used for mutation context.
    pub feedback_size: usize,
    /// Seed for stochastic operations.
    pub seed: u64,
}

impl Default for GepaParams {
    fn default() -> Self {
        Self {
            population_size: 4,
            iterations: 6,
            improvement_threshold: 0.01,
            pareto_set_size: 5,
            feedback_size: 3,
            seed: 42,
        }
    }
}

pub struct Gepa {
    params: GepaParams,
}

impl Gepa {
    pub fn new(params: GepaParams) -> Self { Self { params } }
}

impl Default for Gepa {
    fn default() -> Self { Self::new(GepaParams::default()) }
}

impl TuningStrategy for Gepa {
    fn name(&self) -> &'static str { "gepa" }

    fn tune(
        &self,
        prompt: &str,
        dataset: &[Sample],
        evaluator: &dyn Evaluator,
        llm: &dyn Fn(&str) -> Result<String>,
    ) -> Result<TunerResult> {
        if dataset.is_empty() {
            return Err(crate::tuner::error::Error::config("dataset is required"));
        }

        let p = &self.params;

        // Split dataset into feedback set and evaluation set
        let feedback_size = p.feedback_size.min(dataset.len());
        let mut rng = p.seed.wrapping_add(1);
        let mut pool: Vec<usize> = (0..dataset.len()).collect();
        for i in 0..feedback_size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = i + (rng >> 33) as usize % (dataset.len() - i);
            pool.swap(i, j);
        }
        let feedback_set: Vec<Sample> = pool[..feedback_size].iter().map(|&i| dataset[i].clone()).collect();
        let eval_set: Vec<Sample> = if pool.len() > feedback_size {
            pool[feedback_size..].iter().map(|&i| dataset[i].clone()).collect()
        } else {
            dataset.to_vec()
        };

        let score_on_eval = |candidate: &str| -> f32 {
            primitives::evaluate_prompt(candidate, &eval_set, evaluator, llm)
        };

        // Seed population with LLM-generated variants
        let mut population: Vec<(String, f32)> = {
            let mut pop = Vec::new();
            for _ in 0..p.population_size {
                let mutated = mutate_prompt(prompt, &feedback_set, llm)?;
                let score = score_on_eval(&mutated);
                pop.push((mutated, score));
            }
            let base_score = score_on_eval(prompt);
            pop.push((prompt.to_string(), base_score));
            pop
        };

        // Evolutionary iterations
        for _ in 0..p.iterations {
            // Pareto frontier = top-N by score
            population.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let frontier_end = p.pareto_set_size.min(population.len());
            let frontier = &population[..frontier_end];

            // Sample a parent from frontier
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let parent_idx = (rng >> 33) as usize % frontier.len();
            let parent = &frontier[parent_idx].0.clone();

            let child = mutate_prompt(parent, &feedback_set, llm)?;
            let child_score = score_on_eval(&child);
            population.push((child, child_score));
            // Keep population bounded
            population.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            population.truncate(p.population_size + 1);
        }

        population.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let (best_prompt, best_score) = population.into_iter().next()
            .unwrap_or_else(|| (prompt.to_string(), f32::NEG_INFINITY));

        Ok(TunerResult::new(best_prompt)
            .with_score(best_score)
            .with_meta("strategy", "gepa")
            .with_meta("iterations", p.iterations as i64))
    }
}

fn mutate_prompt(
    prompt: &str,
    feedback_set: &[Sample],
    llm: &dyn Fn(&str) -> Result<String>,
) -> Result<String> {
    let feedback: Vec<String> = feedback_set.iter().map(|s| {
        let expected = s.expected_output.as_deref().unwrap_or("(unknown)");
        format!("Input: {}\nExpected: {}", s.input, expected)
    }).collect();

    let mutation_prompt = format!(
        "Improve the following prompt based on the feedback examples.\n\
         Prompt:\n{}\n\nFeedback examples:\n{}\n\nImproved prompt:",
        prompt,
        feedback.join("\n")
    );
    llm(&mutation_prompt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuner::types::{ExactMatchEvaluator, Sample};

    fn samples() -> Vec<Sample> {
        (0..5).map(|i| Sample::new(format!("q{}", i)).with_output(format!("a{}", i))).collect()
    }

    fn stub_llm(prompt: &str) -> Result<String> {
        let _ = prompt;
        Ok("mutated".to_string())
    }

    #[test]
    fn gepa_returns_result() {
        let result = Gepa::default()
            .tune("base", &samples(), &ExactMatchEvaluator, &stub_llm)
            .unwrap();
        assert!(!result.prompt.is_empty());
    }

    #[test]
    fn gepa_empty_dataset_errors() {
        assert!(Gepa::default()
            .tune("p", &[], &ExactMatchEvaluator, &stub_llm)
            .is_err());
    }

    #[test]
    fn gepa_score_is_set() {
        let result = Gepa::default()
            .tune("p", &samples(), &ExactMatchEvaluator, &stub_llm)
            .unwrap();
        assert!(result.score.is_some());
    }

    #[test]
    fn gepa_name() {
        assert_eq!(Gepa::default().name(), "gepa");
    }
}
