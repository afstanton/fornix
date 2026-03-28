//! MIPROv2 — Multi-prompt Instruction Proposal & Optimisation v2.
//!
//! Generates a set of instruction candidates, pairs each with a random demo
//! set, scores on a minibatch, then runs random trials with periodic full
//! evaluations to find the best instruction+demo combination.

use crate::tuner::{
    error::Result,
    primitives,
    strategies::TuningStrategy,
    types::{Evaluator, Sample, TunerResult},
};

/// MIPROv2 hyperparameters.
#[derive(Debug, Clone)]
pub struct MiproV2Params {
    /// Number of instruction variants to generate (in addition to the original).
    pub proposal_count: usize,
    /// Number of random demo sets to build.
    pub demo_sets: usize,
    /// Number of random (instruction, demo) trials after the initial sweep.
    pub max_trials: usize,
    /// Minibatch size for fast evaluation during trials.
    pub minibatch_size: usize,
    /// Run a full dataset evaluation every N trials.
    pub full_eval_every: usize,
    /// Seed for the demo-set RNG.
    pub seed: u64,
}

impl Default for MiproV2Params {
    fn default() -> Self {
        Self {
            proposal_count: 4,
            demo_sets: 3,
            max_trials: 6,
            minibatch_size: 3,
            full_eval_every: 3,
            seed: 42,
        }
    }
}

pub struct MiproV2 {
    params: MiproV2Params,
}

impl MiproV2 {
    pub fn new(params: MiproV2Params) -> Self { Self { params } }
}

impl Default for MiproV2 {
    fn default() -> Self { Self::new(MiproV2Params::default()) }
}

impl TuningStrategy for MiproV2 {
    fn name(&self) -> &'static str { "mipro_v2" }

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
        let instructions = primitives::generate_variants(prompt, p.proposal_count, llm);
        let demo_sets = primitives::build_demo_sets(dataset, p.demo_sets, p.minibatch_size, p.seed);

        let mut best_prompt = prompt.to_string();
        let mut best_score = f32::NEG_INFINITY;

        // Initial sweep: score each instruction with a random demo set
        for instruction in &instructions {
            let demos = demo_sets.first().cloned().unwrap_or_default();
            let candidate = primitives::assemble_prompt(instruction, &demos);
            let minibatch: Vec<Sample> = dataset
                .iter()
                .take(p.minibatch_size.max(1))
                .cloned()
                .collect();
            let score = primitives::evaluate_prompt(&candidate, &minibatch, evaluator, llm);
            if score > best_score {
                best_score = score;
                best_prompt = candidate;
            }
        }

        // Random trials
        let mut rng = p.seed.wrapping_add(99);
        for trial in 0..p.max_trials {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let instr = &instructions[(rng >> 33) as usize % instructions.len()];
            let demos = if demo_sets.is_empty() {
                vec![]
            } else {
                let i = (rng >> 17) as usize % demo_sets.len();
                demo_sets[i].clone()
            };
            let candidate = primitives::assemble_prompt(instr, &demos);
            let minibatch: Vec<Sample> = dataset
                .iter()
                .take(p.minibatch_size.max(1))
                .cloned()
                .collect();
            let score = primitives::evaluate_prompt(&candidate, &minibatch, evaluator, llm);
            if score > best_score {
                best_score = score;
                best_prompt = candidate.clone();
            }

            // Periodic full evaluation on best candidate
            if p.full_eval_every > 0 && (trial + 1) % p.full_eval_every == 0 {
                let full_score = primitives::evaluate_prompt(&best_prompt, dataset, evaluator, llm);
                if full_score > best_score {
                    best_score = full_score;
                }
            }
        }

        Ok(TunerResult::new(best_prompt)
            .with_score(best_score)
            .with_meta("strategy", "mipro_v2")
            .with_meta("trials", p.max_trials as i64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuner::types::{ExactMatchEvaluator, Sample};

    fn samples() -> Vec<Sample> {
        (0..5).map(|i| Sample::new(format!("q{}", i)).with_output(format!("a{}", i))).collect()
    }

    fn pass_through_llm(prompt: &str) -> Result<String> {
        // Returns empty string to simulate zero-match output
        let _ = prompt;
        Ok(String::new())
    }

    #[test]
    fn mipro_v2_returns_result() {
        let result = MiproV2::default()
            .tune("base prompt", &samples(), &ExactMatchEvaluator, &pass_through_llm)
            .unwrap();
        assert!(!result.prompt.is_empty());
    }

    #[test]
    fn mipro_v2_empty_dataset_errors() {
        let err = MiproV2::default()
            .tune("p", &[], &ExactMatchEvaluator, &pass_through_llm)
            .unwrap_err();
        assert!(matches!(err, crate::tuner::error::Error::Configuration(_)));
    }

    #[test]
    fn mipro_v2_score_is_set() {
        let result = MiproV2::default()
            .tune("p", &samples(), &ExactMatchEvaluator, &pass_through_llm)
            .unwrap();
        assert!(result.score.is_some());
    }

    #[test]
    fn mipro_v2_name() {
        assert_eq!(MiproV2::default().name(), "mipro_v2");
    }

    #[test]
    fn mipro_v2_metadata_contains_strategy() {
        let result = MiproV2::default()
            .tune("p", &samples(), &ExactMatchEvaluator, &pass_through_llm)
            .unwrap();
        assert_eq!(result.metadata.get("strategy").and_then(|v| v.as_str()), Some("mipro_v2"));
    }
}
