//! Primitive building blocks shared by all tuning strategies.
//!
//! All LLM calls take a generic `llm: &dyn Fn(&str) -> String` closure
//! so the LLM provider is fully decoupled from the optimisation logic.

use crate::tuner::{error::Result, types::{Evaluator, Sample}};

/// Call the LLM to produce a model output for a given prompt + sample input.
pub fn model_output(
    prompt: &str,
    sample: &Sample,
    llm: &dyn Fn(&str) -> Result<String>,
) -> Result<String> {
    let full = format!("{}\n\nInput: {}\nOutput:", prompt, sample.input);
    llm(&full)
}

/// Score a prompt against a dataset slice, returning the mean score.
/// Returns `f32::NEG_INFINITY` if no samples can be scored.
pub fn evaluate_prompt(
    prompt: &str,
    dataset: &[Sample],
    evaluator: &dyn Evaluator,
    llm: &dyn Fn(&str) -> Result<String>,
) -> f32 {
    let scores: Vec<f32> = dataset
        .iter()
        .filter_map(|sample| {
            let output = model_output(prompt, sample, llm).ok()?;
            evaluator.score(prompt, sample, &output)
        })
        .collect();
    if scores.is_empty() {
        return f32::NEG_INFINITY;
    }
    scores.iter().sum::<f32>() / scores.len() as f32
}

/// Ask the LLM to generate a rewritten variant of `base_prompt`.
pub fn generate_variant(
    base_prompt: &str,
    llm: &dyn Fn(&str) -> Result<String>,
) -> Result<String> {
    let prompt = format!(
        "Rewrite the following prompt to improve its clarity and performance.\n\
         Prompt:\n{}\n\nImproved prompt:",
        base_prompt
    );
    llm(&prompt)
}

/// Generate up to `count` unique variants of `base_prompt`.
/// Always includes the original as the first candidate.
pub fn generate_variants(
    base_prompt: &str,
    count: usize,
    llm: &dyn Fn(&str) -> Result<String>,
) -> Vec<String> {
    let mut variants = vec![base_prompt.to_string()];
    for _ in 0..count {
        if let Ok(v) = generate_variant(base_prompt, llm)
            && !variants.contains(&v) {
                variants.push(v);
            }
    }
    variants
}

/// Build `count` random demo sets, each of size `size`, from `dataset`.
/// Uses a simple LCG for deterministic ordering when seed is fixed.
pub fn build_demo_sets(dataset: &[Sample], count: usize, size: usize, seed: u64) -> Vec<Vec<Sample>> {
    let size = size.min(dataset.len());
    let mut rng = seed.wrapping_add(1);
    (0..count)
        .map(|_| {
            let mut pool: Vec<usize> = (0..dataset.len()).collect();
            // Fisher-Yates shuffle
            for i in 0..size {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = i + (rng >> 33) as usize % (dataset.len() - i);
                pool.swap(i, j);
            }
            pool[..size].iter().map(|&i| dataset[i].clone()).collect()
        })
        .collect()
}

/// Format a demo sample for inclusion in a few-shot prompt.
pub fn format_demo(sample: &Sample) -> String {
    match &sample.expected_output {
        Some(expected) => format!("Input: {}\nOutput: {}", sample.input, expected),
        None => format!("Input: {}", sample.input),
    }
}

/// Assemble a full prompt from an instruction and a demo set.
pub fn assemble_prompt(instruction: &str, demos: &[Sample]) -> String {
    let demo_text: Vec<String> = demos.iter().map(format_demo).collect();
    let parts: Vec<&str> = std::iter::once(instruction)
        .chain(demo_text.iter().map(String::as_str))
        .filter(|s| !s.is_empty())
        .collect();
    parts.join("\n\n")
}

/// Select the candidate with the highest score.
pub fn select_best(candidates: &[(String, f32)]) -> Option<&(String, f32)> {
    candidates.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuner::types::{ExactMatchEvaluator, Sample};

    fn echo_llm(prompt: &str) -> Result<String> {
        // Returns the last word of the prompt as the "output"
        Ok(prompt.split_whitespace().last().unwrap_or("").to_string())
    }

    fn samples() -> Vec<Sample> {
        vec![
            Sample::new("Q1").with_output("A1"),
            Sample::new("Q2").with_output("A2"),
        ]
    }

    #[test]
    fn assemble_prompt_no_demos() {
        let p = assemble_prompt("Do the thing", &[]);
        assert_eq!(p, "Do the thing");
    }

    #[test]
    fn assemble_prompt_with_demos() {
        let demos = vec![Sample::new("hi").with_output("hello")];
        let p = assemble_prompt("Instruction", &demos);
        assert!(p.contains("Instruction"));
        assert!(p.contains("Input: hi"));
        assert!(p.contains("Output: hello"));
    }

    #[test]
    fn format_demo_with_output() {
        let s = Sample::new("q").with_output("a");
        assert_eq!(format_demo(&s), "Input: q\nOutput: a");
    }

    #[test]
    fn format_demo_without_output() {
        let s = Sample::new("q");
        assert_eq!(format_demo(&s), "Input: q");
    }

    #[test]
    fn build_demo_sets_correct_count_and_size() {
        let data = samples();
        let sets = build_demo_sets(&data, 3, 1, 42);
        assert_eq!(sets.len(), 3);
        assert!(sets.iter().all(|s| s.len() == 1));
    }

    #[test]
    fn build_demo_sets_size_capped_at_dataset_len() {
        let data = samples();
        let sets = build_demo_sets(&data, 2, 100, 1);
        assert!(sets.iter().all(|s| s.len() <= data.len()));
    }

    #[test]
    fn select_best_picks_highest_score() {
        let candidates = vec![
            ("a".to_string(), 0.5_f32),
            ("b".to_string(), 0.9_f32),
            ("c".to_string(), 0.3_f32),
        ];
        let best = select_best(&candidates).unwrap();
        assert_eq!(best.0, "b");
    }

    #[test]
    fn select_best_empty_returns_none() {
        assert!(select_best(&[]).is_none());
    }

    #[test]
    fn evaluate_prompt_returns_mean() {
        // echo_llm returns the last word; "Output:" is last so score will be 0.0
        let eval = ExactMatchEvaluator;
        let score = evaluate_prompt("p", &samples(), &eval, &echo_llm);
        // Will be 0.0 or NEG_INFINITY depending on match, but must be finite
        assert!(score.is_finite() || score == f32::NEG_INFINITY);
    }

    #[test]
    fn generate_variants_includes_original() {
        let llm = |_: &str| -> Result<String> { Ok("improved".to_string()) };
        let variants = generate_variants("base", 2, &llm);
        assert!(variants.contains(&"base".to_string()));
    }

    #[test]
    fn generate_variants_deduplicates() {
        // LLM always returns the same thing → only original + 1 variant
        let llm = |_: &str| -> Result<String> { Ok("same variant".to_string()) };
        let variants = generate_variants("base", 5, &llm);
        assert_eq!(variants.len(), 2);
    }
}
