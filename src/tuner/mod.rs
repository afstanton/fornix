//! Prompt optimisation via automatic instruction and few-shot selection.
//!
//! Strategies:
//! - [`Noop`] — pass-through, no optimisation
//! - [`MiproV2`] — instruction proposal + random trials with minibatch scoring
//! - [`Gepa`] — evolutionary population search with pareto-frontier selection
//!
//! All strategies take an injected `llm: &dyn Fn(&str) -> Result<String>` so
//! the LLM provider is fully decoupled from the optimisation logic.
//! Pass a closure backed by `rig`, `async-openai`, RubyLLM (via Magnus), or
//! a mock for testing.
//!
//! # Quick start
//!
//! ```rust
//! use fornix::tuner::{MiproV2, TuningStrategy, Sample, ExactMatchEvaluator};
//!
//! let dataset = vec![
//!     Sample::new("What is 2+2?").with_output("4"),
//!     Sample::new("Capital of France?").with_output("Paris"),
//! ];
//!
//! // Stub LLM — replace with your actual provider
//! let llm = |prompt: &str| -> fornix::tuner::Result<String> {
//!     Ok(format!("improved: {}", &prompt[..20.min(prompt.len())]))
//! };
//!
//! let result = MiproV2::default()
//!     .tune("Answer the question:", &dataset, &ExactMatchEvaluator, &llm)
//!     .unwrap();
//!
//! println!("Best prompt: {}", result.prompt);
//! ```

pub mod error;
pub mod primitives;
pub mod strategies;
pub mod types;

pub use error::{Error, Result};
pub use strategies::{Gepa, GepaParams, MiproV2, MiproV2Params, Noop, TuningStrategy};
pub use types::{Evaluator, ExactMatchEvaluator, Sample, SubstringEvaluator, TunerResult};
