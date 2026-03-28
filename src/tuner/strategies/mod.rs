//! Tuning strategy trait.

use crate::tuner::{error::Result, types::{Evaluator, Sample, TunerResult}};

pub trait TuningStrategy: Send + Sync {
    fn name(&self) -> &'static str;
    fn tune(
        &self,
        prompt: &str,
        dataset: &[Sample],
        evaluator: &dyn Evaluator,
        llm: &dyn Fn(&str) -> Result<String>,
    ) -> Result<TunerResult>;
}

pub mod gepa;
pub mod mipro_v2;
pub mod noop;

pub use gepa::Gepa;
pub use mipro_v2::MiproV2;
pub use noop::Noop;
