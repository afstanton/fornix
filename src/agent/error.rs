//! Error types for the agent layer.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("configuration error: {0}")]
    Configuration(String),
    #[error("tool error: {0}")]
    Tool(String),
    #[error("policy blocked: {0}")]
    PolicyBlocked(String),
    #[error("max depth exceeded at depth {0}")]
    MaxDepthExceeded(usize),
    #[error("step budget exhausted")]
    StepBudgetExhausted,
    #[error("token budget exhausted: used {used}, max {max}")]
    TokenBudgetExhausted { used: usize, max: usize },
    #[error("time limit exceeded after {steps} step(s)")]
    TimeLimitExceeded { steps: usize },
    #[error("cancelled")]
    Cancelled,
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    pub fn config(msg: impl Into<String>) -> Self { Self::Configuration(msg.into()) }
    pub fn tool(msg: impl Into<String>) -> Self { Self::Tool(msg.into()) }
    pub fn blocked(msg: impl Into<String>) -> Self { Self::PolicyBlocked(msg.into()) }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_error() {
        assert_eq!(Error::config("no model").to_string(), "configuration error: no model");
    }

    #[test]
    fn token_budget_exhausted() {
        let e = Error::TokenBudgetExhausted { used: 5000, max: 4096 };
        assert!(e.to_string().contains("5000"));
    }

    #[test]
    fn cancelled_message() {
        assert_eq!(Error::Cancelled.to_string(), "cancelled");
    }
}
