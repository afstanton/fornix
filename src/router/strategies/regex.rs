//! Regex-based routing strategy.
//!
//! Matches the query content against an ordered list of regex rules.
//! The first matching rule wins; if none match the default rule is used.

use regex::Regex;

use crate::router::{
    error::{Error, Result},
    strategies::RoutingStrategy,
    types::{ModelInfo, RoutingDecision},
};

/// A single regex routing rule.
#[derive(Debug, Clone)]
pub struct RegexRule {
    /// Compiled regex pattern.
    pub pattern: Regex,
    /// Model name to route to when the pattern matches.
    pub model: String,
    /// Provider for that model.
    pub provider: String,
    /// Optional human-readable explanation.
    pub reasoning: Option<String>,
}

impl RegexRule {
    /// Construct a rule, compiling the pattern.
    pub fn new(
        pattern: &str,
        model: impl Into<String>,
        provider: impl Into<String>,
    ) -> std::result::Result<Self, Error> {
        let compiled = Regex::new(pattern)
            .map_err(|e| Error::InvalidPattern(format!("{}: {}", pattern, e)))?;
        Ok(Self {
            pattern: compiled,
            model: model.into(),
            provider: provider.into(),
            reasoning: None,
        })
    }

    pub fn with_reasoning(mut self, r: impl Into<String>) -> Self {
        self.reasoning = Some(r.into());
        self
    }
}

/// Routes by matching query content against regex rules in order.
pub struct RegexStrategy {
    rules: Vec<RegexRule>,
    default_model: String,
    default_provider: String,
}

impl RegexStrategy {
    /// Construct a strategy with an ordered rule list and a default fallback.
    pub fn new(
        rules: Vec<RegexRule>,
        default_model: impl Into<String>,
        default_provider: impl Into<String>,
    ) -> Self {
        Self {
            rules,
            default_model: default_model.into(),
            default_provider: default_provider.into(),
        }
    }
}

impl RoutingStrategy for RegexStrategy {
    fn name(&self) -> &'static str {
        "regex"
    }

    fn route(
        &self,
        content: &str,
        _embedding: Option<&[f32]>,
        _models: &[ModelInfo],
    ) -> Result<RoutingDecision> {
        if let Some(rule) = self.rules.iter().find(|r| r.pattern.is_match(content)) {
            return Ok(RoutingDecision::new(&rule.model, &rule.provider)
                .with_reasoning(
                    rule.reasoning.clone().unwrap_or_else(|| "Matched regex rule".to_string()),
                ));
        }

        Ok(RoutingDecision::new(&self.default_model, &self.default_provider)
            .with_reasoning("Default rule (no regex matched)"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::types::ModelInfo;

    fn models() -> Vec<ModelInfo> {
        vec![
            ModelInfo::new("gpt-4o", "openai"),
            ModelInfo::new("gpt-4o-mini", "openai"),
        ]
    }

    fn strategy() -> RegexStrategy {
        RegexStrategy::new(
            vec![
                RegexRule::new(r"(?i)code|function|algorithm", "gpt-4o", "openai").unwrap(),
                RegexRule::new(r"(?i)summarize|tldr", "gpt-4o-mini", "openai").unwrap(),
            ],
            "gpt-4o-mini",
            "openai",
        )
    }

    #[test]
    fn matches_first_rule() {
        let d = strategy().route("Write a sorting algorithm", None, &models()).unwrap();
        assert_eq!(d.model, "gpt-4o");
    }

    #[test]
    fn matches_second_rule() {
        let d = strategy().route("Please summarize this article", None, &models()).unwrap();
        assert_eq!(d.model, "gpt-4o-mini");
    }

    #[test]
    fn falls_back_to_default() {
        let d = strategy().route("What is the weather?", None, &models()).unwrap();
        assert_eq!(d.model, "gpt-4o-mini");
        assert!(d.reasoning.as_deref().unwrap().contains("Default"));
    }

    #[test]
    fn case_insensitive_match() {
        let d = strategy().route("Write a CODE review", None, &models()).unwrap();
        assert_eq!(d.model, "gpt-4o");
    }

    #[test]
    fn first_matching_rule_wins() {
        // Both "code" and "summarize" in same prompt — first rule should win
        let d = strategy().route("summarize this code algorithm", None, &models()).unwrap();
        assert_eq!(d.model, "gpt-4o"); // code/algorithm rule is first
    }

    #[test]
    fn invalid_pattern_returns_error() {
        let result = RegexRule::new("[invalid(", "m", "p");
        assert!(result.is_err());
    }

    #[test]
    fn reasoning_carried_through() {
        let rule = RegexRule::new(r"test", "m", "p")
            .unwrap()
            .with_reasoning("Custom reason");
        let strat = RegexStrategy::new(vec![rule], "m", "p");
        let d = strat.route("this is a test", None, &[ModelInfo::new("m", "p")]).unwrap();
        assert_eq!(d.reasoning.as_deref(), Some("Custom reason"));
    }

    #[test]
    fn name_is_regex() {
        assert_eq!(strategy().name(), "regex");
    }
}
