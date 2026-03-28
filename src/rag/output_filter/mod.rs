//! Output filter trait, pipeline, and audit types.

use crate::rag::{
    error::Result,
    types::{FilterAuditEntry, FilteredResult, RagResult},
};

/// An output filter that post-processes a [`RagResult`].
///
/// Filters may remove low-confidence contexts, deduplicate, truncate,
/// or transform the context list in any other way.
pub trait OutputFilter: Send + Sync {
    fn name(&self) -> &str;

    /// Apply the filter. Returns a new `RagResult` with the modified contexts.
    fn filter(&self, result: RagResult, query: &str) -> Result<RagResult>;
}

/// A sequential pipeline of output filters.
///
/// Each filter receives the result of the previous one. An audit trail
/// records the context count before and after each filter.
#[derive(Default)]
pub struct FilterPipeline {
    filters: Vec<Box<dyn OutputFilter>>,
}

impl FilterPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a filter to the pipeline.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, filter: impl OutputFilter + 'static) -> Self {
        self.filters.push(Box::new(filter));
        self
    }

    /// Execute all filters in order, returning the final result and audit trail.
    pub fn run(&self, result: RagResult, query: &str) -> Result<FilteredResult> {
        let mut current = result;
        let mut applied = Vec::new();

        for filter in &self.filters {
            let before = current.contexts.len();
            current = filter.filter(current, query)?;
            applied.push(FilterAuditEntry {
                filter: filter.name().to_string(),
                contexts_before: before,
                contexts_after: current.contexts.len(),
            });
        }

        Ok(FilteredResult { result: current, filters_applied: applied })
    }

    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    pub fn len(&self) -> usize {
        self.filters.len()
    }
}

/// A filter that removes contexts with score below a threshold.
pub struct MinScoreFilter {
    pub threshold: f32,
}

impl MinScoreFilter {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl OutputFilter for MinScoreFilter {
    fn name(&self) -> &str {
        "min_score"
    }

    fn filter(&self, mut result: RagResult, _query: &str) -> Result<RagResult> {
        let threshold = self.threshold;
        result.contexts.retain(|c| {
            c.effective_score().is_none_or(|s| s >= threshold)
        });
        result.token_count = result.contexts.iter().map(|c| crate::rag::tokenizer::count_tokens(&c.content)).sum();
        Ok(result)
    }
}

/// A filter that removes duplicate context content.
pub struct DeduplicateFilter;

impl OutputFilter for DeduplicateFilter {
    fn name(&self) -> &str {
        "deduplicate"
    }

    fn filter(&self, mut result: RagResult, _query: &str) -> Result<RagResult> {
        let mut seen = std::collections::HashSet::new();
        result.contexts.retain(|c| seen.insert(c.content.clone()));
        result.token_count = result.contexts.iter().map(|c| crate::rag::tokenizer::count_tokens(&c.content)).sum();
        Ok(result)
    }
}

/// A filter that truncates the context list to `max` items.
pub struct TruncateFilter {
    pub max: usize,
}

impl TruncateFilter {
    pub fn new(max: usize) -> Self {
        Self { max }
    }
}

impl OutputFilter for TruncateFilter {
    fn name(&self) -> &str {
        "truncate"
    }

    fn filter(&self, mut result: RagResult, _query: &str) -> Result<RagResult> {
        result.contexts.truncate(self.max);
        result.token_count = result.contexts.iter().map(|c| crate::rag::tokenizer::count_tokens(&c.content)).sum();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::types::{Context, RagResult};

    fn result_with(contents: &[&str], scores: &[Option<f32>]) -> RagResult {
        let contexts = contents
            .iter()
            .zip(scores.iter())
            .map(|(c, s)| {
                let ctx = Context::new(*c);
                if let Some(score) = s {
                    ctx.with_score(*score)
                } else {
                    ctx
                }
            })
            .collect();
        RagResult::new("q", "test", contexts)
    }

    // --- MinScoreFilter ---

    #[test]
    fn min_score_removes_below_threshold() {
        let r = result_with(&["hi", "lo"], &[Some(0.9), Some(0.3)]);
        let filtered = MinScoreFilter::new(0.5).filter(r, "q").unwrap();
        assert_eq!(filtered.contexts.len(), 1);
        assert_eq!(filtered.contexts[0].content, "hi");
    }

    #[test]
    fn min_score_keeps_contexts_without_score() {
        let r = result_with(&["no score"], &[None]);
        let filtered = MinScoreFilter::new(0.9).filter(r, "q").unwrap();
        assert_eq!(filtered.contexts.len(), 1);
    }

    // --- DeduplicateFilter ---

    #[test]
    fn deduplication_removes_exact_duplicates() {
        let r = result_with(&["dup", "dup", "unique"], &[None, None, None]);
        let filtered = DeduplicateFilter.filter(r, "q").unwrap();
        assert_eq!(filtered.contexts.len(), 2);
    }

    #[test]
    fn deduplication_preserves_order() {
        let r = result_with(&["a", "b", "a"], &[None, None, None]);
        let filtered = DeduplicateFilter.filter(r, "q").unwrap();
        assert_eq!(filtered.contexts[0].content, "a");
        assert_eq!(filtered.contexts[1].content, "b");
    }

    // --- TruncateFilter ---

    #[test]
    fn truncate_limits_count() {
        let r = result_with(&["a", "b", "c", "d"], &[None; 4]);
        let filtered = TruncateFilter::new(2).filter(r, "q").unwrap();
        assert_eq!(filtered.contexts.len(), 2);
    }

    #[test]
    fn truncate_larger_than_input_keeps_all() {
        let r = result_with(&["a", "b"], &[None, None]);
        let filtered = TruncateFilter::new(10).filter(r, "q").unwrap();
        assert_eq!(filtered.contexts.len(), 2);
    }

    // --- FilterPipeline ---

    #[test]
    fn empty_pipeline_returns_unchanged_result() {
        let r = result_with(&["a", "b", "c"], &[None; 3]);
        let pipeline = FilterPipeline::new();
        let fr = pipeline.run(r, "q").unwrap();
        assert_eq!(fr.result.contexts.len(), 3);
        assert!(fr.filters_applied.is_empty());
    }

    #[test]
    fn pipeline_applies_filters_in_order() {
        let r = result_with(&["a", "b", "b", "c"], &[Some(0.9), Some(0.2), Some(0.2), Some(0.8)]);
        let pipeline = FilterPipeline::new()
            .add(MinScoreFilter::new(0.5))
            .add(DeduplicateFilter)
            .add(TruncateFilter::new(1));

        let fr = pipeline.run(r, "q").unwrap();
        assert_eq!(fr.result.contexts.len(), 1);
        assert_eq!(fr.filters_applied.len(), 3);
    }

    #[test]
    fn audit_trail_records_counts() {
        let r = result_with(&["a", "b", "c"], &[Some(0.9), Some(0.1), Some(0.8)]);
        let pipeline = FilterPipeline::new().add(MinScoreFilter::new(0.5));
        let fr = pipeline.run(r, "q").unwrap();
        assert_eq!(fr.filters_applied[0].contexts_before, 3);
        assert_eq!(fr.filters_applied[0].contexts_after, 2);
    }

    #[test]
    fn pipeline_len_and_is_empty() {
        let empty = FilterPipeline::new();
        assert!(empty.is_empty());

        let nonempty = FilterPipeline::new().add(TruncateFilter::new(5));
        assert_eq!(nonempty.len(), 1);
    }
}
