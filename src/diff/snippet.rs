//! Text snippet extraction for document diffs.
//!
//! Given two versions of a text (or a set of structured units), this module
//! produces focused snippet pairs that highlight exactly what changed —
//! wrapping the changed tokens in `[[...]]` markers and trimming the
//! unchanged context to `context_tokens` tokens on either side.
//!
//! The [`focused_pair`] function handles the core use case: plain text vs
//! plain text. [`boundary_aware_stitched_pair`] handles structured documents
//! (e.g. regulatory sections) where a change near a unit boundary should
//! pull in adjacent units for context.
//!
//! Diffing is done at the word level using the `similar` crate (LCS-based,
//! the same algorithm as Ruby's `diff-lcs`).

use similar::{ChangeTag, TextDiff};

/// Delimiter used when stitching multiple units together.
const STITCH_DELIMITER: &str = "\n\n";

/// A structured document unit (e.g. a paragraph or section).
#[derive(Debug, Clone)]
pub struct Unit {
    pub id: String,
    pub position: usize,
    pub text: String,
}

/// The result of a boundary-aware stitched pair extraction.
#[derive(Debug, Clone)]
pub struct StitchedPair {
    pub previous_snippet: String,
    pub current_snippet: String,
    /// Whether adjacent units were included in the context window.
    pub stitched: bool,
    pub stitched_previous_ids: Vec<String>,
    pub stitched_current_ids: Vec<String>,
    pub stitched_previous_positions: Vec<usize>,
    pub stitched_current_positions: Vec<usize>,
}

// ============================================================================
// Public API
// ============================================================================

/// Extract a focused diff snippet pair from two plain texts.
///
/// Returns `(previous_snippet, current_snippet)` where each snippet
/// contains the changed tokens wrapped in `[[...]]` and up to
/// `context_tokens` unchanged tokens on either side. Long texts are
/// truncated to `snippet_max` characters.
pub fn focused_pair(
    previous_text: &str,
    current_text: &str,
    snippet_max: usize,
    context_tokens: usize,
) -> (String, String) {
    let previous = normalize_text(previous_text);
    let current = normalize_text(current_text);

    if previous == current {
        return (truncate(&previous, snippet_max), truncate(&current, snippet_max));
    }

    let prev_tokens: Vec<&str> = previous.split_whitespace().collect();
    let curr_tokens: Vec<&str> = current.split_whitespace().collect();

    // Run word-level LCS diff and collect old/new changed positions
    let (prev_changed, curr_changed) = changed_positions(&prev_tokens, &curr_tokens);

    if prev_changed.is_empty() && curr_changed.is_empty() {
        return (truncate(&previous, snippet_max), truncate(&current, snippet_max));
    }

    let prev_span = minmax_or_full(&prev_changed, prev_tokens.len());
    let curr_span = minmax_or_full(&curr_changed, curr_tokens.len());

    let prev_snippet = token_excerpt_with_delta(
        &prev_tokens,
        prev_span.0,
        prev_span.1,
        snippet_max,
        context_tokens,
    );
    let curr_snippet = token_excerpt_with_delta(
        &curr_tokens,
        curr_span.0,
        curr_span.1,
        snippet_max,
        context_tokens,
    );

    (prev_snippet, curr_snippet)
}

/// Extract a focused snippet pair from structured document units, pulling in
/// adjacent units when the change is near a unit boundary.
///
/// `stitch_radius` controls how many neighbouring units on each side are
/// included. `boundary_threshold_chars` controls the minimum common-prefix
/// or -suffix length below which a change is considered "boundary sensitive."
#[allow(clippy::too_many_arguments)]
pub fn boundary_aware_stitched_pair(
    previous_units: &[Unit],
    current_units: &[Unit],
    target_previous_id: &str,
    target_current_id: &str,
    snippet_max: usize,
    context_tokens: usize,
    stitch_radius: usize,
    stitch_max_chars: usize,
    boundary_threshold_chars: usize,
) -> StitchedPair {
    let target_previous = previous_units.iter().find(|u| u.id == target_previous_id);
    let target_current = current_units.iter().find(|u| u.id == target_current_id);

    let prev_target_text = normalize_text(target_previous.map(|u| u.text.as_str()).unwrap_or(""));
    let curr_target_text = normalize_text(target_current.map(|u| u.text.as_str()).unwrap_or(""));

    let boundary_sensitive = is_boundary_sensitive_change(
        &prev_target_text,
        &curr_target_text,
        boundary_threshold_chars,
    );

    let radius = if boundary_sensitive { stitch_radius } else { 0 };

    let prev_window = window_for_units(previous_units, target_previous_id, radius);
    let curr_window = window_for_units(current_units, target_current_id, radius);

    let mut prev_text = stitched_text(&prev_window, stitch_max_chars);
    let mut curr_text = stitched_text(&curr_window, stitch_max_chars);
    if prev_text.is_empty() {
        prev_text = prev_target_text.clone();
    }
    if curr_text.is_empty() {
        curr_text = curr_target_text.clone();
    }

    let (prev_snippet, curr_snippet) =
        focused_pair(&prev_text, &curr_text, snippet_max, context_tokens);

    let stitched = boundary_sensitive && (prev_window.len() > 1 || curr_window.len() > 1);

    StitchedPair {
        previous_snippet: prev_snippet,
        current_snippet: curr_snippet,
        stitched,
        stitched_previous_ids: prev_window.iter().map(|u| u.id.clone()).collect(),
        stitched_current_ids: curr_window.iter().map(|u| u.id.clone()).collect(),
        stitched_previous_positions: prev_window.iter().map(|u| u.position).collect(),
        stitched_current_positions: curr_window.iter().map(|u| u.position).collect(),
    }
}

// ============================================================================
// Core diff helpers
// ============================================================================

/// Return `(old_changed_positions, new_changed_positions)` using word-level
/// LCS diff. Position is the 0-based token index in each respective sequence.
fn changed_positions(old_tokens: &[&str], new_tokens: &[&str]) -> (Vec<usize>, Vec<usize>) {
    // similar::TextDiff operates on sequences of items. We feed it slices
    // of string slices and use iter_all_changes to walk the edit script.
    let diff = TextDiff::from_slices(old_tokens, new_tokens);

    let mut old_pos = 0usize;
    let mut new_pos = 0usize;
    let mut old_changed = Vec::new();
    let mut new_changed = Vec::new();

    for change in diff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Equal => {
                old_pos += 1;
                new_pos += 1;
            }
            ChangeTag::Delete => {
                old_changed.push(old_pos);
                old_pos += 1;
            }
            ChangeTag::Insert => {
                new_changed.push(new_pos);
                new_pos += 1;
            }
        }
    }

    (old_changed, new_changed)
}

/// Return `(min, max)` of changed positions, or `(0, len-1)` for the full
/// span if the positions list is empty.
fn minmax_or_full(positions: &[usize], seq_len: usize) -> (usize, usize) {
    if positions.is_empty() || seq_len == 0 {
        return (0, seq_len.saturating_sub(1));
    }
    let min = *positions.iter().min().unwrap();
    let max = *positions.iter().max().unwrap();
    (min, max)
}

/// Extract a focused token window around `[changed_start, changed_end]`
/// with `context_tokens` unchanged tokens on each side. The changed span
/// is marked with `[[...]]`. Leading/trailing ellipses indicate truncation.
fn token_excerpt_with_delta(
    tokens: &[&str],
    changed_start: usize,
    changed_end: usize,
    snippet_max: usize,
    context_tokens: usize,
) -> String {
    if tokens.is_empty() {
        return String::new();
    }

    let start_index = changed_start.saturating_sub(context_tokens);
    let end_index = (changed_end + context_tokens).min(tokens.len() - 1);

    let excerpt = &tokens[start_index..=end_index];
    if excerpt.is_empty() {
        return String::new();
    }

    // Re-anchor changed span into the excerpt's coordinate space
    let local_start = changed_start.saturating_sub(start_index);
    let local_end = (changed_end - start_index).min(excerpt.len() - 1);
    let local_end = local_end.max(local_start);

    let prefix = &excerpt[..local_start];
    let delta = &excerpt[local_start..=local_end];
    let suffix = if local_end + 1 < excerpt.len() { &excerpt[local_end + 1..] } else { &[] };

    let mut parts: Vec<String> = Vec::new();
    if !prefix.is_empty() {
        parts.push(prefix.join(" "));
    }
    if !delta.is_empty() {
        parts.push(format!("[[{}]]", delta.join(" ")));
    }
    if !suffix.is_empty() {
        parts.push(suffix.join(" "));
    }

    let mut text = parts.join(" ");
    if start_index > 0 {
        text = format!("...{}", text);
    }
    if end_index < tokens.len() - 1 {
        text = format!("{}...", text);
    }

    truncate(&text, snippet_max)
}

// ============================================================================
// Boundary / stitching helpers
// ============================================================================

/// Return the slice of units within `radius` positions of the target unit.
fn window_for_units<'a>(units: &'a [Unit], target_id: &str, radius: usize) -> Vec<&'a Unit> {
    let Some(idx) = units.iter().position(|u| u.id == target_id) else {
        return Vec::new();
    };
    let start = idx.saturating_sub(radius);
    let end = (idx + radius).min(units.len() - 1);
    units[start..=end].iter().collect()
}

/// Join a window of units into a single string, truncated to `max_chars`.
fn stitched_text(units: &[&Unit], max_chars: usize) -> String {
    let raw: String = units
        .iter()
        .map(|u| normalize_text(&u.text))
        .collect::<Vec<_>>()
        .join(STITCH_DELIMITER);
    truncate(&normalize_text(&raw), max_chars)
}

/// Returns `true` if the change between `prev` and `curr` is near a boundary —
/// i.e. the common prefix or suffix is shorter than `threshold_chars`.
fn is_boundary_sensitive_change(prev: &str, curr: &str, threshold_chars: usize) -> bool {
    if prev.is_empty() || curr.is_empty() || prev == curr {
        return false;
    }
    let prefix = common_prefix_len(prev, curr);
    let suffix = common_suffix_len(prev, curr, prefix);
    prefix <= threshold_chars || suffix <= threshold_chars
}

fn common_prefix_len(a: &str, b: &str) -> usize {
    a.chars()
        .zip(b.chars())
        .take_while(|(x, y)| x == y)
        .count()
}

fn common_suffix_len(a: &str, b: &str, prefix: usize) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let mut ai = a_chars.len();
    let mut bi = b_chars.len();
    let mut count = 0;
    while ai > prefix && bi > prefix && a_chars[ai - 1] == b_chars[bi - 1] {
        count += 1;
        ai -= 1;
        bi -= 1;
    }
    count
}

// ============================================================================
// Utility
// ============================================================================

fn normalize_text(text: &str) -> String {
    let s: String = text
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    s
}

fn truncate(text: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    // Truncate on a char boundary
    let mut end = max.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- normalize_text ---

    #[test]
    fn normalize_collapses_whitespace() {
        assert_eq!(normalize_text("  hello   world  "), "hello world");
    }

    #[test]
    fn normalize_empty_is_empty() {
        assert_eq!(normalize_text(""), "");
    }

    // --- truncate ---

    #[test]
    fn truncate_short_text_unchanged() {
        assert_eq!(truncate("hello", 100), "hello");
    }

    #[test]
    fn truncate_at_exact_length() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn truncate_longer_text() {
        assert_eq!(truncate("hello world", 5), "hello");
    }

    #[test]
    fn truncate_zero_returns_empty() {
        assert_eq!(truncate("anything", 0), "");
    }

    // --- common_prefix_len / common_suffix_len ---

    #[test]
    fn prefix_len_identical_strings() {
        assert_eq!(common_prefix_len("abc", "abc"), 3);
    }

    #[test]
    fn prefix_len_no_common() {
        assert_eq!(common_prefix_len("abc", "xyz"), 0);
    }

    #[test]
    fn prefix_len_partial() {
        assert_eq!(common_prefix_len("abcdef", "abcxyz"), 3);
    }

    #[test]
    fn suffix_len_identical() {
        assert_eq!(common_suffix_len("abcdef", "abcdef", 0), 6);
    }

    #[test]
    fn suffix_len_partial() {
        assert_eq!(common_suffix_len("prefix_abc", "other_abc", 0), 4);
    }

    // --- is_boundary_sensitive_change ---

    #[test]
    fn not_sensitive_when_equal() {
        assert!(!is_boundary_sensitive_change("same text", "same text", 10));
    }

    #[test]
    fn not_sensitive_when_empty() {
        assert!(!is_boundary_sensitive_change("", "new text", 10));
    }

    #[test]
    fn sensitive_when_prefix_short() {
        // Only 1 char in common at start → below threshold of 5
        assert!(is_boundary_sensitive_change("aXXXXXXXX", "aYYYYYYYY", 5));
    }

    #[test]
    fn not_sensitive_when_long_common_prefix_and_suffix() {
        // Long common prefix and suffix — change is in the middle
        let prev = "The quick brown fox jumps over the lazy dog";
        let curr = "The quick REPLACED fox jumps over the lazy dog";
        assert!(!is_boundary_sensitive_change(prev, curr, 3));
    }

    // --- changed_positions ---

    #[test]
    fn no_changes_returns_empty_vecs() {
        let tokens = vec!["hello", "world"];
        let (old, new) = changed_positions(&tokens, &tokens);
        assert!(old.is_empty());
        assert!(new.is_empty());
    }

    #[test]
    fn deletion_appears_in_old_positions() {
        let old = vec!["a", "b", "c"];
        let new = vec!["a", "c"];
        let (old_changed, new_changed) = changed_positions(&old, &new);
        assert!(old_changed.contains(&1)); // "b" deleted at position 1
        assert!(new_changed.is_empty());
    }

    #[test]
    fn insertion_appears_in_new_positions() {
        let old = vec!["a", "c"];
        let new = vec!["a", "b", "c"];
        let (old_changed, new_changed) = changed_positions(&old, &new);
        assert!(old_changed.is_empty());
        assert!(new_changed.contains(&1)); // "b" inserted at position 1
    }

    #[test]
    fn substitution_appears_in_both() {
        let old = vec!["hello", "world"];
        let new = vec!["hello", "rust"];
        let (old_changed, new_changed) = changed_positions(&old, &new);
        assert!(!old_changed.is_empty());
        assert!(!new_changed.is_empty());
    }

    // --- token_excerpt_with_delta ---

    #[test]
    fn empty_tokens_returns_empty() {
        assert_eq!(token_excerpt_with_delta(&[], 0, 0, 100, 3), "");
    }

    #[test]
    fn full_excerpt_no_context_needed() {
        let tokens = vec!["a", "b", "c"];
        let result = token_excerpt_with_delta(&tokens, 1, 1, 1000, 0);
        assert!(result.contains("[[b]]"));
        assert!(result.contains("..."));
    }

    #[test]
    fn context_tokens_included() {
        let tokens = vec!["before", "CHANGED", "after"];
        let result = token_excerpt_with_delta(&tokens, 1, 1, 1000, 1);
        assert!(result.contains("before"));
        assert!(result.contains("[[CHANGED]]"));
        assert!(result.contains("after"));
    }

    #[test]
    fn ellipsis_added_when_truncated_at_start() {
        let tokens: Vec<&str> = (0..20).map(|_| "x").collect();
        let result = token_excerpt_with_delta(&tokens, 15, 15, 1000, 2);
        assert!(result.starts_with("..."));
    }

    #[test]
    fn ellipsis_added_when_truncated_at_end() {
        let tokens: Vec<&str> = (0..20).map(|_| "x").collect();
        let result = token_excerpt_with_delta(&tokens, 2, 2, 1000, 2);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn result_truncated_to_snippet_max() {
        let tokens: Vec<&str> = vec!["hello", "world", "this", "is", "a", "test"];
        let result = token_excerpt_with_delta(&tokens, 1, 2, 5, 10);
        assert!(result.len() <= 5);
    }

    // --- focused_pair ---

    #[test]
    fn identical_texts_return_unchanged_pair() {
        let (prev, curr) = focused_pair("hello world", "hello world", 1000, 5);
        assert_eq!(prev, "hello world");
        assert_eq!(curr, "hello world");
    }

    #[test]
    fn changed_word_is_highlighted_in_each_snippet() {
        let prev_text = "The quick brown fox";
        let curr_text = "The quick red fox";
        let (prev, curr) = focused_pair(prev_text, curr_text, 1000, 5);
        assert!(prev.contains("[[brown]]"), "prev snippet: {}", prev);
        assert!(curr.contains("[[red]]"), "curr snippet: {}", curr);
    }

    #[test]
    fn addition_highlighted_in_current() {
        let (prev, curr) = focused_pair("hello world", "hello beautiful world", 1000, 2);
        assert!(curr.contains("[[beautiful]]"), "curr: {}", curr);
        // "beautiful" is inserted, so prev should mark the adjacent area
        assert!(!prev.contains("[[beautiful]]"));
    }

    #[test]
    fn deletion_highlighted_in_previous() {
        let (prev, curr) = focused_pair("hello cruel world", "hello world", 1000, 2);
        assert!(prev.contains("[[cruel]]"), "prev: {}", prev);
        assert!(!curr.contains("[[cruel]]"));
    }

    #[test]
    fn snippets_are_truncated_to_max() {
        let long = "word ".repeat(200);
        let modified = long.replacen("word", "WORD", 1);
        let (prev, curr) = focused_pair(&long, &modified, 50, 2);
        assert!(prev.len() <= 50);
        assert!(curr.len() <= 50);
    }

    #[test]
    fn empty_previous_returns_marked_current() {
        let (prev, curr) = focused_pair("", "hello world", 1000, 3);
        assert!(prev.is_empty() || prev.is_empty());
        assert!(!curr.is_empty());
    }

    // --- window_for_units ---

    fn units(ids: &[&str]) -> Vec<Unit> {
        ids.iter()
            .enumerate()
            .map(|(i, id)| Unit { id: id.to_string(), position: i, text: format!("text {}", id) })
            .collect()
    }

    #[test]
    fn window_radius_zero_returns_only_target() {
        let us = units(&["a", "b", "c"]);
        let w = window_for_units(&us, "b", 0);
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].id, "b");
    }

    #[test]
    fn window_radius_one_returns_three() {
        let us = units(&["a", "b", "c"]);
        let w = window_for_units(&us, "b", 1);
        assert_eq!(w.len(), 3);
    }

    #[test]
    fn window_clamps_at_boundaries() {
        let us = units(&["a", "b", "c"]);
        let w = window_for_units(&us, "a", 5);
        assert_eq!(w.len(), 3); // clamped to [0, 2]
    }

    #[test]
    fn window_missing_id_returns_empty() {
        let us = units(&["a", "b"]);
        let w = window_for_units(&us, "z", 1);
        assert!(w.is_empty());
    }

    // --- boundary_aware_stitched_pair ---

    #[test]
    fn stitched_pair_not_stitched_for_stable_boundaries() {
        let prev_units = vec![
            Unit { id: "1".into(), position: 0, text: "The quick brown fox jumps over the lazy dog".into() },
            Unit { id: "2".into(), position: 1, text: "Section two content here".into() },
        ];
        let curr_units = vec![
            Unit { id: "1".into(), position: 0, text: "The quick brown fox jumps over the FAST dog".into() },
            Unit { id: "2".into(), position: 1, text: "Section two content here".into() },
        ];
        let result = boundary_aware_stitched_pair(
            &prev_units, &curr_units,
            "1", "1",
            1000, 5, 2, 2000, 3,
        );
        // Change is in the middle, long common prefix/suffix → not boundary sensitive
        assert!(!result.stitched);
        assert!(result.previous_snippet.contains("[[brown]]") || result.previous_snippet.contains("[["));
    }

    #[test]
    fn stitched_pair_stitches_boundary_sensitive_change() {
        // Change starts at position 0 → very short prefix → boundary sensitive
        let prev_units = vec![
            Unit { id: "1".into(), position: 0, text: "Alpha section".into() },
            Unit { id: "2".into(), position: 1, text: "Beta section text".into() },
            Unit { id: "3".into(), position: 2, text: "Gamma section".into() },
        ];
        let curr_units = vec![
            Unit { id: "1".into(), position: 0, text: "Alpha section".into() },
            Unit { id: "2".into(), position: 1, text: "Changed section text".into() },
            Unit { id: "3".into(), position: 2, text: "Gamma section".into() },
        ];
        let result = boundary_aware_stitched_pair(
            &prev_units, &curr_units,
            "2", "2",
            2000, 3, 1, 5000, 100,
        );
        // "Changed" at the start → short prefix → boundary sensitive → stitched
        assert!(result.stitched);
        assert_eq!(result.stitched_previous_ids.len(), 3);
    }

    #[test]
    fn stitched_pair_ids_and_positions_populated() {
        let prev_units = vec![
            Unit { id: "sec-1".into(), position: 1, text: "First section.".into() },
            Unit { id: "sec-2".into(), position: 2, text: "Second section.".into() },
        ];
        let curr_units = prev_units.clone();
        let result = boundary_aware_stitched_pair(
            &prev_units, &curr_units,
            "sec-1", "sec-1",
            1000, 3, 1, 5000, 5,
        );
        assert!(!result.stitched_previous_ids.is_empty());
        assert!(!result.stitched_previous_positions.is_empty());
    }
}
