//! RAG tokenizer — token span extraction.
//!
//! Locates non-whitespace tokens in a string and records their byte offsets.
//! Used by token-aware chunkers to split text at token boundaries.

/// A located token: its string content and byte-offset span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenSpan {
    pub token: String,
    /// Byte offset of the first character (inclusive).
    pub start: usize,
    /// Byte offset past the last character (exclusive).
    pub end: usize,
}

impl TokenSpan {
    pub fn new(token: impl Into<String>, start: usize, end: usize) -> Self {
        Self { token: token.into(), start, end }
    }

    /// Number of bytes in this token.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Extract all non-whitespace token spans from `text`.
///
/// Each span records the token text and its byte offsets in the original
/// string. Works on UTF-8 text; offsets are byte-based, not char-based.
pub fn token_spans(text: &str) -> Vec<TokenSpan> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut spans = Vec::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        // Skip whitespace
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        // Scan to end of token
        let start = i;
        while i < len && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let end = i;
        let token = &text[start..end];
        spans.push(TokenSpan::new(token, start, end));
    }

    spans
}

/// Count the number of whitespace-delimited tokens in `text`.
pub fn count_tokens(text: &str) -> usize {
    token_spans(text).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string_returns_no_spans() {
        assert!(token_spans("").is_empty());
    }

    #[test]
    fn single_word_produces_one_span() {
        let spans = token_spans("hello");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].token, "hello");
        assert_eq!(spans[0].start, 0);
        assert_eq!(spans[0].end, 5);
    }

    #[test]
    fn two_words_produce_two_spans() {
        let spans = token_spans("hello world");
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].token, "hello");
        assert_eq!(spans[1].token, "world");
    }

    #[test]
    fn leading_whitespace_is_skipped() {
        let spans = token_spans("  word");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].start, 2);
    }

    #[test]
    fn trailing_whitespace_is_ignored() {
        let spans = token_spans("word  ");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].end, 4);
    }

    #[test]
    fn multiple_spaces_between_words() {
        let spans = token_spans("a   b");
        assert_eq!(spans.len(), 2);
    }

    #[test]
    fn span_byte_offsets_reconstruct_token() {
        let text = "foo bar baz";
        let spans = token_spans(text);
        for span in &spans {
            assert_eq!(&text[span.start..span.end], span.token);
        }
    }

    #[test]
    fn punctuation_is_treated_as_part_of_token() {
        let spans = token_spans("hello, world!");
        // "hello," and "world!" are single tokens (no whitespace inside)
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].token, "hello,");
        assert_eq!(spans[1].token, "world!");
    }

    #[test]
    fn newlines_and_tabs_split_tokens() {
        let spans = token_spans("a\tb\nc");
        assert_eq!(spans.len(), 3);
    }

    #[test]
    fn count_tokens_matches_span_count() {
        let text = "the quick brown fox";
        assert_eq!(count_tokens(text), token_spans(text).len());
        assert_eq!(count_tokens(text), 4);
    }

    #[test]
    fn count_tokens_empty_is_zero() {
        assert_eq!(count_tokens(""), 0);
    }

    #[test]
    fn span_len_equals_token_byte_length() {
        let spans = token_spans("foo");
        assert_eq!(spans[0].len(), 3);
    }
}
