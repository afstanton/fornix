//! Token-count chunker.
//!
//! Splits text into windows of at most `max_tokens` whitespace-delimited
//! tokens with optional overlap.

use serde_json::json;

use crate::rag::{chunkers::Chunker, tokenizer::token_spans, types::Chunk};

/// Splits text into windows measured in token count.
#[derive(Debug, Clone)]
pub struct TokenCount {
    pub max_tokens: usize,
    pub overlap: usize,
}

impl TokenCount {
    pub fn new(max_tokens: usize, overlap: usize) -> Self {
        assert!(max_tokens > 0, "max_tokens must be > 0");
        assert!(overlap < max_tokens, "overlap must be < max_tokens");
        Self { max_tokens, overlap }
    }
}

impl Default for TokenCount {
    fn default() -> Self {
        Self::new(200, 0)
    }
}

impl Chunker for TokenCount {
    fn name(&self) -> &'static str {
        "token_count"
    }

    fn chunk(&self, text: &str) -> Vec<Chunk> {
        let spans = token_spans(text);
        if spans.is_empty() {
            return Vec::new();
        }

        let step = (self.max_tokens - self.overlap).max(1);
        let mut chunks = Vec::new();
        let mut index = 0;
        let mut start_idx = 0;

        while start_idx < spans.len() {
            let end_idx = (start_idx + self.max_tokens - 1).min(spans.len() - 1);
            let start_offset = spans[start_idx].start;
            let end_offset = spans[end_idx].end;
            let content = &text[start_offset..end_offset];

            let mut metadata = crate::common::metadata::Metadata::new();
            metadata.insert("overlap_applied".to_string(), json!(self.overlap > 0));

            let mut chunk = Chunk::new(content, index, start_offset, end_offset);
            chunk.metadata = metadata;
            chunks.push(chunk);

            index += 1;
            start_idx += step;
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::chunkers::Chunker;

    #[test]
    fn empty_text_returns_no_chunks() {
        assert!(TokenCount::default().chunk("").is_empty());
    }

    #[test]
    fn whitespace_only_text_returns_no_chunks() {
        assert!(TokenCount::new(5, 0).chunk("   \n\t  ").is_empty());
    }

    #[test]
    fn fewer_tokens_than_max_is_one_chunk() {
        let chunks = TokenCount::new(10, 0).chunk("hello world");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "hello world");
    }

    #[test]
    fn splits_at_max_token_boundary() {
        // 6 tokens, max 3 → 2 chunks
        let chunks = TokenCount::new(3, 0).chunk("one two three four five six");
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].content, "one two three");
        assert_eq!(chunks[1].content, "four five six");
    }

    #[test]
    fn overlap_repeats_tokens() {
        // 6 tokens, max 4, overlap 2 → step 2 → [0..4], [2..6]
        let chunks = TokenCount::new(4, 2).chunk("a b c d e f");
        assert_eq!(chunks.len(), 3); // [a b c d], [c d e f], [e f]
        assert!(chunks[0].content.contains("a"));
        assert!(chunks[1].content.contains("c")); // overlap
    }

    #[test]
    fn chunk_offsets_roundtrip_to_original_text() {
        let text = "The quick brown fox jumps";
        let chunks = TokenCount::new(2, 0).chunk(text);
        for chunk in &chunks {
            assert_eq!(&text[chunk.start_offset..chunk.end_offset], chunk.content);
        }
    }

    #[test]
    fn chunk_indices_are_sequential() {
        let chunks = TokenCount::new(2, 0).chunk("a b c d e f");
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
        }
    }

    #[test]
    fn token_count_of_each_chunk_at_most_max_tokens() {
        let chunks = TokenCount::new(3, 0).chunk("one two three four five six seven");
        for chunk in &chunks {
            assert!(chunk.token_count() <= 3);
        }
    }

    #[test]
    fn name_is_token_count() {
        assert_eq!(TokenCount::default().name(), "token_count");
    }

    #[test]
    fn metadata_overlap_flag_set_correctly() {
        let no_overlap = TokenCount::new(5, 0).chunk("a b c d e f");
        assert_eq!(no_overlap[0].metadata["overlap_applied"], serde_json::json!(false));

        let with_overlap = TokenCount::new(5, 2).chunk("a b c d e f g h i j");
        assert_eq!(with_overlap[0].metadata["overlap_applied"], serde_json::json!(true));
    }
}
