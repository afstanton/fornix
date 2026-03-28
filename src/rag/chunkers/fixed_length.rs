//! Fixed-length character chunker.
//!
//! Splits text into chunks of at most `max_length` bytes (characters for
//! ASCII text) with an optional overlapping window.

use crate::rag::{chunkers::Chunker, types::Chunk};
use serde_json::json;

/// Splits text into fixed-length character windows.
#[derive(Debug, Clone)]
pub struct FixedLength {
    /// Maximum characters per chunk.
    pub max_length: usize,
    /// Number of characters from the end of each chunk to repeat at the
    /// start of the next. Zero disables overlap.
    pub overlap: usize,
}

impl FixedLength {
    pub fn new(max_length: usize, overlap: usize) -> Self {
        assert!(max_length > 0, "max_length must be > 0");
        assert!(overlap < max_length, "overlap must be < max_length");
        Self { max_length, overlap }
    }
}

impl Default for FixedLength {
    fn default() -> Self {
        Self::new(1000, 0)
    }
}

impl Chunker for FixedLength {
    fn name(&self) -> &'static str {
        "fixed_length"
    }

    fn chunk(&self, text: &str) -> Vec<Chunk> {
        if text.is_empty() {
            return Vec::new();
        }

        let step = (self.max_length - self.overlap).max(1);
        let bytes = text.as_bytes();
        let total = bytes.len();
        let mut chunks = Vec::new();
        let mut index = 0;
        let mut start = 0;

        while start < total {
            let end = (start + self.max_length).min(total);
            // Ensure we slice on a char boundary
            let end = advance_to_char_boundary(text, end);
            let content = &text[start..end];

            let mut metadata = crate::common::metadata::Metadata::new();
            metadata.insert("overlap_applied".to_string(), json!(self.overlap > 0));

            let mut chunk = Chunk::new(content, index, start, end);
            chunk.metadata = metadata;
            chunks.push(chunk);

            index += 1;
            start += step;
        }

        chunks
    }
}

/// Advance `pos` forward until it falls on a UTF-8 character boundary.
fn advance_to_char_boundary(s: &str, mut pos: usize) -> usize {
    while pos < s.len() && !s.is_char_boundary(pos) {
        pos += 1;
    }
    pos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_returns_no_chunks() {
        assert!(FixedLength::default().chunk("").is_empty());
    }

    #[test]
    fn text_shorter_than_max_is_one_chunk() {
        let chunks = FixedLength::new(100, 0).chunk("hello");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "hello");
    }

    #[test]
    fn text_exactly_max_length_is_one_chunk() {
        let text = "a".repeat(10);
        let chunks = FixedLength::new(10, 0).chunk(&text);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn text_splits_into_multiple_chunks_without_overlap() {
        let text = "a".repeat(25);
        let chunks = FixedLength::new(10, 0).chunk(&text);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].content.len(), 10);
        assert_eq!(chunks[1].content.len(), 10);
        assert_eq!(chunks[2].content.len(), 5);
    }

    #[test]
    fn overlap_causes_repeated_content() {
        let text = "abcdefghij"; // 10 chars
        let chunks = FixedLength::new(6, 2).chunk(text);
        // step = 4: [0..6], [4..10]
        assert_eq!(chunks.len(), 2);
        assert_eq!(&chunks[0].content, "abcdef");
        assert_eq!(&chunks[1].content, "efghij");
    }

    #[test]
    fn chunk_offsets_are_correct() {
        let text = "hello world";
        let chunks = FixedLength::new(5, 0).chunk(text);
        assert_eq!(chunks[0].start_offset, 0);
        assert_eq!(chunks[0].end_offset, 5);
        assert_eq!(chunks[1].start_offset, 5);
    }

    #[test]
    fn chunk_indices_are_sequential() {
        let text = "a".repeat(30);
        let chunks = FixedLength::new(10, 0).chunk(&text);
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
        }
    }

    #[test]
    fn metadata_overlap_applied_flag() {
        let no_overlap = FixedLength::new(10, 0).chunk("hello world foo");
        assert_eq!(no_overlap[0].metadata["overlap_applied"], serde_json::json!(false));

        let with_overlap = FixedLength::new(10, 2).chunk("hello world foo");
        assert_eq!(with_overlap[0].metadata["overlap_applied"], serde_json::json!(true));
    }

    #[test]
    fn all_content_covered() {
        let text = "The quick brown fox jumps over the lazy dog.";
        let chunks = FixedLength::new(10, 0).chunk(text);
        let reconstructed: String = chunks.iter().map(|c| c.content.as_str()).collect();
        assert_eq!(reconstructed, text);
    }

    #[test]
    fn name_is_fixed_length() {
        assert_eq!(FixedLength::default().name(), "fixed_length");
    }
}
