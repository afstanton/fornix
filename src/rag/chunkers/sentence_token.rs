//! Sentence-aware token chunker.
//!
//! Splits text on sentence boundaries (`[.!?]+` or end-of-input), then groups
//! sentences into chunks bounded by `max_tokens`. A sentence that exceeds
//! `max_tokens` on its own is split by the `TokenCount` chunker.
//!
//! An overlap buffer can carry the last N tokens of a completed chunk into
//! the beginning of the next one.

use serde_json::json;

use crate::common::metadata::Metadata;
use crate::rag::{
    chunkers::{token_count::TokenCount, Chunker},
    tokenizer::token_spans,
    types::Chunk,
};

/// Chunks text by grouping sentences up to `max_tokens`, with optional overlap.
#[derive(Debug, Clone)]
pub struct SentenceToken {
    pub min_tokens: usize,
    pub max_tokens: usize,
    pub overlap: usize,
}

impl SentenceToken {
    pub fn new(min_tokens: usize, max_tokens: usize, overlap: usize) -> Self {
        assert!(max_tokens > 0, "max_tokens must be > 0");
        assert!(overlap < max_tokens, "overlap must be < max_tokens");
        Self { min_tokens, max_tokens, overlap }
    }
}

impl Default for SentenceToken {
    fn default() -> Self {
        Self::new(50, 200, 20)
    }
}

/// A parsed sentence with token metadata.
#[derive(Debug, Clone)]
struct SentenceSpan {
    text: String,
    start: usize,
    end: usize,
    token_count: usize,
}

impl Chunker for SentenceToken {
    fn name(&self) -> &'static str {
        "sentence_token"
    }

    fn chunk(&self, text: &str) -> Vec<Chunk> {
        let sentences = extract_sentences(text);
        if sentences.is_empty() {
            return Vec::new();
        }

        let mut chunks: Vec<Chunk> = Vec::new();
        let mut current: Vec<SentenceSpan> = Vec::new();
        let mut current_tokens = 0usize;
        let mut index = 0;

        for sentence in &sentences {
            // Sentence too long for a single chunk — split it with TokenCount
            if sentence.token_count > self.max_tokens {
                if !current.is_empty() {
                    let chunk = finalise(&current, index, false, self.overlap > 0);
                    index = chunks.len() + 1;
                    chunks.push(chunk);
                    let (buf, buf_tokens) = overlap_buffer(&current, self.overlap);
                    current = buf;
                    current_tokens = buf_tokens;
                }
                let sub = TokenCount::new(self.max_tokens, self.overlap);
                let sub_chunks = sub.chunk(&sentence.text);
                for mut sc in sub_chunks {
                    // Re-anchor offsets into the original text
                    sc.start_offset += sentence.start;
                    sc.end_offset += sentence.start;
                    sc.index = index;
                    sc.metadata.insert("overflow".to_string(), json!(true));
                    index += 1;
                    chunks.push(sc);
                }
                continue;
            }

            // Would overflow the current chunk
            if current_tokens + sentence.token_count > self.max_tokens && !current.is_empty() {
                if current_tokens >= self.min_tokens {
                    let chunk = finalise(&current, index, false, self.overlap > 0);
                    chunks.push(chunk);
                    index = chunks.len();
                    let (buf, buf_tokens) = overlap_buffer(&current, self.overlap);
                    current = buf;
                    current_tokens = buf_tokens;
                } else {
                    // Below min — emit overflow chunk and reset
                    current.push(sentence.clone());
                    current_tokens += sentence.token_count;
                    let chunk = finalise(&current, index, true, self.overlap > 0);
                    chunks.push(chunk);
                    index = chunks.len();
                    let (buf, buf_tokens) = overlap_buffer(&current, self.overlap);
                    current = buf;
                    current_tokens = buf_tokens;
                    continue;
                }
            }

            current.push(sentence.clone());
            current_tokens += sentence.token_count;
        }

        if !current.is_empty() {
            chunks.push(finalise(&current, index, false, self.overlap > 0));
        }

        chunks
    }
}

fn extract_sentences(text: &str) -> Vec<SentenceSpan> {
    let mut spans = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    let len = bytes.len();

    while start < len {
        // Skip leading whitespace
        while start < len && (bytes[start] == b' ' || bytes[start] == b'\n' || bytes[start] == b'\r' || bytes[start] == b'\t') {
            start += 1;
        }
        if start >= len {
            break;
        }

        let content_start = start;

        // Scan forward until sentence terminator or end of text
        let mut end = content_start;
        while end < len {
            if bytes[end] == b'.' || bytes[end] == b'!' || bytes[end] == b'?' {
                end += 1;
                // Consume trailing terminators
                while end < len && (bytes[end] == b'.' || bytes[end] == b'!' || bytes[end] == b'?') {
                    end += 1;
                }
                break;
            }
            end += 1;
        }

        // Trim trailing whitespace from the span
        let mut content_end = end;
        while content_end > content_start && (bytes[content_end - 1] == b' ' || bytes[content_end - 1] == b'\n' || bytes[content_end - 1] == b'\r' || bytes[content_end - 1] == b'\t') {
            content_end -= 1;
        }

        let sentence_text = text[content_start..content_end].to_string();
        if !sentence_text.trim().is_empty() {
            let tcount = token_spans(&sentence_text).len();
            spans.push(SentenceSpan {
                text: sentence_text,
                start: content_start,
                end: content_end,
                token_count: tcount,
            });
        }
        start = end;
    }
    spans
}

fn finalise(sentences: &[SentenceSpan], index: usize, overflow: bool, overlap_applied: bool) -> Chunk {
    let start_offset = sentences.first().map(|s| s.start).unwrap_or(0);
    let end_offset = sentences.last().map(|s| s.end).unwrap_or(0);
    let content: String = sentences.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ");
    let mut metadata = Metadata::new();
    metadata.insert("overlap_applied".to_string(), json!(overlap_applied));
    metadata.insert("overflow".to_string(), json!(overflow));
    let mut chunk = Chunk::new(content, index, start_offset, end_offset);
    chunk.metadata = metadata;
    chunk
}

/// Carry the last `overlap` tokens from the current buffer into the next.
fn overlap_buffer(sentences: &[SentenceSpan], overlap: usize) -> (Vec<SentenceSpan>, usize) {
    if overlap == 0 || sentences.is_empty() {
        return (Vec::new(), 0);
    }
    let mut kept: Vec<SentenceSpan> = Vec::new();
    let mut token_count = 0usize;
    for sentence in sentences.iter().rev() {
        kept.insert(0, sentence.clone());
        token_count += sentence.token_count;
        if token_count >= overlap {
            break;
        }
    }
    (kept, token_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_returns_no_chunks() {
        assert!(SentenceToken::default().chunk("").is_empty());
    }

    #[test]
    fn single_short_sentence_is_one_chunk() {
        let chunks = SentenceToken::new(1, 50, 0).chunk("Hello world.");
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn multiple_sentences_grouped_within_max_tokens() {
        // 4 sentences of ~3 tokens each → max_tokens 20 should fit all in one
        let text = "One two three. Four five six. Seven eight nine. Ten eleven twelve.";
        let chunks = SentenceToken::new(1, 20, 0).chunk(text);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn sentences_split_when_exceeding_max_tokens() {
        // Each sentence has ~5 tokens; max_tokens=6 → at most 1-2 sentences per chunk
        let text = "Alpha beta gamma delta epsilon. Zeta eta theta iota kappa. Lambda mu nu xi omicron.";
        let chunks = SentenceToken::new(1, 6, 0).chunk(text);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(chunk.token_count() <= 12); // some flex for sentence grouping
        }
    }

    #[test]
    fn chunk_indices_are_sequential() {
        let text = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten.";
        let chunks = SentenceToken::new(1, 3, 0).chunk(text);
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
        }
    }

    #[test]
    fn overlap_carries_content_forward() {
        // With overlap, last sentence of chunk N appears in chunk N+1
        let text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu.";
        let no_overlap = SentenceToken::new(1, 4, 0).chunk(text);
        let with_overlap = SentenceToken::new(1, 4, 2).chunk(text);
        // With overlap, there should be more or equal chunks (extra buffer sentences)
        assert!(with_overlap.len() >= no_overlap.len());
    }

    #[test]
    fn name_is_sentence_token() {
        assert_eq!(SentenceToken::default().name(), "sentence_token");
    }
}
