//! Parent-child chunker.
//!
//! First splits text into large "parent" chunks (for context), then splits
//! each parent into smaller "child" chunks (for precise retrieval). Both
//! parents and children are emitted in the output, interleaved: each parent
//! is immediately followed by its children.
//!
//! Children carry `parent_id` pointing to their parent's `index`.

use serde_json::json;

use crate::rag::{
    chunkers::{token_count::TokenCount, Chunker},
    types::Chunk,
};

/// Hierarchical parent/child chunker.
#[derive(Debug, Clone)]
pub struct ParentChild {
    pub parent_size: usize,
    pub child_size: usize,
}

impl ParentChild {
    pub fn new(parent_size: usize, child_size: usize) -> Self {
        assert!(parent_size > 0, "parent_size must be > 0");
        assert!(child_size > 0, "child_size must be > 0");
        assert!(child_size <= parent_size, "child_size must be <= parent_size");
        Self { parent_size, child_size }
    }
}

impl Default for ParentChild {
    fn default() -> Self {
        Self::new(1024, 256)
    }
}

impl Chunker for ParentChild {
    fn name(&self) -> &'static str {
        "parent_child"
    }

    fn chunk(&self, text: &str) -> Vec<Chunk> {
        if text.trim().is_empty() {
            return Vec::new();
        }

        let parents = TokenCount::new(self.parent_size, 0).chunk(text);
        let mut output: Vec<Chunk> = Vec::new();
        let mut global_index = 0;

        for parent in parents {
            let parent_index = global_index;

            let mut parent_chunk = Chunk::new(
                &parent.content,
                parent_index,
                parent.start_offset,
                parent.end_offset,
            );
            parent_chunk.metadata.insert("chunk_type".to_string(), json!("parent"));
            global_index += 1;

            let children = TokenCount::new(self.child_size, 0).chunk(&parent.content);
            let child_chunks: Vec<Chunk> = children
                .into_iter()
                .map(|child| {
                    let abs_start = parent.start_offset + child.start_offset;
                    let abs_end = parent.start_offset + child.end_offset;
                    let mut c = Chunk::new(&child.content, global_index, abs_start, abs_end);
                    c.metadata.insert("chunk_type".to_string(), json!("child"));
                    c.parent_id = Some(parent_index);
                    global_index += 1;
                    c
                })
                .collect();

            output.push(parent_chunk);
            output.extend(child_chunks);
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_returns_no_chunks() {
        assert!(ParentChild::default().chunk("").is_empty());
    }

    #[test]
    fn whitespace_only_returns_no_chunks() {
        assert!(ParentChild::new(10, 5).chunk("   ").is_empty());
    }

    #[test]
    fn short_text_produces_parent_and_children() {
        // 10 tokens: parent_size=10 (1 parent), child_size=5 (2 children)
        let text = "one two three four five six seven eight nine ten";
        let chunks = ParentChild::new(10, 5).chunk(text);
        // Should be 1 parent + 2 children = 3 chunks
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn parents_precede_their_children() {
        let text = "one two three four five six seven eight nine ten";
        let chunks = ParentChild::new(10, 5).chunk(text);
        // First chunk must be a parent
        assert_eq!(chunks[0].metadata["chunk_type"], serde_json::json!("parent"));
        assert!(chunks[0].parent_id.is_none());
        // Subsequent chunks should be children pointing to index 0
        assert_eq!(chunks[1].metadata["chunk_type"], serde_json::json!("child"));
        assert_eq!(chunks[1].parent_id, Some(0));
    }

    #[test]
    fn child_parent_id_points_to_correct_parent() {
        let text: String = (0..30).map(|i| format!("word{}", i)).collect::<Vec<_>>().join(" ");
        let chunks = ParentChild::new(15, 5).chunk(&text);

        for chunk in &chunks {
            if chunk.metadata["chunk_type"] == serde_json::json!("child") {
                let parent_idx = chunk.parent_id.unwrap();
                let parent = &chunks[parent_idx];
                assert_eq!(parent.metadata["chunk_type"], serde_json::json!("parent"));
            }
        }
    }

    #[test]
    fn child_offsets_are_absolute_within_original_text() {
        let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
        let chunks = ParentChild::new(10, 3).chunk(text);
        for chunk in &chunks {
            let reconstructed = &text[chunk.start_offset..chunk.end_offset];
            // Content should match the corresponding text slice
            assert_eq!(reconstructed.trim(), chunk.content.trim());
        }
    }

    #[test]
    fn chunk_indices_are_sequential() {
        let text: String = (0..20).map(|i| format!("t{}", i)).collect::<Vec<_>>().join(" ");
        let chunks = ParentChild::new(10, 5).chunk(&text);
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
        }
    }

    #[test]
    fn name_is_parent_child() {
        assert_eq!(ParentChild::default().name(), "parent_child");
    }
}
