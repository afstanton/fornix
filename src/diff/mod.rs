//! Text diffing and focused snippet extraction.
//!
//! Pure logic, no adapter pattern needed.

/// A pair of focused text snippets centred on the changed region.
pub struct SnippetPair {
    pub previous: String,
    pub current: String,
}

/// Interface for computing a focused snippet pair from two full text versions.
pub trait TextDiffer: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn focused_snippet_pair(
        &self,
        previous: &str,
        current: &str,
        snippet_max: usize,
        context_tokens: usize,
    ) -> Result<SnippetPair, Self::Error>;
}
