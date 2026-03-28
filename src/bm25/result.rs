//! BM25 search result type.

/// A single BM25 search result.
#[derive(Debug, Clone)]
pub struct Bm25Result {
    /// The document identifier.
    pub id: String,
    /// The BM25 score. Higher is more relevant. Always >= 0.0.
    pub score: f32,
    /// The query tokens that were found in this document.
    pub matched_terms: Vec<String>,
    /// Per-field score breakdown (field name → score contribution).
    pub field_scores: std::collections::HashMap<String, f32>,
}

impl Bm25Result {
    /// Construct a result with a score and matched terms.
    pub fn new(
        id: impl Into<String>,
        score: f32,
        matched_terms: Vec<String>,
        field_scores: std::collections::HashMap<String, f32>,
    ) -> Self {
        Self {
            id: id.into(),
            score: score.max(0.0),
            matched_terms,
            field_scores,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn score_is_clamped_to_zero() {
        let r = Bm25Result::new("id", -1.0, vec![], HashMap::new());
        assert_eq!(r.score, 0.0);
    }

    #[test]
    fn positive_score_is_preserved() {
        let r = Bm25Result::new("id", std::f32::consts::PI, vec![], HashMap::new());
        assert!((r.score - std::f32::consts::PI).abs() < 1e-5);
    }

    #[test]
    fn id_is_stored() {
        let r = Bm25Result::new("doc-42", 1.0, vec![], HashMap::new());
        assert_eq!(r.id, "doc-42");
    }

    #[test]
    fn matched_terms_are_stored() {
        let terms = vec!["rust".to_string(), "programming".to_string()];
        let r = Bm25Result::new("id", 1.0, terms.clone(), HashMap::new());
        assert_eq!(r.matched_terms, terms);
    }

    #[test]
    fn field_scores_are_stored() {
        let mut fs = HashMap::new();
        fs.insert("title".to_string(), 2.5_f32);
        fs.insert("body".to_string(), 0.8_f32);
        let r = Bm25Result::new("id", 3.3, vec![], fs.clone());
        assert_eq!(r.field_scores["title"], 2.5);
        assert_eq!(r.field_scores["body"], 0.8);
    }
}
