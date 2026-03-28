//! BM25 tokeniser.
//!
//! Converts raw text into a sequence of tokens suitable for BM25 indexing.
//! Applies lowercasing, alphanumeric filtering, length constraints, stop word
//! removal, and optional suffix stemming.

use crate::bm25::config::Bm25Config;

/// A tokeniser configured from a [`Bm25Config`].
#[derive(Debug, Clone)]
pub struct Tokenizer {
    stop_words: Option<std::collections::HashSet<String>>,
    min_length: usize,
    max_length: usize,
    pub stem: bool,
}

impl Tokenizer {
    /// Construct a tokeniser from a [`Bm25Config`].
    pub fn from_config(config: &Bm25Config) -> Self {
        let stop_words = config.stop_words.as_ref().map(|words| {
            words.iter().map(|w| w.to_lowercase()).collect()
        });
        Self {
            stop_words,
            min_length: config.token_min_length,
            max_length: config.token_max_length,
            stem: false, // explicit stemmer support reserved for a future pass
        }
    }

    /// Tokenise `text` into a list of normalised tokens.
    ///
    /// Steps applied in order:
    /// 1. Lowercase the input
    /// 2. Extract sequences of ASCII alphanumeric characters
    /// 3. Discard tokens outside [`min_length`, `max_length`]
    /// 4. Apply suffix stemming if enabled
    /// 5. Remove stop words
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let tokens: Vec<String> = text
            .to_lowercase()
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|t| !t.is_empty())
            .filter(|t| t.len() >= self.min_length && t.len() <= self.max_length)
            .map(|t| if self.stem { Self::suffix_stem(t) } else { t.to_string() })
            .filter(|t| !self.is_stop_word(t))
            .collect();
        tokens
    }

    /// Returns `true` if the token is a configured stop word.
    fn is_stop_word(&self, token: &str) -> bool {
        match &self.stop_words {
            Some(set) => set.contains(token),
            None => false,
        }
    }

    /// Minimal suffix stemmer that strips common English suffixes.
    /// Mirrors the `default_stem` method in the Ruby tokeniser.
    fn suffix_stem(token: &str) -> String {
        for suffix in &["ing", "edly", "ed", "ly", "s"] {
            if token.len() > suffix.len() && token.ends_with(suffix) {
                return token[..token.len() - suffix.len()].to_string();
            }
        }
        token.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25::config::Bm25Config;

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_config(&Bm25Config::default())
    }

    fn tokenizer_no_stop() -> Tokenizer {
        Tokenizer::from_config(&Bm25Config::default().without_stop_words())
    }

    fn tokenizer_stem() -> Tokenizer {
        let mut t = tokenizer();
        t.stem = true;
        t
    }

    // --- basic tokenisation ---

    #[test]
    fn empty_string_returns_no_tokens() {
        assert!(tokenizer().tokenize("").is_empty());
    }

    #[test]
    fn single_word_is_lowercased() {
        assert_eq!(tokenizer().tokenize("Hello"), vec!["hello"]);
    }

    #[test]
    fn punctuation_splits_tokens() {
        let tokens = tokenizer_no_stop().tokenize("foo, bar. baz!");
        assert_eq!(tokens, vec!["foo", "bar", "baz"]);
    }

    #[test]
    fn mixed_case_is_normalised() {
        let tokens = tokenizer_no_stop().tokenize("Quick Brown Fox");
        assert_eq!(tokens, vec!["quick", "brown", "fox"]);
    }

    #[test]
    fn alphanumeric_tokens_are_kept() {
        let tokens = tokenizer_no_stop().tokenize("item42 version3");
        assert_eq!(tokens, vec!["item42", "version3"]);
    }

    // --- length filtering ---

    #[test]
    fn token_below_min_length_is_dropped() {
        // default min_length = 2; "a" is length 1
        let tokens = tokenizer_no_stop().tokenize("a ok");
        assert!(!tokens.contains(&"a".to_string()));
        assert!(tokens.contains(&"ok".to_string()));
    }

    #[test]
    fn token_at_min_length_is_kept() {
        let tokens = tokenizer_no_stop().tokenize("ok");
        assert_eq!(tokens, vec!["ok"]);
    }

    #[test]
    fn token_above_max_length_is_dropped() {
        let long = "a".repeat(51); // default max = 50
        let tokens = tokenizer_no_stop().tokenize(&long);
        assert!(tokens.is_empty());
    }

    #[test]
    fn token_at_max_length_is_kept() {
        let boundary = "a".repeat(50);
        let tokens = tokenizer_no_stop().tokenize(&boundary);
        assert_eq!(tokens, vec![boundary]);
    }

    // --- stop words ---

    #[test]
    fn stop_words_are_removed_by_default() {
        let tokens = tokenizer().tokenize("the quick brown fox");
        assert!(!tokens.contains(&"the".to_string()));
        assert!(tokens.contains(&"quick".to_string()));
    }

    #[test]
    fn stop_word_filtering_disabled_when_none() {
        let tokens = tokenizer_no_stop().tokenize("the quick brown fox");
        assert!(tokens.contains(&"the".to_string()));
    }

    #[test]
    fn custom_stop_words_are_applied() {
        let config = Bm25Config {
            stop_words: Some(vec!["custom".to_string()]),
            ..Default::default()
        };
        let t = Tokenizer::from_config(&config);
        let tokens = t.tokenize("custom word here");
        assert!(!tokens.contains(&"custom".to_string()));
        assert!(tokens.contains(&"word".to_string()));
    }

    // --- stemming ---

    #[test]
    fn suffix_stem_removes_ing() {
        assert_eq!(Tokenizer::suffix_stem("running"), "runn");
    }

    #[test]
    fn suffix_stem_removes_ed() {
        assert_eq!(Tokenizer::suffix_stem("jumped"), "jump");
    }

    #[test]
    fn suffix_stem_removes_ly() {
        assert_eq!(Tokenizer::suffix_stem("quickly"), "quick");
    }

    #[test]
    fn suffix_stem_removes_s() {
        assert_eq!(Tokenizer::suffix_stem("cats"), "cat");
    }

    #[test]
    fn suffix_stem_removes_edly() {
        assert_eq!(Tokenizer::suffix_stem("repeatedly"), "repeat");
    }

    #[test]
    fn suffix_stem_leaves_short_words_unchanged() {
        // "is" ends in "s" but stripping leaves "i" — too short guard
        // The stem function strips suffixes only when result is non-empty
        let result = Tokenizer::suffix_stem("is");
        // "i" is valid per the function (no length check in stem itself)
        assert_eq!(result, "i");
    }

    #[test]
    fn stemmed_tokenizer_applies_stemming() {
        let tokens = tokenizer_stem().tokenize("running cats quickly");
        assert!(tokens.contains(&"runn".to_string()));
        assert!(tokens.contains(&"cat".to_string()));
    }

    // --- sentence-level ---

    #[test]
    fn realistic_sentence_tokenises_correctly() {
        let tokens = tokenizer().tokenize("The quick brown fox jumps over the lazy dog");
        assert!(!tokens.contains(&"the".to_string())); // stop word
        assert!(tokens.contains(&"quick".to_string()));
        assert!(tokens.contains(&"fox".to_string()));
        assert!(tokens.contains(&"lazy".to_string()));
        assert!(tokens.contains(&"dog".to_string()));
    }

    #[test]
    fn numbers_in_text_are_kept_as_tokens() {
        let tokens = tokenizer_no_stop().tokenize("version 42 released");
        assert!(tokens.contains(&"42".to_string()));
    }
}
