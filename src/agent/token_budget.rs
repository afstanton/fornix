//! Thread-safe token budget tracking.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Tracks cumulative token usage across a solve session.
/// Thread-safe via `AtomicUsize`.
#[derive(Default)]
pub struct TokenBudget {
    used: AtomicUsize,
}

impl TokenBudget {
    pub fn new() -> Self { Self::default() }

    /// Add `count` tokens to the running total (negative values are ignored).
    pub fn add(&self, count: usize) {
        self.used.fetch_add(count, Ordering::Relaxed);
    }

    /// Current token usage.
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    /// Tokens remaining before `maximum` is hit, or `None` if uncapped.
    pub fn remaining(&self, maximum: Option<usize>) -> Option<usize> {
        Some(maximum?.saturating_sub(self.used()))
    }

    /// `true` when `used > maximum`.
    pub fn exceeded(&self, maximum: Option<usize>) -> bool {
        match maximum {
            Some(max) => self.used() > max,
            None => false,
        }
    }

    /// Reset the counter to zero.
    pub fn reset(&self) {
        self.used.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_at_zero() {
        assert_eq!(TokenBudget::new().used(), 0);
    }

    #[test]
    fn add_accumulates() {
        let b = TokenBudget::new();
        b.add(100);
        b.add(200);
        assert_eq!(b.used(), 300);
    }

    #[test]
    fn not_exceeded_when_under_limit() {
        let b = TokenBudget::new();
        b.add(100);
        assert!(!b.exceeded(Some(200)));
    }

    #[test]
    fn exceeded_when_over_limit() {
        let b = TokenBudget::new();
        b.add(300);
        assert!(b.exceeded(Some(200)));
    }

    #[test]
    fn never_exceeded_with_no_limit() {
        let b = TokenBudget::new();
        b.add(usize::MAX / 2);
        assert!(!b.exceeded(None));
    }

    #[test]
    fn remaining_with_limit() {
        let b = TokenBudget::new();
        b.add(100);
        assert_eq!(b.remaining(Some(400)), Some(300));
    }

    #[test]
    fn remaining_no_limit_is_none() {
        assert!(TokenBudget::new().remaining(None).is_none());
    }

    #[test]
    fn remaining_saturates_at_zero() {
        let b = TokenBudget::new();
        b.add(500);
        assert_eq!(b.remaining(Some(100)), Some(0));
    }

    #[test]
    fn reset_clears_count() {
        let b = TokenBudget::new();
        b.add(1000);
        b.reset();
        assert_eq!(b.used(), 0);
    }
}
