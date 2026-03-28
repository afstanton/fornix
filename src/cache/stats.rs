//! Cache statistics for observability.

/// Snapshot of cache activity for a given namespace (or the whole adapter).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of successful cache reads.
    pub hits: u64,
    /// Number of cache reads that found no entry (including expired entries).
    pub misses: u64,
    /// Number of entries removed because their TTL expired.
    pub evictions: u64,
    /// Current number of live entries in scope.
    pub size: usize,
}

impl CacheStats {
    /// Returns the total number of read attempts (`hits + misses`).
    pub fn total_reads(&self) -> u64 {
        self.hits + self.misses
    }

    /// Returns the hit rate as a value in `[0.0, 1.0]`.
    /// Returns `0.0` if no reads have been recorded.
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_reads();
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Merge another stats snapshot into this one (used for cross-namespace aggregation).
    pub fn merge(&mut self, other: &CacheStats) {
        self.hits += other.hits;
        self.misses += other.misses;
        self.evictions += other.evictions;
        self.size += other.size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_all_zeros() {
        let s = CacheStats::default();
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
        assert_eq!(s.evictions, 0);
        assert_eq!(s.size, 0);
    }

    #[test]
    fn total_reads_is_hits_plus_misses() {
        let s = CacheStats { hits: 7, misses: 3, ..Default::default() };
        assert_eq!(s.total_reads(), 10);
    }

    #[test]
    fn hit_rate_zero_when_no_reads() {
        let s = CacheStats::default();
        assert_eq!(s.hit_rate(), 0.0);
    }

    #[test]
    fn hit_rate_one_when_all_hits() {
        let s = CacheStats { hits: 5, misses: 0, ..Default::default() };
        assert_eq!(s.hit_rate(), 1.0);
    }

    #[test]
    fn hit_rate_zero_when_all_misses() {
        let s = CacheStats { hits: 0, misses: 4, ..Default::default() };
        assert_eq!(s.hit_rate(), 0.0);
    }

    #[test]
    fn hit_rate_partial() {
        let s = CacheStats { hits: 3, misses: 1, ..Default::default() };
        assert!((s.hit_rate() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn merge_accumulates_all_fields() {
        let mut a = CacheStats { hits: 5, misses: 2, evictions: 1, size: 10 };
        let b = CacheStats { hits: 3, misses: 1, evictions: 2, size: 5 };
        a.merge(&b);
        assert_eq!(a.hits, 8);
        assert_eq!(a.misses, 3);
        assert_eq!(a.evictions, 3);
        assert_eq!(a.size, 15);
    }

    #[test]
    fn merge_with_default_is_identity() {
        let original = CacheStats { hits: 4, misses: 1, evictions: 0, size: 3 };
        let mut s = original.clone();
        s.merge(&CacheStats::default());
        assert_eq!(s, original);
    }
}
