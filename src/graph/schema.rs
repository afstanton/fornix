//! Graph schema constants: causal types, infrastructure types, and property keys.

/// Property key for external reference objects stored in entity properties.
pub const EXTERNAL_REFS: &str = "external_refs";

/// Property key for claim source objects stored in entity/relation properties.
pub const CLAIM_SOURCES: &str = "claim_sources";

/// Property key for causal strength (f32 in [0.0, 1.0]).
pub const CAUSAL_STRENGTH: &str = "causal_strength";

/// Property key for causal direction override.
pub const CAUSAL_DIRECTION: &str = "causal_direction";

/// Property key for the is_causal boolean flag.
pub const IS_CAUSAL: &str = "is_causal";

/// Known external identifier sources.
pub mod external_sources {
    pub const WIKIDATA:  &str = "wikidata";
    pub const DBPEDIA:   &str = "dbpedia";
    pub const LEI:       &str = "lei";
    pub const CUSIP:     &str = "cusip";
    pub const ISIN:      &str = "isin";
    pub const SEDOL:     &str = "sedol";
    pub const CIK:       &str = "cik";
    pub const EUR_LEX:   &str = "eur_lex";
    pub const GRID:      &str = "grid";
    pub const ROR:       &str = "ror";
    pub const VIAF:      &str = "viaf";
    pub const FREEBASE:  &str = "freebase";
}

/// Causal relation type string constants.
pub mod causal_types {
    pub const CAUSES:          &str = "CAUSES";
    pub const ENABLES:         &str = "ENABLES";
    pub const PREVENTS:        &str = "PREVENTS";
    pub const INHIBITS:        &str = "INHIBITS";
    pub const TRIGGERS:        &str = "TRIGGERS";
    pub const REQUIRES:        &str = "REQUIRES";
    pub const SUPERSEDES:      &str = "SUPERSEDES";
    pub const AMENDS:          &str = "AMENDS";
    pub const QUALIFIES:       &str = "QUALIFIES";
    pub const CONTRADICTS:     &str = "CONTRADICTS";
    pub const REPEALS:         &str = "REPEALS";
    pub const EXEMPTS:         &str = "EXEMPTS";
    pub const CORRELATED_WITH: &str = "CORRELATED_WITH";
}

/// All causal types as a static slice for membership checks.
pub const ALL_CAUSAL_TYPES: &[&str] = &[
    causal_types::CAUSES,
    causal_types::ENABLES,
    causal_types::PREVENTS,
    causal_types::INHIBITS,
    causal_types::TRIGGERS,
    causal_types::REQUIRES,
    causal_types::SUPERSEDES,
    causal_types::AMENDS,
    causal_types::QUALIFIES,
    causal_types::CONTRADICTS,
    causal_types::REPEALS,
    causal_types::EXEMPTS,
    causal_types::CORRELATED_WITH,
];

/// Mechanistic causal types (direct physical/biological causation).
pub const MECHANISTIC_CAUSAL_TYPES: &[&str] = &[
    causal_types::CAUSES,
    causal_types::PREVENTS,
    causal_types::INHIBITS,
    causal_types::TRIGGERS,
];

/// Dependency causal types.
pub const DEPENDENCY_CAUSAL_TYPES: &[&str] = &[
    causal_types::ENABLES,
    causal_types::REQUIRES,
];

/// Regulatory causal types.
pub const REGULATORY_CAUSAL_TYPES: &[&str] = &[
    causal_types::SUPERSEDES,
    causal_types::AMENDS,
    causal_types::QUALIFIES,
    causal_types::CONTRADICTS,
    causal_types::REPEALS,
    causal_types::EXEMPTS,
];

/// Infrastructure relation types — internal use only, never extracted from content.
pub mod infrastructure_types {
    /// Reserved for graph-traversable version-chain representation.
    pub const VERSION_SUPERSEDES: &str = "__version_supersedes__";
}

/// All infrastructure relation types.
pub const ALL_INFRASTRUCTURE_TYPES: &[&str] = &[
    infrastructure_types::VERSION_SUPERSEDES,
];

/// Returns `true` if `relation_type` is a causal type.
pub fn is_causal(relation_type: &str) -> bool {
    ALL_CAUSAL_TYPES.contains(&relation_type)
}

/// Returns `true` if `relation_type` is an infrastructure type.
pub fn is_infrastructure(relation_type: &str) -> bool {
    ALL_INFRASTRUCTURE_TYPES.contains(&relation_type)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causes_is_causal() {
        assert!(is_causal(causal_types::CAUSES));
    }

    #[test]
    fn correlated_with_is_causal() {
        assert!(is_causal(causal_types::CORRELATED_WITH));
    }

    #[test]
    fn arbitrary_type_is_not_causal() {
        assert!(!is_causal("RELATED_TO"));
    }

    #[test]
    fn version_supersedes_is_infrastructure() {
        assert!(is_infrastructure(infrastructure_types::VERSION_SUPERSEDES));
    }

    #[test]
    fn causes_is_not_infrastructure() {
        assert!(!is_infrastructure(causal_types::CAUSES));
    }

    #[test]
    fn mechanistic_types_are_subset_of_all_causal() {
        for t in MECHANISTIC_CAUSAL_TYPES {
            assert!(is_causal(t), "{} should be causal", t);
        }
    }

    #[test]
    fn regulatory_types_are_subset_of_all_causal() {
        for t in REGULATORY_CAUSAL_TYPES {
            assert!(is_causal(t), "{} should be causal", t);
        }
    }

    #[test]
    fn all_causal_types_count() {
        assert_eq!(ALL_CAUSAL_TYPES.len(), 13);
    }
}
