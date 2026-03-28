//! GraphRAG schema constants — property key strings used across all
//! GraphRAG operations. Mirrors `Cortex::GraphRAG::Schema`.

// --- Source provenance ---
pub const SOURCE_HASH:     &str = "source_hash";
pub const SOURCE_ID:       &str = "source_id";
pub const SOURCE_IDS:      &str = "source_ids";
pub const CLAIM_SOURCES:   &str = "claim_sources";
pub const SOURCE_TYPE:     &str = "source_type";

// --- Ingestion lifecycle ---
pub const INGESTION_PASS:  &str = "ingestion_pass";
pub const INGESTION_STAGE: &str = "ingestion_stage";
pub const INGESTED_AT:     &str = "ingested_at";

// --- Entity statistics ---
pub const FREQUENCY:       &str = "frequency";
pub const DESCRIPTIONS:    &str = "descriptions";
pub const MENTION_COUNT:   &str = "mention_count";
pub const SOURCE_COUNT:    &str = "source_count";

// --- Extraction metadata ---
pub const EXTRACTION_ATTRIBUTES:  &str = "attributes";
pub const EXTRACTION_CONFIDENCE:  &str = "confidence";
pub const EXTRACTION_KEYWORDS:    &str = "keywords";

// --- Relation metadata ---
pub const RELATION_WEIGHT: &str = "weight";
pub const RELATION_EMBEDDING: &str = "embedding";
pub const RELATION_DESCRIPTION_EMBEDDING: &str = "description_embedding";

// --- Coverage scoring ---
pub const COVERAGE_SCORE:      &str = "coverage_score";
pub const COVERAGE_SCORED_AT:  &str = "coverage_scored_at";
pub const COVERAGE_DIMENSIONS: &str = "coverage_dimensions";

// --- Community detection ---
pub const SUMMARY:               &str = "summary";
pub const COMMUNITY_ID:          &str = "community_id";
pub const COMMUNITY_KEY:         &str = "community_key";
pub const COMMUNITY_LEVEL:       &str = "community_level";
pub const COMMUNITY_PARENT_ID:   &str = "community_parent_id";
pub const COMMUNITY_CHILD_IDS:   &str = "community_child_ids";
pub const COMMUNITY_ENTITIES:    &str = "community_entities";
pub const LAST_GENERATED_AT:     &str = "last_generated_at";
pub const ENTITY_COUNT:          &str = "entity_count";
pub const RELATION_COUNT:        &str = "relation_count";

// --- Inference traces (Plan 33) ---
pub const INFERENCE_TRACE_TYPE:   &str = "INFERENCE_TRACE";
pub const INFERRED_FROM:          &str = "INFERRED_FROM";
pub const SUPPORTED_BY:           &str = "SUPPORTED_BY";
pub const CONCLUSION_TEXT:        &str = "conclusion";
pub const TRACE_CHAIN:            &str = "trace_chain";
pub const TRACE_EXPIRES_AT:       &str = "trace_expires_at";
pub const TRACE_QUERY:            &str = "trace_query";
pub const TRACE_SESSION_ID:       &str = "trace_session_id";
pub const TRACE_HASH:             &str = "trace_hash";
pub const TRACE_CONFIDENCE:       &str = "trace_confidence";
pub const TRACE_ASSERTION_STATE:  &str = "trace_assertion_state";
pub const TRACE_DERIVATION_TYPE:  &str = "trace_derivation_type";
pub const TRACE_ENTITY_IDS:       &str = "trace_entity_ids";
pub const TRACE_RELATION_IDS:     &str = "trace_relation_ids";
pub const TRACE_SOURCE_IDS:       &str = "trace_source_ids";
pub const TRACE_CREATED_AT:       &str = "trace_created_at";

/// Default entity types recognised by the extraction pipeline.
pub const DEFAULT_ENTITY_TYPES: &[&str] = &[
    "Person", "Organization", "Location", "Event",
    "Concept", "Product", "Technology", "Document",
];

/// Default relation types.
pub const DEFAULT_RELATION_TYPES: &[&str] = &[
    "WORKS_FOR", "LOCATED_IN", "CREATED", "RELATED_TO",
    "PART_OF", "INFLUENCED", "PRECEDED", "FOLLOWED",
    "COLLABORATED_WITH", "MENTIONED_IN",
];
