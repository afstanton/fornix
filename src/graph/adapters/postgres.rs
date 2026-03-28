//! Postgres-backed graph adapter.
//!
//! Stores entities and relations in Postgres with full temporal versioning
//! (valid-time and system-time bitemporality) and a changelog table.
//!
//! # Status
//!
//! Stubbed. Implementation will be added once `sqlx` is wired in.
//!
//! # Expected schema
//!
//! See the `cortex-graph` Ruby gem for the reference Postgres schema,
//! including the `cortex_entities`, `cortex_relations`, and
//! `cortex_graph_changelog` tables with their bitemporality columns.
//!
//! Integration tests requiring a live Postgres instance are located in
//! `tests/graph_postgres_integration.rs`, excluded from the default
//! `cargo test` run.
