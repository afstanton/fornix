//! Postgres-backed cache adapter.
//!
//! Stores cache entries in a Postgres table with namespace, key, value,
//! expiry, and hit-count columns. Requires the `sqlx` crate with the
//! `postgres` feature, which will be added when this adapter is implemented.
//!
//! # Status
//!
//! Stubbed. The trait skeleton is in place; the SQL implementation will be
//! added once `sqlx` is wired into the workspace.

// TODO: implement using sqlx when added as a dependency.
// The table schema expected by this adapter:
//
//   CREATE TABLE fornix_cache (
//     namespace   TEXT        NOT NULL,
//     key         TEXT        NOT NULL,
//     value       BYTEA       NOT NULL,
//     expires_at  TIMESTAMPTZ,
//     hits        BIGINT      NOT NULL DEFAULT 0,
//     PRIMARY KEY (namespace, key)
//   );
//
// Tests for this adapter are integration tests requiring a live Postgres
// instance and are located in tests/cache_postgres_integration.rs,
// which is excluded from the default `cargo test` run.
