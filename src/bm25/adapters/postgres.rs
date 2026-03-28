//! Postgres-backed BM25 adapter using database extension support.
//!
//! Supports pg_bm25 / ParadeDB and Postgres full-text search as backends.
//! Requires `sqlx` with the `postgres` feature.
//!
//! # Status
//!
//! Stubbed. Implementation will be added once `sqlx` is wired into the
//! workspace.
//!
//! # Expected schema (synthetic/blind-index approach)
//!
//! ```sql
//! CREATE TABLE fornix_bm25_tokens (
//!   namespace       TEXT    NOT NULL,
//!   doc_id          TEXT    NOT NULL,
//!   field_name      TEXT    NOT NULL,
//!   token_hash      TEXT    NOT NULL,  -- HMAC-SHA256 of token when blind index enabled
//!   position        INT     NOT NULL,
//!   PRIMARY KEY (namespace, doc_id, field_name, position)
//! );
//!
//! CREATE TABLE fornix_bm25_document_stats (
//!   namespace         TEXT    NOT NULL,
//!   doc_id            TEXT    NOT NULL,
//!   field_name        TEXT    NOT NULL,
//!   token_count       INT     NOT NULL,
//!   unique_token_count INT    NOT NULL,
//!   PRIMARY KEY (namespace, doc_id, field_name)
//! );
//!
//! CREATE TABLE fornix_bm25_term_stats (
//!   namespace          TEXT    NOT NULL,
//!   field_name         TEXT    NOT NULL,
//!   token_hash         TEXT    NOT NULL,
//!   document_frequency INT     NOT NULL DEFAULT 0,
//!   PRIMARY KEY (namespace, field_name, token_hash)
//! );
//!
//! CREATE TABLE fornix_bm25_corpus_stats (
//!   namespace          TEXT    NOT NULL,
//!   field_name         TEXT    NOT NULL,
//!   document_count     INT     NOT NULL DEFAULT 0,
//!   avg_document_length FLOAT  NOT NULL DEFAULT 0.0,
//!   PRIMARY KEY (namespace, field_name)
//! );
//! ```
//!
//! Integration tests for this adapter require a live Postgres instance
//! and are located in `tests/bm25_postgres_integration.rs`, excluded from
//! the default `cargo test` run.
