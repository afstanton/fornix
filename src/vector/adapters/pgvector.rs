//! pgvector-backed vector adapter.
//!
//! Stores vectors in a Postgres table using the `pgvector` extension for
//! ANN search. Requires `sqlx` with the `postgres` feature and the
//! `pgvector` extension installed in the database.
//!
//! # Status
//!
//! Stubbed. Implementation will be added once `sqlx` is wired into the
//! workspace.
//!
//! # Expected schema
//!
//! ```sql
//! CREATE EXTENSION IF NOT EXISTS vector;
//!
//! CREATE TABLE fornix_vectors (
//!   id         TEXT        NOT NULL,
//!   namespace  TEXT        NOT NULL,
//!   vector     vector(N)   NOT NULL,   -- N = your embedding dimension
//!   metadata   JSONB       NOT NULL DEFAULT '{}',
//!   PRIMARY KEY (id, namespace)
//! );
//!
//! -- ANN index (choose one):
//! CREATE INDEX ON fornix_vectors USING ivfflat (vector vector_cosine_ops);
//! -- or: CREATE INDEX ON fornix_vectors USING hnsw (vector vector_cosine_ops);
//! ```
//!
//! Integration tests for this adapter require a live Postgres instance with
//! pgvector installed and are located in `tests/vector_pgvector_integration.rs`,
//! which is excluded from the default `cargo test` run.
