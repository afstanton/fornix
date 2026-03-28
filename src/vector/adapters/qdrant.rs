//! Qdrant-backed vector adapter.
//!
//! Uses Qdrant collections for namespace isolation (one collection per
//! namespace, or a single collection with payload-based filtering).
//!
//! # Status
//!
//! Stubbed. Implementation will be added once a Rust Qdrant client is
//! selected and wired into the workspace.
//!
//! Integration tests for this adapter require a running Qdrant instance
//! and are located in `tests/vector_qdrant_integration.rs`, which is
//! excluded from the default `cargo test` run.
