//! Redis-backed cache adapter.
//!
//! Uses Redis `SET`/`GET`/`DEL`/`FLUSHDB` with optional `PX` TTL.
//! Requires the `redis` crate, which will be added when this adapter
//! is implemented.
//!
//! # Status
//!
//! Stubbed. The trait skeleton is in place; the Redis implementation will be
//! added once the `redis` crate is wired into the workspace.

// TODO: implement using the `redis` crate with async support.
// Tests for this adapter are integration tests requiring a live Redis
// instance and are located in tests/cache_redis_integration.rs,
// which is excluded from the default `cargo test` run.
