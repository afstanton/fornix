//! Fornix — knowledge storage, retrieval, and graph infrastructure
//! for cognitive systems.
//!
//! Each module is gated behind a Cargo feature of the same name.
//! Enable only what you need:
//!
//! ```toml
//! [dependencies]
//! fornix = { version = "0.1", features = ["vector", "graph", "rag"] }
//! ```

#[cfg(feature = "store")]
pub mod store;

#[cfg(feature = "cache")]
pub mod cache;

#[cfg(feature = "vector")]
pub mod vector;

#[cfg(feature = "bm25")]
pub mod bm25;

#[cfg(feature = "hybrid")]
pub mod hybrid;

#[cfg(feature = "graph")]
pub mod graph;

#[cfg(feature = "graphrag")]
pub mod graphrag;

#[cfg(feature = "rag")]
pub mod rag;

#[cfg(feature = "router")]
pub mod router;

#[cfg(feature = "diff")]
pub mod diff;

#[cfg(feature = "tuner")]
pub mod tuner;

#[cfg(feature = "agent")]
pub mod agent;
