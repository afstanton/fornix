//! Common types, traits, and utilities shared across all fornix modules.
//!
//! This module is always compiled regardless of which feature flags are enabled.
//! Import the prelude for convenient access to the most commonly needed items:
//!
//! ```rust
//! use fornix::common::prelude::*;
//! ```

pub mod metadata;
pub mod namespace;
pub mod pagination;
pub mod prelude;
pub mod value;
