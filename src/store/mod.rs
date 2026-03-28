//! Base adapter infrastructure and shared storage foundations.
//!
//! This module provides the building blocks that all other fornix storage
//! modules compose on top of:
//!
//! - [`adapter::StorageAdapter`] — the base async trait for all backends
//! - [`adapter::AdapterFactory`] — factory trait for constructing adapters
//! - [`config::AdapterConfig`] — validation trait for adapter configuration
//! - [`config::ConnectionConfig`] — general URL-based connection configuration
//! - [`health::HealthStatus`] — typed health reporting with degraded/unhealthy states
//! - [`health::HealthReport`] — timed health check result
//! - [`error::Error`] — typed error enum covering all failure modes

pub mod adapter;
pub mod config;
pub mod error;
pub mod health;

// Flatten the most commonly needed items to the module root for convenience.
pub use adapter::{AdapterFactory, StorageAdapter};
pub use config::{AdapterConfig, ConnectionConfig};
pub use error::{Error, Result};
pub use health::{HealthReport, HealthStatus};
