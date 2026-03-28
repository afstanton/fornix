//! The base `StorageAdapter` trait implemented by all fornix backends.

use async_trait::async_trait;
use crate::store::{
    config::AdapterConfig,
    error::Result,
    health::HealthReport,
};

/// The base trait for all fornix storage adapters.
///
/// Every backend — whether for vectors, graphs, cache, or BM25 — implements
/// this trait to provide a consistent connection lifecycle and health
/// reporting interface.
///
/// The adapter owns its connection pool or connection handle. Callers obtain
/// an adapter instance (typically via a builder or factory), call `connect()`,
/// and then use the more specific trait for that backend
/// (e.g. [`crate::vector::VectorAdapter`]).
#[async_trait]
pub trait StorageAdapter: Send + Sync {
    /// The configuration type for this adapter.
    type Config: AdapterConfig;

    /// The human-readable name of this adapter (e.g. `"pgvector"`, `"qdrant"`).
    fn name(&self) -> &'static str;

    /// Establish a connection to the backend.
    ///
    /// Must be called before any operations. Calling `connect()` on an already
    /// connected adapter should be a no-op or re-validate the connection.
    async fn connect(&mut self) -> Result<()>;

    /// Gracefully close the connection and release resources.
    ///
    /// After calling `disconnect()`, the adapter may be reconnected by calling
    /// `connect()` again.
    async fn disconnect(&mut self) -> Result<()>;

    /// Returns `true` if the adapter currently has an active connection.
    fn is_connected(&self) -> bool;

    /// Check the health of the backend connection.
    ///
    /// Returns a [`HealthReport`] including status, latency, and timestamp.
    /// This should perform a lightweight round-trip to the backend rather
    /// than a deep diagnostic.
    async fn healthcheck(&self) -> HealthReport;
}

/// A factory for constructing and connecting a [`StorageAdapter`].
///
/// This is the entry point for obtaining a ready-to-use adapter instance.
/// Implementations validate configuration and establish the initial connection.
#[async_trait]
pub trait AdapterFactory: Send + Sync {
    /// The type of adapter this factory produces.
    type Adapter: StorageAdapter;
    /// The configuration type accepted by this factory.
    type Config: AdapterConfig;

    /// Build and connect an adapter from the given configuration.
    ///
    /// Validates the configuration, constructs the adapter, and calls
    /// `connect()` before returning. Returns an error if any step fails.
    async fn build(&self, config: Self::Config) -> Result<Self::Adapter>;
}

#[cfg(test)]
pub(crate) mod mock {
    //! In-process mock adapter used by unit tests throughout the store module.
    //!
    //! Not compiled in production builds.

    use super::*;
    use crate::store::{
        config::{AdapterConfig, ConnectionConfig},
        error::{Error, Result},
        health::{HealthReport, HealthStatus},
    };

    /// A trivial in-memory adapter that tracks connection state.
    #[derive(Debug)]
    pub struct MockAdapter {
        pub _config: ConnectionConfig,
        connected: bool,
        /// When `true`, `connect()` returns an error.
        pub fail_connect: bool,
        /// When `true`, `healthcheck()` returns Unhealthy.
        pub fail_health: bool,
    }

    impl MockAdapter {
        pub fn new(url: &str) -> Self {
            Self {
                _config: ConnectionConfig::new(url),
                connected: false,
                fail_connect: false,
                fail_health: false,
            }
        }
    }

    #[async_trait]
    impl StorageAdapter for MockAdapter {
        type Config = ConnectionConfig;

        fn name(&self) -> &'static str {
            "mock"
        }

        async fn connect(&mut self) -> Result<()> {
            if self.fail_connect {
                return Err(Error::connection("mock: forced connect failure"));
            }
            self.connected = true;
            Ok(())
        }

        async fn disconnect(&mut self) -> Result<()> {
            self.connected = false;
            Ok(())
        }

        fn is_connected(&self) -> bool {
            self.connected
        }

        async fn healthcheck(&self) -> HealthReport {
            let status = if self.fail_health {
                HealthStatus::Unhealthy { reason: "mock: forced failure".to_string() }
            } else if self.connected {
                HealthStatus::Healthy
            } else {
                HealthStatus::Unhealthy { reason: "not connected".to_string() }
            };
            HealthReport::begin("mock").finish(status)
        }
    }

    /// A factory that builds and connects a [`MockAdapter`].
    pub struct MockAdapterFactory;

    #[async_trait]
    impl AdapterFactory for MockAdapterFactory {
        type Adapter = MockAdapter;
        type Config = ConnectionConfig;

        async fn build(&self, config: Self::Config) -> Result<Self::Adapter> {
            config.validate()?;
            let mut adapter = MockAdapter {
                _config: config,
                connected: false,
                fail_connect: false,
                fail_health: false,
            };
            adapter.connect().await?;
            Ok(adapter)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::mock::{MockAdapter, MockAdapterFactory};
    use super::*;
    use crate::store::{
        config::ConnectionConfig,
        error::Error,
        health::HealthStatus,
    };

    // --- Basic lifecycle ---

    #[tokio::test]
    async fn adapter_starts_disconnected() {
        let adapter = MockAdapter::new("mock://localhost");
        assert!(!adapter.is_connected());
    }

    #[tokio::test]
    async fn connect_sets_connected_state() {
        let mut adapter = MockAdapter::new("mock://localhost");
        adapter.connect().await.unwrap();
        assert!(adapter.is_connected());
    }

    #[tokio::test]
    async fn disconnect_clears_connected_state() {
        let mut adapter = MockAdapter::new("mock://localhost");
        adapter.connect().await.unwrap();
        adapter.disconnect().await.unwrap();
        assert!(!adapter.is_connected());
    }

    #[tokio::test]
    async fn reconnect_after_disconnect() {
        let mut adapter = MockAdapter::new("mock://localhost");
        adapter.connect().await.unwrap();
        adapter.disconnect().await.unwrap();
        adapter.connect().await.unwrap();
        assert!(adapter.is_connected());
    }

    // --- Name ---

    #[tokio::test]
    async fn adapter_name_is_mock() {
        let adapter = MockAdapter::new("mock://localhost");
        assert_eq!(adapter.name(), "mock");
    }

    // --- Forced connect failure ---

    #[tokio::test]
    async fn connect_failure_returns_error() {
        let mut adapter = MockAdapter::new("mock://localhost");
        adapter.fail_connect = true;
        let err = adapter.connect().await.unwrap_err();
        assert!(matches!(err, Error::Connection(_)));
        assert!(!adapter.is_connected());
    }

    // --- Healthcheck ---

    #[tokio::test]
    async fn healthcheck_healthy_when_connected() {
        let mut adapter = MockAdapter::new("mock://localhost");
        adapter.connect().await.unwrap();
        let report = adapter.healthcheck().await;
        assert_eq!(report.status, HealthStatus::Healthy);
        assert_eq!(report.adapter, "mock");
    }

    #[tokio::test]
    async fn healthcheck_unhealthy_when_not_connected() {
        let adapter = MockAdapter::new("mock://localhost");
        let report = adapter.healthcheck().await;
        assert!(!report.status.is_usable());
        assert!(report.status.reason().is_some());
    }

    #[tokio::test]
    async fn healthcheck_unhealthy_when_forced() {
        let mut adapter = MockAdapter::new("mock://localhost");
        adapter.connect().await.unwrap();
        adapter.fail_health = true;
        let report = adapter.healthcheck().await;
        assert!(!report.status.is_usable());
    }

    #[tokio::test]
    async fn healthcheck_report_has_latency() {
        let mut adapter = MockAdapter::new("mock://localhost");
        adapter.connect().await.unwrap();
        let report = adapter.healthcheck().await;
        // Latency should be set (even if near-zero for an in-memory mock)
        let _ = report.latency; // just verify it's accessible and doesn't panic
    }

    // --- Factory ---

    #[tokio::test]
    async fn factory_builds_and_connects_adapter() {
        let factory = MockAdapterFactory;
        let config = ConnectionConfig::new("mock://localhost");
        let adapter = factory.build(config).await.unwrap();
        assert!(adapter.is_connected());
        assert_eq!(adapter.name(), "mock");
    }

    #[tokio::test]
    async fn factory_rejects_invalid_config() {
        let factory = MockAdapterFactory;
        let config = ConnectionConfig::new(""); // empty URL fails validation
        let err = factory.build(config).await.unwrap_err();
        assert!(matches!(err, Error::Configuration(_)));
    }
}
