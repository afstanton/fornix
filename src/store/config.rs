//! Configuration trait and base configuration struct for storage adapters.

use crate::store::error::{Error, Result};

/// Implemented by all adapter-specific configuration types.
///
/// This is the Rust equivalent of the `validate!` pattern in the Ruby base
/// adapter — but here validation is typed and must actually return an error
/// rather than always returning true.
pub trait AdapterConfig: Send + Sync {
    /// The canonical name of the adapter this configuration targets.
    ///
    /// Used for logging and error messages. Examples: `"pgvector"`, `"qdrant"`, `"memory"`.
    fn adapter_name(&self) -> &'static str;

    /// Validate the configuration, returning a descriptive error if invalid.
    ///
    /// Called by the adapter before attempting to connect. Implementations
    /// should check that required fields are present, that values are in
    /// expected ranges, and that any mutually exclusive options do not conflict.
    fn validate(&self) -> Result<()>;
}

/// A general-purpose connection configuration shared across all backends
/// that use a URL-style connection string.
///
/// Backend-specific options beyond the connection URL should be provided
/// via a backend-specific config struct that embeds or wraps this one.
#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    /// Connection URL (e.g. `postgres://user:pass@host/db`).
    pub url: String,
    /// Maximum number of connections to keep in the pool.
    /// `None` lets the backend choose a sensible default.
    pub pool_size: Option<u32>,
    /// Connection timeout in seconds.
    pub connect_timeout_secs: u64,
    /// How long to wait for a pool connection to become available, in seconds.
    pub acquire_timeout_secs: u64,
}

impl ConnectionConfig {
    /// Construct a minimal connection config from a URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            pool_size: None,
            connect_timeout_secs: 10,
            acquire_timeout_secs: 30,
        }
    }
}

impl AdapterConfig for ConnectionConfig {
    fn adapter_name(&self) -> &'static str {
        "connection"
    }

    fn validate(&self) -> Result<()> {
        if self.url.is_empty() {
            return Err(Error::config("connection URL must not be empty"));
        }
        if self.connect_timeout_secs == 0 {
            return Err(Error::config("connect_timeout_secs must be greater than zero"));
        }
        if self.acquire_timeout_secs == 0 {
            return Err(Error::config("acquire_timeout_secs must be greater than zero"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> ConnectionConfig {
        ConnectionConfig::new("postgres://localhost/test")
    }

    // --- ConnectionConfig::new defaults ---

    #[test]
    fn new_stores_url() {
        let c = valid_config();
        assert_eq!(c.url, "postgres://localhost/test");
    }

    #[test]
    fn new_has_no_pool_size() {
        let c = valid_config();
        assert!(c.pool_size.is_none());
    }

    #[test]
    fn new_has_default_connect_timeout() {
        let c = valid_config();
        assert_eq!(c.connect_timeout_secs, 10);
    }

    #[test]
    fn new_has_default_acquire_timeout() {
        let c = valid_config();
        assert_eq!(c.acquire_timeout_secs, 30);
    }

    #[test]
    fn adapter_name_is_connection() {
        assert_eq!(valid_config().adapter_name(), "connection");
    }

    // --- Validation passes ---

    #[test]
    fn validate_passes_for_valid_config() {
        assert!(valid_config().validate().is_ok());
    }

    #[test]
    fn validate_passes_with_explicit_pool_size() {
        let mut c = valid_config();
        c.pool_size = Some(10);
        assert!(c.validate().is_ok());
    }

    // --- Validation failures ---

    #[test]
    fn validate_fails_for_empty_url() {
        let c = ConnectionConfig::new("");
        let err = c.validate().unwrap_err();
        assert!(matches!(err, Error::Configuration(_)));
        assert!(err.to_string().contains("URL must not be empty"));
    }

    #[test]
    fn validate_fails_for_zero_connect_timeout() {
        let mut c = valid_config();
        c.connect_timeout_secs = 0;
        let err = c.validate().unwrap_err();
        assert!(matches!(err, Error::Configuration(_)));
        assert!(err.to_string().contains("connect_timeout_secs"));
    }

    #[test]
    fn validate_fails_for_zero_acquire_timeout() {
        let mut c = valid_config();
        c.acquire_timeout_secs = 0;
        let err = c.validate().unwrap_err();
        assert!(matches!(err, Error::Configuration(_)));
        assert!(err.to_string().contains("acquire_timeout_secs"));
    }

    // --- Clone ---

    #[test]
    fn clone_is_independent() {
        let c = valid_config();
        let mut cloned = c.clone();
        cloned.url = "other://url".to_string();
        assert_eq!(c.url, "postgres://localhost/test");
    }

    // --- Custom AdapterConfig implementation ---

    struct MinimalConfig {
        pub value: u32,
    }

    impl AdapterConfig for MinimalConfig {
        fn adapter_name(&self) -> &'static str {
            "minimal"
        }

        fn validate(&self) -> Result<()> {
            if self.value == 0 {
                return Err(Error::config("value must be non-zero"));
            }
            Ok(())
        }
    }

    #[test]
    fn custom_config_validate_passes() {
        let cfg = MinimalConfig { value: 1 };
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.adapter_name(), "minimal");
    }

    #[test]
    fn custom_config_validate_fails() {
        let cfg = MinimalConfig { value: 0 };
        assert!(cfg.validate().is_err());
    }
}
