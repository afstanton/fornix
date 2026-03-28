//! Health status reporting for storage adapters.

use std::time::Instant;

/// The health of a storage adapter at a point in time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthStatus {
    /// The adapter is operating normally.
    Healthy,

    /// The adapter is reachable but operating with reduced capacity or
    /// elevated latency. Callers may continue with caution.
    Degraded {
        /// Human-readable explanation of the degraded condition.
        reason: String,
    },

    /// The adapter is not functional. Callers should not attempt operations.
    Unhealthy {
        /// Human-readable explanation of the failure.
        reason: String,
    },
}

impl HealthStatus {
    /// Returns `true` if the adapter is fully healthy.
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Healthy)
    }

    /// Returns `true` if the adapter is usable (healthy or degraded).
    pub fn is_usable(&self) -> bool {
        !matches!(self, Self::Unhealthy { .. })
    }

    /// Returns the reason string for degraded or unhealthy states.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Degraded { reason } | Self::Unhealthy { reason } => Some(reason.as_str()),
            Self::Healthy => None,
        }
    }
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded { reason } => write!(f, "degraded: {}", reason),
            Self::Unhealthy { reason } => write!(f, "unhealthy: {}", reason),
        }
    }
}

/// A health check result with timing information.
#[derive(Debug, Clone)]
pub struct HealthReport {
    /// The adapter name or identifier.
    pub adapter: String,
    /// The health status at the time of the check.
    pub status: HealthStatus,
    /// How long the health check itself took to complete.
    pub latency: std::time::Duration,
    /// When the check was performed.
    pub checked_at: std::time::SystemTime,
}

impl HealthReport {
    /// Begin timing a health check. Call `.finish(status)` when done.
    pub fn begin(adapter: impl Into<String>) -> PendingHealthReport {
        PendingHealthReport {
            adapter: adapter.into(),
            started_at: Instant::now(),
        }
    }
}

/// A health check in progress. Created by [`HealthReport::begin`].
pub struct PendingHealthReport {
    adapter: String,
    started_at: Instant,
}

impl PendingHealthReport {
    /// Complete the health check with the given status.
    pub fn finish(self, status: HealthStatus) -> HealthReport {
        HealthReport {
            adapter: self.adapter,
            status,
            latency: self.started_at.elapsed(),
            checked_at: std::time::SystemTime::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    // --- HealthStatus predicates ---

    #[test]
    fn healthy_is_healthy_and_usable() {
        let s = HealthStatus::Healthy;
        assert!(s.is_healthy());
        assert!(s.is_usable());
    }

    #[test]
    fn degraded_is_not_healthy_but_is_usable() {
        let s = HealthStatus::Degraded { reason: "high latency".to_string() };
        assert!(!s.is_healthy());
        assert!(s.is_usable());
    }

    #[test]
    fn unhealthy_is_not_healthy_and_not_usable() {
        let s = HealthStatus::Unhealthy { reason: "connection refused".to_string() };
        assert!(!s.is_healthy());
        assert!(!s.is_usable());
    }

    // --- HealthStatus::reason ---

    #[test]
    fn healthy_has_no_reason() {
        assert_eq!(HealthStatus::Healthy.reason(), None);
    }

    #[test]
    fn degraded_reason_is_accessible() {
        let s = HealthStatus::Degraded { reason: "pool exhausted".to_string() };
        assert_eq!(s.reason(), Some("pool exhausted"));
    }

    #[test]
    fn unhealthy_reason_is_accessible() {
        let s = HealthStatus::Unhealthy { reason: "disk full".to_string() };
        assert_eq!(s.reason(), Some("disk full"));
    }

    // --- Display ---

    #[test]
    fn healthy_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
    }

    #[test]
    fn degraded_display_includes_reason() {
        let s = HealthStatus::Degraded { reason: "slow queries".to_string() };
        assert_eq!(s.to_string(), "degraded: slow queries");
    }

    #[test]
    fn unhealthy_display_includes_reason() {
        let s = HealthStatus::Unhealthy { reason: "no route to host".to_string() };
        assert_eq!(s.to_string(), "unhealthy: no route to host");
    }

    // --- Equality ---

    #[test]
    fn healthy_equals_healthy() {
        assert_eq!(HealthStatus::Healthy, HealthStatus::Healthy);
    }

    #[test]
    fn degraded_equal_when_same_reason() {
        let a = HealthStatus::Degraded { reason: "r".to_string() };
        let b = HealthStatus::Degraded { reason: "r".to_string() };
        assert_eq!(a, b);
    }

    #[test]
    fn degraded_not_equal_with_different_reason() {
        let a = HealthStatus::Degraded { reason: "a".to_string() };
        let b = HealthStatus::Degraded { reason: "b".to_string() };
        assert_ne!(a, b);
    }

    #[test]
    fn healthy_not_equal_to_degraded() {
        let degraded = HealthStatus::Degraded { reason: "r".to_string() };
        assert_ne!(HealthStatus::Healthy, degraded);
    }

    // --- HealthReport timing ---

    #[test]
    fn report_stores_adapter_name_and_status() {
        let report = HealthReport::begin("pgvector").finish(HealthStatus::Healthy);
        assert_eq!(report.adapter, "pgvector");
        assert_eq!(report.status, HealthStatus::Healthy);
    }

    #[test]
    fn report_latency_is_non_negative() {
        let report = HealthReport::begin("memory").finish(HealthStatus::Healthy);
        // Latency must be a non-negative duration (it can be zero on fast machines)
        assert!(report.latency.as_nanos() < 1_000_000_000); // under 1s for a no-op
    }

    #[test]
    fn report_checked_at_is_recent() {
        let before = SystemTime::now();
        let report = HealthReport::begin("test").finish(HealthStatus::Healthy);
        let after = SystemTime::now();
        assert!(report.checked_at >= before);
        assert!(report.checked_at <= after);
    }

    #[test]
    fn report_carries_degraded_status() {
        let status = HealthStatus::Degraded { reason: "slow".to_string() };
        let report = HealthReport::begin("redis").finish(status.clone());
        assert_eq!(report.status, status);
        assert!(!report.status.is_healthy());
        assert!(report.status.is_usable());
    }

    #[test]
    fn report_carries_unhealthy_status() {
        let status = HealthStatus::Unhealthy { reason: "down".to_string() };
        let report = HealthReport::begin("neo4j").finish(status.clone());
        assert_eq!(report.status, status);
        assert!(!report.status.is_usable());
    }

    #[test]
    fn report_clone_is_independent() {
        let report = HealthReport::begin("x").finish(HealthStatus::Healthy);
        let cloned = report.clone();
        assert_eq!(cloned.adapter, report.adapter);
        assert_eq!(cloned.status, report.status);
    }
}
