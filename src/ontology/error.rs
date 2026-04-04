//! Error types for the ontology module.

use thiserror::Error;

/// Errors produced by ontology operations.
#[derive(Debug, Error)]
pub enum Error {
    /// The provided configuration is invalid.
    #[error("configuration error: {0}")]
    Configuration(String),

    /// An ontology, version, or type was not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// A version is required but was not provided and could not be inferred.
    #[error("version required: {0}")]
    VersionRequired(String),

    /// An entity or relation type is not in the ontology.
    #[error("unknown entity type: {0}")]
    UnknownEntityType(String),

    /// A relation type is not in the ontology.
    #[error("unknown relation type: {0}")]
    UnknownRelationType(String),

    /// A graph write violated the active ontology schema.
    #[error("ontology violation: {0}")]
    Violation(String),

    /// JSON serialisation or deserialisation failed.
    #[error("serialisation error: {0}")]
    Serialisation(String),

    /// A general operation failure.
    #[error("operation failed: {0}")]
    Operation(String),
}

impl Error {
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Configuration(msg.into())
    }
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }
    pub fn version_required(msg: impl Into<String>) -> Self {
        Self::VersionRequired(msg.into())
    }
    pub fn violation(msg: impl Into<String>) -> Self {
        Self::Violation(msg.into())
    }
    pub fn serialisation(msg: impl Into<String>) -> Self {
        Self::Serialisation(msg.into())
    }
    pub fn operation(msg: impl Into<String>) -> Self {
        Self::Operation(msg.into())
    }
}

/// Shorthand result type for ontology operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_message() {
        assert_eq!(Error::config("bad").to_string(), "configuration error: bad");
    }

    #[test]
    fn not_found_message() {
        assert_eq!(
            Error::not_found("regulatory@1.0").to_string(),
            "not found: regulatory@1.0"
        );
    }

    #[test]
    fn version_required_message() {
        assert!(Error::version_required("no version on definition").to_string().contains("version required"));
    }

    #[test]
    fn unknown_entity_type_message() {
        assert!(Error::UnknownEntityType("Foo".into()).to_string().contains("Foo"));
    }

    #[test]
    fn violation_message() {
        assert!(Error::violation("entity type not in ontology").to_string().contains("violation"));
    }

    #[test]
    fn serialisation_message() {
        assert!(Error::serialisation("invalid json").to_string().contains("serialisation"));
    }

    #[test]
    fn result_ok() {
        let r: Result<i32> = Ok(42);
        assert!(r.is_ok());
    }

    #[test]
    fn result_err() {
        let r: Result<i32> = Err(Error::config("x"));
        assert!(r.is_err());
    }
}
