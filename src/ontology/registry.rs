//! Versioned ontology registry.
//!
//! The registry stores named, versioned [`Definition`] objects and tracks
//! which version is active for each name. It mirrors the semantics of
//! `Cortex::Ontology::Registry` in Ruby.
//!
//! # Version source of truth
//!
//! `Definition::version` is the canonical version. The `version` argument to
//! [`OntologyRegistry::register`] is an **override/fallback only**:
//!
//! - If `version` is `None`, `definition.version` is used as the storage key.
//! - If `version` is `Some(v)` and differs from `definition.version`, `v` is
//!   the registry key — `definition.version` is not mutated.
//! - If both are `None`, registration fails with [`Error::VersionRequired`].
//!
//! # Thread safety
//!
//! [`MemoryOntologyRegistry`] is built on `DashMap` and is safe to share
//! across threads via `Arc`.

use std::sync::Arc;

use dashmap::DashMap;

use crate::ontology::{
    error::{Error, Result},
    types::Definition,
};

/// A versioned, named store for ontology definitions.
pub trait OntologyRegistry: Send + Sync {
    /// Store a definition.
    ///
    /// `version` is the registry storage key (override/fallback).
    /// If `None`, `definition.version` is used. Fails if neither is set.
    /// If `set_active` is `true` (the default), this version becomes active.
    fn register(
        &self,
        name: &str,
        definition: Definition,
        version: Option<&str>,
        set_active: bool,
    ) -> Result<()>;

    /// Retrieve a definition by name.
    ///
    /// If `version` is `None`, the currently active version is returned.
    /// Returns `Err(NotFound)` if the name or version does not exist, or if
    /// no active version is set.
    fn get(&self, name: &str, version: Option<&str>) -> Result<Arc<Definition>>;

    /// Make `version` the active version for `name`.
    ///
    /// Returns `Err(NotFound)` if the named version does not exist.
    fn activate(&self, name: &str, version: &str) -> Result<()>;

    /// List all registered version strings for `name` in insertion order.
    fn versions(&self, name: &str) -> Vec<String>;

    /// Return the currently active version string for `name`, or `None`.
    fn active_version(&self, name: &str) -> Option<String>;

    /// List all registered ontology names in insertion order.
    fn names(&self) -> Vec<String>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory implementation
// ─────────────────────────────────────────────────────────────────────────────

/// An in-process, thread-safe ontology registry backed by [`DashMap`].
///
/// Suitable for local development, integration tests, and as the runtime
/// registry when a Postgres backend is not required. Definitions are held in
/// memory only and do not persist across process restarts.
pub struct MemoryOntologyRegistry {
    /// Outer key: ontology name.
    /// Inner key: version string.
    definitions: DashMap<String, DashMap<String, Arc<Definition>>>,
    /// Tracks the active version per name.
    active: DashMap<String, String>,
    /// Preserves insertion order for `versions()` and `names()`.
    name_order: parking_lot::RwLock<Vec<String>>,
    version_order: DashMap<String, parking_lot::RwLock<Vec<String>>>,
}

impl Default for MemoryOntologyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryOntologyRegistry {
    /// Construct an empty in-memory registry.
    pub fn new() -> Self {
        Self {
            definitions: DashMap::new(),
            active: DashMap::new(),
            name_order: parking_lot::RwLock::new(Vec::new()),
            version_order: DashMap::new(),
        }
    }

    fn resolve_version<'a>(
        definition: &'a Definition,
        version_override: Option<&'a str>,
    ) -> Result<&'a str> {
        if let Some(v) = version_override {
            return Ok(v);
        }
        definition
            .version
            .as_deref()
            .ok_or_else(|| Error::version_required(
                "definition has no version and no version: override was supplied",
            ))
    }
}

impl OntologyRegistry for MemoryOntologyRegistry {
    fn register(
        &self,
        name: &str,
        definition: Definition,
        version: Option<&str>,
        set_active: bool,
    ) -> Result<()> {
        let key = Self::resolve_version(&definition, version)?.to_string();

        // Track name insertion order
        {
            let mut order = self.name_order.write();
            if !order.contains(&name.to_string()) {
                order.push(name.to_string());
            }
        }

        let versions_map = self.definitions.entry(name.to_string()).or_default();
        // Track version insertion order
        self.version_order
            .entry(name.to_string())
            .or_insert_with(|| parking_lot::RwLock::new(Vec::new()));
        {
            let vo = self.version_order.get(name).unwrap();
            let mut vo_write = vo.write();
            if !vo_write.contains(&key) {
                vo_write.push(key.clone());
            }
        }
        versions_map.insert(key.clone(), Arc::new(definition));

        if set_active {
            self.active.insert(name.to_string(), key);
        }

        Ok(())
    }

    fn get(&self, name: &str, version: Option<&str>) -> Result<Arc<Definition>> {
        let key = match version {
            Some(v) => v.to_string(),
            None => self
                .active
                .get(name)
                .map(|r| r.value().clone())
                .ok_or_else(|| {
                    Error::not_found(format!("no active version for '{}'", name))
                })?,
        };

        self.definitions
            .get(name)
            .and_then(|versions| versions.get(&key).map(|d| d.value().clone()))
            .ok_or_else(|| Error::not_found(format!("{}@{}", name, key)))
    }

    fn activate(&self, name: &str, version: &str) -> Result<()> {
        // Verify the version exists before activating it
        self.get(name, Some(version))?;
        self.active.insert(name.to_string(), version.to_string());
        Ok(())
    }

    fn versions(&self, name: &str) -> Vec<String> {
        match self.version_order.get(name) {
            Some(order) => order.read().clone(),
            None => Vec::new(),
        }
    }

    fn active_version(&self, name: &str) -> Option<String> {
        self.active.get(name).map(|r| r.value().clone())
    }

    fn names(&self) -> Vec<String> {
        self.name_order.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::types::Definition;

    fn def(name: &str, version: &str) -> Definition {
        let mut d = Definition::new(name);
        d.version = Some(version.to_string());
        d
    }

    fn registry() -> MemoryOntologyRegistry {
        MemoryOntologyRegistry::new()
    }

    // ── register / get basic ──

    #[test]
    fn register_and_get_active() {
        let r = registry();
        r.register("regulatory", def("Regulatory", "1.0.0"), None, true).unwrap();
        let d = r.get("regulatory", None).unwrap();
        assert_eq!(d.version.as_deref(), Some("1.0.0"));
    }

    #[test]
    fn register_with_version_override() {
        let r = registry();
        let mut d = Definition::new("Regulatory");
        d.version = None; // no version on the definition
        r.register("regulatory", d, Some("1.0.0"), true).unwrap();
        let fetched = r.get("regulatory", Some("1.0.0")).unwrap();
        assert!(fetched.version.is_none()); // definition.version unchanged
    }

    #[test]
    fn register_fails_without_any_version() {
        let r = registry();
        let d = Definition::new("NoVersion"); // no version
        let result = r.register("x", d, None, true);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), crate::ontology::error::Error::VersionRequired(_)));
    }

    #[test]
    fn get_specific_version() {
        let r = registry();
        r.register("ont", def("Ont", "1.0.0"), None, false).unwrap();
        r.register("ont", def("Ont", "1.1.0"), None, true).unwrap();
        let v1 = r.get("ont", Some("1.0.0")).unwrap();
        assert_eq!(v1.version.as_deref(), Some("1.0.0"));
        let v2 = r.get("ont", Some("1.1.0")).unwrap();
        assert_eq!(v2.version.as_deref(), Some("1.1.0"));
    }

    #[test]
    fn get_not_found_by_name() {
        let r = registry();
        let result = r.get("nonexistent", None);
        assert!(result.is_err());
    }

    #[test]
    fn get_not_found_by_version() {
        let r = registry();
        r.register("ont", def("Ont", "1.0.0"), None, true).unwrap();
        let result = r.get("ont", Some("9.9.9"));
        assert!(result.is_err());
    }

    // ── set_active = false ──

    #[test]
    fn register_with_set_active_false_does_not_activate() {
        let r = registry();
        r.register("ont", def("Ont", "1.0.0"), None, false).unwrap();
        assert!(r.active_version("ont").is_none());
        // Can still get by explicit version
        let d = r.get("ont", Some("1.0.0")).unwrap();
        assert_eq!(d.version.as_deref(), Some("1.0.0"));
    }

    // ── activate ──

    #[test]
    fn activate_changes_active_version() {
        let r = registry();
        r.register("ont", def("Ont", "1.0.0"), None, true).unwrap();
        r.register("ont", def("Ont", "1.1.0"), None, false).unwrap();
        assert_eq!(r.active_version("ont").as_deref(), Some("1.0.0"));
        r.activate("ont", "1.1.0").unwrap();
        assert_eq!(r.active_version("ont").as_deref(), Some("1.1.0"));
    }

    #[test]
    fn activate_rollback() {
        let r = registry();
        r.register("ont", def("Ont", "1.0.0"), None, false).unwrap();
        r.register("ont", def("Ont", "1.1.0"), None, true).unwrap();
        r.activate("ont", "1.0.0").unwrap();
        assert_eq!(r.active_version("ont").as_deref(), Some("1.0.0"));
    }

    #[test]
    fn activate_nonexistent_version_fails() {
        let r = registry();
        r.register("ont", def("Ont", "1.0.0"), None, true).unwrap();
        let result = r.activate("ont", "9.9.9");
        assert!(result.is_err());
    }

    // ── versions() ──

    #[test]
    fn versions_empty_for_unknown_name() {
        let r = registry();
        assert!(r.versions("unknown").is_empty());
    }

    #[test]
    fn versions_in_insertion_order() {
        let r = registry();
        r.register("ont", def("Ont", "1.0.0"), None, false).unwrap();
        r.register("ont", def("Ont", "1.1.0"), None, false).unwrap();
        r.register("ont", def("Ont", "2.0.0"), None, true).unwrap();
        let vs = r.versions("ont");
        assert_eq!(vs, vec!["1.0.0", "1.1.0", "2.0.0"]);
    }

    // ── names() ──

    #[test]
    fn names_empty_initially() {
        let r = registry();
        assert!(r.names().is_empty());
    }

    #[test]
    fn names_in_insertion_order() {
        let r = registry();
        r.register("beta", def("Beta", "1.0"), None, true).unwrap();
        r.register("alpha", def("Alpha", "1.0"), None, true).unwrap();
        let names = r.names();
        assert_eq!(names, vec!["beta", "alpha"]);
    }

    #[test]
    fn re_registering_same_name_does_not_duplicate_in_names() {
        let r = registry();
        r.register("ont", def("Ont", "1.0.0"), None, false).unwrap();
        r.register("ont", def("Ont", "1.1.0"), None, true).unwrap();
        assert_eq!(r.names().len(), 1);
    }

    // ── Arc sharing ──

    #[test]
    fn get_returns_arc() {
        let r = registry();
        r.register("ont", def("Ont", "1.0.0"), None, true).unwrap();
        let a = r.get("ont", None).unwrap();
        let b = r.get("ont", Some("1.0.0")).unwrap();
        // Both arcs point to the same allocation
        assert!(Arc::ptr_eq(&a, &b));
    }
}
