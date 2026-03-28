//! Namespace type for scoping stored records.
//!
//! Namespaces allow a single adapter instance to serve multiple logical
//! partitions — e.g. different tenants, document types, or embedding models —
//! without requiring separate connections or tables.

/// A namespace identifier used to scope stored records.
///
/// `None` means the default namespace for that adapter.
/// `Some("documents")` scopes to the "documents" partition.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Namespace(pub Option<String>);

impl Namespace {
    /// The default (unscoped) namespace.
    pub fn default_ns() -> Self {
        Self(None)
    }

    /// A named namespace.
    pub fn named(name: impl Into<String>) -> Self {
        Self(Some(name.into()))
    }

    /// Returns the namespace as an `Option<&str>`.
    pub fn as_deref(&self) -> Option<&str> {
        self.0.as_deref()
    }

    /// Returns `true` if this is the default namespace.
    pub fn is_default(&self) -> bool {
        self.0.is_none()
    }
}

impl From<&str> for Namespace {
    fn from(s: &str) -> Self {
        Self::named(s)
    }
}

impl From<String> for Namespace {
    fn from(s: String) -> Self {
        Self::named(s)
    }
}

impl From<Option<String>> for Namespace {
    fn from(opt: Option<String>) -> Self {
        Self(opt)
    }
}

impl From<Option<&str>> for Namespace {
    fn from(opt: Option<&str>) -> Self {
        Self(opt.map(String::from))
    }
}

impl std::fmt::Display for Namespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Some(s) => write!(f, "{}", s),
            None => write!(f, "<default>"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_ns_is_none() {
        let ns = Namespace::default_ns();
        assert!(ns.is_default());
        assert_eq!(ns.as_deref(), None);
        assert_eq!(ns.0, None);
    }

    #[test]
    fn default_trait_produces_default_ns() {
        let ns = Namespace::default();
        assert!(ns.is_default());
    }

    #[test]
    fn named_stores_the_name() {
        let ns = Namespace::named("documents");
        assert!(!ns.is_default());
        assert_eq!(ns.as_deref(), Some("documents"));
    }

    #[test]
    fn named_accepts_string_and_str() {
        let from_str = Namespace::named("tenants");
        let from_string = Namespace::named("tenants".to_string());
        assert_eq!(from_str, from_string);
    }

    #[test]
    fn from_str_slice() {
        let ns: Namespace = "embeddings".into();
        assert_eq!(ns.as_deref(), Some("embeddings"));
    }

    #[test]
    fn from_owned_string() {
        let ns: Namespace = String::from("graphs").into();
        assert_eq!(ns.as_deref(), Some("graphs"));
    }

    #[test]
    fn from_option_string_some() {
        let ns: Namespace = Some(String::from("cache")).into();
        assert_eq!(ns.as_deref(), Some("cache"));
    }

    #[test]
    fn from_option_string_none() {
        let ns: Namespace = Option::<String>::None.into();
        assert!(ns.is_default());
    }

    #[test]
    fn from_option_str_some() {
        let ns: Namespace = Some("vectors").into();
        assert_eq!(ns.as_deref(), Some("vectors"));
    }

    #[test]
    fn from_option_str_none() {
        let ns: Namespace = Option::<&str>::None.into();
        assert!(ns.is_default());
    }

    #[test]
    fn display_default_shows_placeholder() {
        let ns = Namespace::default_ns();
        assert_eq!(ns.to_string(), "<default>");
    }

    #[test]
    fn display_named_shows_name() {
        let ns = Namespace::named("my-tenant");
        assert_eq!(ns.to_string(), "my-tenant");
    }

    #[test]
    fn equality_same_name() {
        assert_eq!(Namespace::named("a"), Namespace::named("a"));
    }

    #[test]
    fn equality_different_names() {
        assert_ne!(Namespace::named("a"), Namespace::named("b"));
    }

    #[test]
    fn equality_named_vs_default() {
        assert_ne!(Namespace::named("x"), Namespace::default_ns());
    }

    #[test]
    fn clone_produces_equal_value() {
        let ns = Namespace::named("clone-me");
        assert_eq!(ns.clone(), ns);
    }

    #[test]
    fn usable_as_hashmap_key() {
        use std::collections::HashMap;
        let mut map: HashMap<Namespace, &str> = HashMap::new();
        map.insert(Namespace::named("ns1"), "value1");
        map.insert(Namespace::default_ns(), "default");
        assert_eq!(map[&Namespace::named("ns1")], "value1");
        assert_eq!(map[&Namespace::default_ns()], "default");
    }
}
