//! Community detection algorithms for knowledge graphs.
//!
//! Two algorithms are provided:
//! - **Connected components** — pure Rust DFS, no dependencies
//! - **Leiden** — reserved for the Leiden native extension (stubbed)

use std::collections::{HashMap, HashSet};

/// A community expressed as a set of entity ids.
pub type CommunityIds = Vec<Vec<String>>;

/// An edge in the community detection graph.
#[derive(Debug, Clone)]
pub struct Edge {
    pub from_id: String,
    pub to_id: String,
    pub weight: f64,
}

impl Edge {
    pub fn new(from: impl Into<String>, to: impl Into<String>, weight: f64) -> Self {
        Self { from_id: from.into(), to_id: to.into(), weight }
    }

    pub fn unweighted(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self::new(from, to, 1.0)
    }
}

/// Detect communities using the connected-components algorithm.
///
/// Returns a list of components; each component is a `Vec<String>` of entity ids.
/// Isolated nodes (no edges) appear as single-element components.
pub fn connected_components(node_ids: &[String], edges: &[Edge]) -> CommunityIds {
    // Build undirected adjacency list
    let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();
    for edge in edges {
        adjacency.entry(&edge.from_id).or_default().push(&edge.to_id);
        adjacency.entry(&edge.to_id).or_default().push(&edge.from_id);
    }

    let mut visited: HashSet<&str> = HashSet::new();
    let mut components: CommunityIds = Vec::new();

    for node_id in node_ids {
        let key = node_id.as_str();
        if visited.contains(key) {
            continue;
        }

        // DFS to collect the connected component
        let mut component: Vec<String> = Vec::new();
        let mut stack = vec![key];

        while let Some(current) = stack.pop() {
            if visited.contains(current) {
                continue;
            }
            visited.insert(current);
            component.push(current.to_string());

            if let Some(neighbors) = adjacency.get(current) {
                for neighbor in neighbors {
                    if !visited.contains(*neighbor) {
                        stack.push(neighbor);
                    }
                }
            }
        }

        if !component.is_empty() {
            components.push(component);
        }
    }

    components
}

/// Leiden community detection (stubbed).
///
/// The Leiden algorithm produces higher-quality communities than connected
/// components but requires the native Leiden extension (compiled Rust via
/// the cortex_graph_leiden cdylib). This function is a placeholder that
/// falls back to connected components until the extension is wired in.
///
/// When the extension is available, call the native entry point directly.
pub fn leiden(node_ids: &[String], edges: &[Edge]) -> CommunityIds {
    // TODO: replace with call to native Leiden implementation
    connected_components(node_ids, edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(s: &[&str]) -> Vec<String> {
        s.iter().map(|id| id.to_string()).collect()
    }

    fn edge(from: &str, to: &str) -> Edge {
        Edge::unweighted(from, to)
    }

    // --- connected_components ---

    #[test]
    fn empty_graph_returns_no_components() {
        assert!(connected_components(&[], &[]).is_empty());
    }

    #[test]
    fn single_node_no_edges_is_its_own_component() {
        let components = connected_components(&ids(&["a"]), &[]);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0], vec!["a"]);
    }

    #[test]
    fn two_connected_nodes_form_one_component() {
        let components = connected_components(&ids(&["a", "b"]), &[edge("a", "b")]);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 2);
    }

    #[test]
    fn two_disconnected_nodes_form_two_components() {
        let components = connected_components(&ids(&["a", "b"]), &[]);
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn triangle_is_one_component() {
        let nodes = ids(&["a", "b", "c"]);
        let edges = vec![edge("a", "b"), edge("b", "c"), edge("a", "c")];
        let components = connected_components(&nodes, &edges);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 3);
    }

    #[test]
    fn two_disconnected_pairs() {
        let nodes = ids(&["a", "b", "c", "d"]);
        let edges = vec![edge("a", "b"), edge("c", "d")];
        let components = connected_components(&nodes, &edges);
        assert_eq!(components.len(), 2);
        assert!(components.iter().all(|c| c.len() == 2));
    }

    #[test]
    fn edges_are_treated_as_undirected() {
        // edge a→b should make a and b reachable from each other
        let nodes = ids(&["a", "b"]);
        let edges = vec![edge("a", "b")];
        let components = connected_components(&nodes, &edges);
        assert_eq!(components.len(), 1);
    }

    #[test]
    fn all_components_together_cover_all_nodes() {
        let nodes = ids(&["a", "b", "c", "d", "e"]);
        let edges = vec![edge("a", "b"), edge("c", "d")];
        let components = connected_components(&nodes, &edges);
        let total: usize = components.iter().map(|c| c.len()).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn weighted_edges_accepted_but_weight_ignored_in_connected_components() {
        let nodes = ids(&["x", "y"]);
        let edges = vec![Edge::new("x", "y", 0.5)];
        let components = connected_components(&nodes, &edges);
        assert_eq!(components.len(), 1);
    }

    #[test]
    fn chain_of_nodes_forms_one_component() {
        let nodes = ids(&["a", "b", "c", "d", "e"]);
        let edges = vec![
            edge("a", "b"),
            edge("b", "c"),
            edge("c", "d"),
            edge("d", "e"),
        ];
        let components = connected_components(&nodes, &edges);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 5);
    }

    // --- Edge construction ---

    #[test]
    fn unweighted_edge_has_weight_one() {
        let e = Edge::unweighted("a", "b");
        assert!((e.weight - 1.0).abs() < 1e-10);
    }

    #[test]
    fn weighted_edge_stores_weight() {
        let e = Edge::new("a", "b", 0.7);
        assert!((e.weight - 0.7).abs() < 1e-10);
    }
}
