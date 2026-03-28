//! Tests for the in-memory graph adapter.

#[cfg(test)]
mod tests {
    use crate::graph::{
        adapter::{
            CausalOptions, EntitySearchOptions, GraphAdapter, RelationOptions, TraversalDirection,
        },
        adapters::memory::MemoryGraphAdapter,
        config::GraphConfig,
        error::Error,
        types::AssertionState,
    };

    async fn adapter() -> MemoryGraphAdapter {
        MemoryGraphAdapter::connect(GraphConfig::default()).await.unwrap()
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    #[tokio::test]
    async fn new_is_disconnected() {
        let a = MemoryGraphAdapter::new(GraphConfig::default());
        assert!(!a.is_connected());
    }

    #[tokio::test]
    async fn connect_produces_connected_adapter() {
        assert!(adapter().await.is_connected());
    }

    #[tokio::test]
    async fn operations_fail_when_not_connected() {
        let a = MemoryGraphAdapter::new(GraphConfig::default());
        let err = a.create_entity("e", "T", None, None).await.unwrap_err();
        assert!(matches!(err, Error::NotConnected));
    }

    #[tokio::test]
    async fn name_is_memory() {
        assert_eq!(adapter().await.name(), "memory");
    }

    // =========================================================================
    // Entity CRUD
    // =========================================================================

    #[tokio::test]
    async fn create_and_find_entity() {
        let a = adapter().await;
        let e = a.create_entity("Alice", "Person", None, None).await.unwrap();
        assert_eq!(e.name, "Alice");
        assert_eq!(e.entity_type, "Person");

        let found = a.find_entity(e.id, None).await.unwrap().unwrap();
        assert_eq!(found.id, e.id);
        assert_eq!(found.name, "Alice");
    }

    #[tokio::test]
    async fn find_missing_entity_returns_none() {
        let a = adapter().await;
        assert!(a.find_entity(9999, None).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn find_entity_by_name() {
        let a = adapter().await;
        a.create_entity("Bob", "Person", None, None).await.unwrap();
        let found = a.find_entity_by_name("Bob", None).await.unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().name, "Bob");
    }

    #[tokio::test]
    async fn search_entities_by_type() {
        let a = adapter().await;
        a.create_entity("Alice", "Person", None, None).await.unwrap();
        a.create_entity("Acme Corp", "Company", None, None).await.unwrap();
        let results = a
            .search_entities(EntitySearchOptions::new().with_type("Person"), None)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Alice");
    }

    #[tokio::test]
    async fn search_entities_by_query() {
        let a = adapter().await;
        a.create_entity("Alice Smith", "Person", None, None).await.unwrap();
        a.create_entity("Bob Jones", "Person", None, None).await.unwrap();
        let results = a
            .search_entities(EntitySearchOptions::new().with_query("alice"), None)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn update_entity_name() {
        let a = adapter().await;
        let e = a.create_entity("Old Name", "Type", None, None).await.unwrap();
        let updated = a
            .update_entity(e.id, [("name".to_string(), serde_json::json!("New Name"))].into(), None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(updated.name, "New Name");
    }

    #[tokio::test]
    async fn delete_entity_removes_it_and_its_relations() {
        let a = adapter().await;
        let e1 = a.create_entity("E1", "T", None, None).await.unwrap();
        let e2 = a.create_entity("E2", "T", None, None).await.unwrap();
        a.create_relation(e1.id, e2.id, "RELATED_TO", None, None).await.unwrap();

        assert!(a.delete_entity(e1.id, None).await.unwrap());
        assert!(a.find_entity(e1.id, None).await.unwrap().is_none());

        // Relation should also be gone
        let rels = a.find_relations(
            RelationOptions { from_id: Some(e1.id), ..Default::default() },
            None,
        ).await.unwrap();
        assert!(rels.is_empty());
    }

    #[tokio::test]
    async fn delete_missing_entity_returns_false() {
        let a = adapter().await;
        assert!(!a.delete_entity(9999, None).await.unwrap());
    }

    // =========================================================================
    // Relation CRUD
    // =========================================================================

    #[tokio::test]
    async fn create_and_find_relation() {
        let a = adapter().await;
        let e1 = a.create_entity("A", "T", None, None).await.unwrap();
        let e2 = a.create_entity("B", "T", None, None).await.unwrap();
        let r = a.create_relation(e1.id, e2.id, "CAUSES", None, None).await.unwrap();
        assert_eq!(r.from_id, e1.id);
        assert_eq!(r.to_id, e2.id);
        assert_eq!(r.relation_type, "CAUSES");

        let rels = a.find_relations(
            RelationOptions { from_id: Some(e1.id), ..Default::default() },
            None,
        ).await.unwrap();
        assert_eq!(rels.len(), 1);
    }

    #[tokio::test]
    async fn delete_relation() {
        let a = adapter().await;
        let e1 = a.create_entity("A", "T", None, None).await.unwrap();
        let e2 = a.create_entity("B", "T", None, None).await.unwrap();
        let r = a.create_relation(e1.id, e2.id, "CAUSES", None, None).await.unwrap();
        assert!(a.delete_relation(r.id, None).await.unwrap());
        assert!(!a.delete_relation(r.id, None).await.unwrap());
    }

    #[tokio::test]
    async fn upsert_relation_embedding_and_find_by_embedding() {
        let a = adapter().await;
        let e1 = a.create_entity("A", "T", None, None).await.unwrap();
        let e2 = a.create_entity("B", "T", None, None).await.unwrap();
        let r = a.create_relation(e1.id, e2.id, "CAUSES", None, None).await.unwrap();

        a.upsert_relation_embedding(r.id, vec![1.0, 0.0], None, None).await.unwrap();

        let found = a
            .find_relations_by_embedding(&[1.0, 0.0], 10, 0.9, None)
            .await
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, r.id);
    }

    // =========================================================================
    // Traversal
    // =========================================================================

    #[tokio::test]
    async fn neighbors_depth_one() {
        let a = adapter().await;
        let e1 = a.create_entity("A", "T", None, None).await.unwrap();
        let e2 = a.create_entity("B", "T", None, None).await.unwrap();
        let e3 = a.create_entity("C", "T", None, None).await.unwrap();
        a.create_relation(e1.id, e2.id, "CAUSES", None, None).await.unwrap();
        a.create_relation(e1.id, e3.id, "CAUSES", None, None).await.unwrap();

        let neighbors = a
            .neighbors(e1.id, 1, TraversalDirection::Outgoing, None)
            .await
            .unwrap();
        assert_eq!(neighbors.len(), 2);
    }

    #[tokio::test]
    async fn neighbors_depth_zero_is_invalid() {
        let a = adapter().await;
        let e = a.create_entity("A", "T", None, None).await.unwrap();
        let err = a.neighbors(e.id, 0, TraversalDirection::Both, None).await.unwrap_err();
        assert!(matches!(err, Error::InvalidArgument(_)));
    }

    #[tokio::test]
    async fn shortest_path_direct() {
        let a = adapter().await;
        let e1 = a.create_entity("A", "T", None, None).await.unwrap();
        let e2 = a.create_entity("B", "T", None, None).await.unwrap();
        a.create_relation(e1.id, e2.id, "CAUSES", None, None).await.unwrap();

        let path = a.shortest_path(e1.id, e2.id, None).await.unwrap();
        assert_eq!(path.len(), 2);
        assert_eq!(path[0].id, e1.id);
        assert_eq!(path[1].id, e2.id);
    }

    #[tokio::test]
    async fn shortest_path_no_connection_returns_empty() {
        let a = adapter().await;
        let e1 = a.create_entity("A", "T", None, None).await.unwrap();
        let e2 = a.create_entity("B", "T", None, None).await.unwrap();
        let path = a.shortest_path(e1.id, e2.id, None).await.unwrap();
        assert!(path.is_empty());
    }

    #[tokio::test]
    async fn subgraph_includes_seed_and_neighbors() {
        let a = adapter().await;
        let e1 = a.create_entity("A", "T", None, None).await.unwrap();
        let e2 = a.create_entity("B", "T", None, None).await.unwrap();
        let e3 = a.create_entity("C", "T", None, None).await.unwrap();
        a.create_relation(e1.id, e2.id, "CAUSES", None, None).await.unwrap();
        a.create_relation(e2.id, e3.id, "CAUSES", None, None).await.unwrap();

        let (entities, relations) = a.subgraph(&[e1.id], 2, None).await.unwrap();
        let ids: Vec<u64> = entities.iter().map(|e| e.id).collect();
        assert!(ids.contains(&e1.id));
        assert!(ids.contains(&e2.id));
        assert!(ids.contains(&e3.id));
        assert_eq!(relations.len(), 2);
    }

    // =========================================================================
    // Causal traversal
    // =========================================================================

    #[tokio::test]
    async fn causal_descendants_finds_chain() {
        let a = adapter().await;
        let e1 = a.create_entity("A", "T", None, None).await.unwrap();
        let e2 = a.create_entity("B", "T", None, None).await.unwrap();
        let e3 = a.create_entity("C", "T", None, None).await.unwrap();
        a.create_relation(e1.id, e2.id, "CAUSES", None, None).await.unwrap();
        a.create_relation(e2.id, e3.id, "ENABLES", None, None).await.unwrap();

        let paths = a
            .causal_descendants(e1.id, CausalOptions::default(), None)
            .await
            .unwrap();
        assert!(!paths.is_empty());
    }

    #[tokio::test]
    async fn causal_paths_between_entities() {
        let a = adapter().await;
        let e1 = a.create_entity("Root", "T", None, None).await.unwrap();
        let e2 = a.create_entity("Mid", "T", None, None).await.unwrap();
        let e3 = a.create_entity("Leaf", "T", None, None).await.unwrap();
        a.create_relation(e1.id, e2.id, "CAUSES", None, None).await.unwrap();
        a.create_relation(e2.id, e3.id, "TRIGGERS", None, None).await.unwrap();

        let paths = a
            .causal_paths(e1.id, e3.id, CausalOptions::default(), None)
            .await
            .unwrap();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].is_complete);
        assert_eq!(paths[0].hop_count(), 2);
    }

    #[tokio::test]
    async fn causal_paths_no_path_returns_empty() {
        let a = adapter().await;
        let e1 = a.create_entity("A", "T", None, None).await.unwrap();
        let e2 = a.create_entity("B", "T", None, None).await.unwrap();
        let paths = a.causal_paths(e1.id, e2.id, CausalOptions::default(), None).await.unwrap();
        assert!(paths.is_empty());
    }

    #[tokio::test]
    async fn non_causal_relations_not_followed() {
        let a = adapter().await;
        let e1 = a.create_entity("A", "T", None, None).await.unwrap();
        let e2 = a.create_entity("B", "T", None, None).await.unwrap();
        a.create_relation(e1.id, e2.id, "RELATED_TO", None, None).await.unwrap();

        let paths = a
            .causal_descendants(e1.id, CausalOptions::default(), None)
            .await
            .unwrap();
        assert!(paths.is_empty());
    }

    // =========================================================================
    // Temporal API
    // =========================================================================

    #[tokio::test]
    async fn retract_entity_marks_it_retracted() {
        let a = adapter().await;
        let e = a.create_entity("E", "T", None, None).await.unwrap();
        let retracted = a.retract_entity(e.id, None, None).await.unwrap();
        assert_eq!(retracted.assertion_state, AssertionState::Retracted);
    }

    #[tokio::test]
    async fn supersede_entity_creates_new_version() {
        let a = adapter().await;
        let old = a.create_entity("Old", "T", None, None).await.unwrap();
        let new = a
            .supersede_entity(
                old.id,
                [("name".to_string(), serde_json::json!("New"))].into(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(new.name, "New");
        assert_ne!(new.id, old.id);

        // Old record should be superseded
        let old_now = a.find_entity(old.id, None).await.unwrap().unwrap();
        assert_eq!(old_now.assertion_state, AssertionState::Superseded);
        assert_eq!(old_now.superseded_by, Some(new.id));
    }

    #[tokio::test]
    async fn retract_relation_marks_it_retracted() {
        let a = adapter().await;
        let e1 = a.create_entity("A", "T", None, None).await.unwrap();
        let e2 = a.create_entity("B", "T", None, None).await.unwrap();
        let r = a.create_relation(e1.id, e2.id, "CAUSES", None, None).await.unwrap();
        let retracted = a.retract_relation(r.id, None, None).await.unwrap();
        assert_eq!(retracted.assertion_state, AssertionState::Retracted);
    }

    // =========================================================================
    // External references
    // =========================================================================

    #[tokio::test]
    async fn add_and_find_external_ref() {
        use crate::graph::types::ExternalRef;
        let a = adapter().await;
        let e = a.create_entity("Acme", "Company", None, None).await.unwrap();
        let ext = ExternalRef {
            source: "lei".to_string(),
            external_id: "ABCDEF".to_string(),
            confidence: Some(0.99),
            valid_from: None,
            valid_to: None,
            properties: None,
        };
        a.add_external_ref(e.id, ext, None).await.unwrap();

        let found = a.find_by_external_ref("lei", "ABCDEF", true, None).await.unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].id, e.id);
    }

    #[tokio::test]
    async fn find_by_external_ref_returns_empty_when_no_match() {
        let a = adapter().await;
        let found = a.find_by_external_ref("wikidata", "Q999", true, None).await.unwrap();
        assert!(found.is_empty());
    }

    // =========================================================================
    // Community detection
    // =========================================================================

    #[tokio::test]
    async fn community_detection_separates_disconnected_subgraphs() {
        let a = adapter().await;
        let e1 = a.create_entity("A", "T", None, None).await.unwrap();
        let e2 = a.create_entity("B", "T", None, None).await.unwrap();
        let e3 = a.create_entity("C", "T", None, None).await.unwrap();
        let e4 = a.create_entity("D", "T", None, None).await.unwrap();

        a.create_relation(e1.id, e2.id, "RELATED_TO", None, None).await.unwrap();
        a.create_relation(e3.id, e4.id, "RELATED_TO", None, None).await.unwrap();

        let communities = a.detect_communities(None).await.unwrap();
        assert_eq!(communities.len(), 2);
    }

    #[tokio::test]
    async fn community_has_central_entity() {
        let a = adapter().await;
        let hub = a.create_entity("Hub", "T", None, None).await.unwrap();
        let spoke1 = a.create_entity("S1", "T", None, None).await.unwrap();
        let spoke2 = a.create_entity("S2", "T", None, None).await.unwrap();
        a.create_relation(hub.id, spoke1.id, "CAUSES", None, None).await.unwrap();
        a.create_relation(hub.id, spoke2.id, "CAUSES", None, None).await.unwrap();

        let communities = a.detect_communities(None).await.unwrap();
        assert_eq!(communities.len(), 1);
        assert!(communities[0].central_entity.is_some());
        assert_eq!(communities[0].central_entity.as_ref().unwrap().id, hub.id);
    }

    // =========================================================================
    // Namespace isolation
    // =========================================================================

    #[tokio::test]
    async fn namespaces_are_isolated() {
        use crate::common::namespace::Namespace;
        let a = adapter().await;
        let ns1 = Namespace::named("ns1");
        let ns2 = Namespace::named("ns2");

        a.create_entity("Alice", "Person", None, Some(&ns1)).await.unwrap();
        a.create_entity("Bob", "Person", None, Some(&ns2)).await.unwrap();

        let r1 = a.search_entities(EntitySearchOptions::new(), Some(&ns1)).await.unwrap();
        let r2 = a.search_entities(EntitySearchOptions::new(), Some(&ns2)).await.unwrap();

        assert_eq!(r1.len(), 1);
        assert_eq!(r2.len(), 1);
        assert_eq!(r1[0].name, "Alice");
        assert_eq!(r2[0].name, "Bob");
    }

    // =========================================================================
    // Healthcheck
    // =========================================================================

    #[tokio::test]
    async fn healthcheck_healthy_when_connected() {
        let r = adapter().await.healthcheck().await;
        assert!(r.status.is_healthy());
    }

    #[tokio::test]
    async fn healthcheck_unhealthy_when_not_connected() {
        let a = MemoryGraphAdapter::new(GraphConfig::default());
        let r = a.healthcheck().await;
        assert!(!r.status.is_usable());
    }
}
