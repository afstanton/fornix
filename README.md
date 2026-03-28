# fornix

Knowledge storage, retrieval, and graph infrastructure for cognitive systems.

fornix is a modular Rust library for building retrieval-heavy and agentic systems. It combines vector search, BM25 search, hybrid fusion, graph reasoning, GraphRAG, routing, prompt tuning, and an autonomous tool-using agent runtime behind feature flags so you can enable only what you need.

## Who This Is For

- Teams building RAG pipelines that need both lexical and semantic retrieval.
- Systems that need graph-aware retrieval and causal traversal.
- Agent runtimes that require strict policy controls around tool usage.
- Projects that prefer composable, feature-gated crates over monolithic frameworks.

## Highlights

- Feature-gated architecture for lean builds.
- Pure in-memory adapters for fast local development and testing.
- Typed, composable APIs across storage, retrieval, graph, and agent layers.
- Built-in evaluation, filtering, and query-gap tracking for RAG workflows.
- End-to-end tested with unit tests plus doctests.

## Installation

Add the crate with only the features you want.

```toml
[dependencies]
fornix = { version = "0.2.0", features = ["vector", "bm25", "hybrid"] }
```

Or enable everything:

```toml
[dependencies]
fornix = { version = "0.2.0", features = ["full"] }
```

## Pick Features by Use Case

- Basic vector search: `vector`
- Keyword search: `bm25`
- Hybrid retrieval (vector + BM25): `vector`, `bm25`, `hybrid`
- Graph knowledge layer: `graph`
- Graph-augmented retrieval: `graph`, `rag`, `hybrid`, `graphrag`
- Prompt optimization: `rag`, `tuner`
- Agent runtime with routing: `router`, `agent`

Minimal hybrid example dependency:

```toml
[dependencies]
fornix = { version = "0.2.0", features = ["vector", "bm25", "hybrid"] }
```

## Feature Flags

The crate is layered; higher-level modules depend on lower-level ones:

- `store`: base adapter traits, config, health, and error types.
- `cache`: caching adapters (`store`).
- `vector`: vector adapters and analysis (`store`).
- `bm25`: keyword retrieval adapters (`store`).
- `hybrid`: fused vector + BM25 retrieval (`vector`, `bm25`).
- `graph`: knowledge graph + temporal/causal APIs (`store`).
- `rag`: chunking, filters, eval, reranking (`hybrid`).
- `graphrag`: graph-augmented retrieval (`graph`, `rag`, `hybrid`).
- `router`: model routing strategies + forest (`cache`).
- `diff`: boundary-aware textual diff snippets.
- `tuner`: prompt optimization strategies (`rag`).
- `agent`: recursive tool-using agent runtime (`router`).
- `full`: enables all modules.

The `common` module is always available.

## Module Overview

- `store`: foundational traits such as `StorageAdapter` and `AdapterFactory`.
- `cache`: memory/null cache adapters and deterministic cache keying.
- `vector`: vector storage, nearest-neighbor search, and embedding analytics.
- `bm25`: Okapi BM25 tokenization + scoring + indexing.
- `hybrid`: weighted fusion (`Rrf`, `Linear`) and confidence scoring.
- `graph`: entity/relation graph with bitemporal semantics and causal traversal.
- `graphrag`: local/global/hybrid graph search modes.
- `rag`: chunkers, post-filters, rerankers, evaluation metrics, query-gap tracker.
- `router`: regex, round-robin, weighted random, embedding-threshold, and RoRF routing.
- `diff`: focused and stitched snippets with change markers.
- `tuner`: MIPROv2, GEPA, and no-op prompt tuning.
- `agent`: solve loop with tool execution, recursion, policy controls, and token budgeting.

## Quick Start

Most adapters are async. Use a Tokio runtime in your app:

```rust,no_run
use fornix::vector::{VectorAdapter, VectorConfig, adapters::MemoryVectorAdapter};
use fornix::vector::adapter::SearchOptions;

#[tokio::main]
async fn main() {
	let adapter = MemoryVectorAdapter::connect(VectorConfig::with_dimension(2))
		.await
		.unwrap();

	adapter.upsert("doc-1", vec![1.0, 0.0], None, None).await.unwrap();

	let results = adapter
		.nearest_neighbors(&[1.0, 0.0], None, SearchOptions::default())
		.await
		.unwrap();

	println!("top id: {}", results[0].id);
}
```

### Hybrid Retrieval

```rust,no_run
use fornix::bm25::{adapters::MemoryBm25Adapter, adapter::IndexDocument, Bm25Adapter, Bm25Config};
use fornix::vector::{adapters::MemoryVectorAdapter, VectorAdapter, VectorConfig};
use fornix::hybrid::{HybridConfig, HybridSearch, search::HybridSearchOptions};

#[tokio::main]
async fn main() {
	let bm25 = MemoryBm25Adapter::connect(Bm25Config::default()).await.unwrap();
	let vector = MemoryVectorAdapter::connect(VectorConfig::with_dimension(2)).await.unwrap();

	bm25.index(IndexDocument::new("doc-1", "rust systems programming"), None)
		.await
		.unwrap();
	vector.upsert("doc-1", vec![1.0, 0.0], None, None).await.unwrap();

	let search = HybridSearch::new(bm25, vector, HybridConfig::default());
	let results = search
		.search("rust", &[1.0, 0.0], None, HybridSearchOptions::new())
		.await
		.unwrap();

	println!("hybrid top: {}", results[0].id);
}
```

### Graph + Causal Traversal

```rust,no_run
use fornix::graph::{adapters::MemoryGraphAdapter, GraphAdapter, GraphConfig};
use fornix::graph::adapter::CausalOptions;

#[tokio::main]
async fn main() {
	let graph = MemoryGraphAdapter::connect(GraphConfig::default()).await.unwrap();

	let rain = graph.create_entity("Heavy Rain", "Weather", None, None).await.unwrap();
	let flood = graph.create_entity("Flooding", "Event", None, None).await.unwrap();

	graph.create_relation(rain.id, flood.id, "CAUSES", None, None)
		.await
		.unwrap();

	let paths = graph
		.causal_descendants(rain.id, CausalOptions::default(), None)
		.await
		.unwrap();

	println!("first relation: {}", paths[0].edges[0].relation_type);
}
```

## Agent Runtime

The `agent` module provides an autonomous solve loop that can:

- call an injected model client,
- dispatch registered tools,
- recurse through sub-tasks,
- enforce policy constraints,
- track token/time/step budgets,
- compact memory when context grows.

You provide implementations for `ModelClient` and `ToolRegistry`.

## Design Notes

- `common` is always available and not feature-gated.
- In-memory adapters are intentionally first-class for deterministic tests and local prototyping.
- Higher-level modules (`graphrag`, `agent`) build on lower-level primitives rather than hiding them.
- APIs favor explicit typed options (`SearchOptions`, `CausalOptions`, config structs) over implicit globals.

## Development

Run all tests:

```bash
cargo test --features full
```

Run clippy across all targets/features:

```bash
cargo clippy --all-targets --all-features
```

## Notes on Adapters

- In-memory adapters are ideal for local development, integration tests, and reference behavior.
- Backend adapters (Postgres/Qdrant/Redis) are exposed as module surfaces; some implementations are currently stubs in this crate layout.
- For production deployments, verify backend adapter completeness against your selected features before rollout.

## Versioning

Current crate version: `0.3.1`.

As the module surface is broad and evolving, pin a minor version in production environments and review changelogs before upgrading.

## License

MIT
