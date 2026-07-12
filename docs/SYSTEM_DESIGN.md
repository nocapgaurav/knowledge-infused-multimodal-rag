# System Design

This document explains *why* Research Workspace is built the way it is, including which technology was chosen for each major decision and what the alternatives were. For *how* it works mechanically, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

> **A note on scope:** this document addresses the questions engineering reviewers typically ask about a RAG system's design — including two ("why FAISS," "why NetworkX") that assume libraries this project does not actually use. Rather than answer a hypothetical stack, the relevant sections below cover the equivalent real decisions: **why Qdrant** (the vector store actually used) and **why a real graph database (Neo4j) instead of an in-process graph library**. This keeps the document accurate to the codebase rather than describing a system that doesn't exist.

## Design Goals

The system was designed around three non-negotiable properties, in priority order:

1. **An answer must be verifiable, not just plausible.** A fluent answer that can't be checked against the source document is worse than no answer, because it's indistinguishable from a fabricated one. Every design decision downstream of this — Knowledge Units, citation resolution, grounding validation — exists to serve this property.
2. **Every pipeline stage must be independently inspectable.** A RAG system that only exposes "ask a question, get an answer" is nearly impossible to debug when retrieval quality degrades — there's no way to tell whether the problem is parsing, chunking, embedding, retrieval ranking, or generation. This system exposes each stage as its own endpoint with its own persisted manifest specifically so a failure or a quality regression can be localized to one stage.
3. **The system must run with zero external dependencies beyond open, self-hostable infrastructure.** No paid API key should be required to use, evaluate, or extend this project.

## Local-First Philosophy

Every model in this system — the embedding model, the generation model, the vision model — runs locally via Ollama and `sentence-transformers`, and every datastore (Qdrant, Neo4j) runs in a container on the same machine as the application. This wasn't a cost-cutting shortcut; it was the starting design constraint, for three concrete reasons:

- **Privacy is structural, not a policy.** A research paper — and everything derived from it — never has to cross a network boundary to a third party for the system to function. That's a property of the architecture, not a setting someone could accidentally leave misconfigured.
- **Reproducibility requires a fixed model.** A hosted API's underlying model can change behind a stable-looking endpoint name. A pinned local model (`qwen2.5:7b-instruct`, `gemma3:4b`, `BAAI/bge-m3`) answers the same way today as it will next month, which matters for a system whose evaluation harness (see `backend/evaluation/`) measures grounding and citation accuracy over time.
- **Zero marginal inference cost removes a barrier to experimentation.** Retrieval tuning (top-K, RRF weighting, graph expansion depth) benefits from being able to re-run hundreds of questions during development without a per-token bill. That kind of iteration is what actually produced the current retrieval parameters — it wouldn't have been affordable against a metered API.

The corollary is covered in the README's Deployment section: none of this is architecturally exclusive to local models. See **Future Migration Path** below.

## Why FastAPI

The backend needed to be a plain, typed HTTP service — nothing about the problem (document processing, retrieval, generation) called for a full batteries-included web framework. FastAPI was chosen specifically because:

- **Pydantic-native request/response validation** matches the domain modeling already needed for `Chunk`, `Paper`, and the various DTOs — one validation system, not two.
- **Dependency injection via `Depends()`** is what makes `backend/api/dependencies.py` a single, explicit place where every concrete provider (Docling, Qdrant, Neo4j, Ollama) gets wired up — routes never construct their own dependencies, which is what keeps route handlers "thin" by construction rather than by convention.
- **Async support** matters for a service whose real bottleneck is I/O-bound calls to Qdrant, Neo4j, and Ollama, not CPU-bound request handling.
- **Automatic OpenAPI docs** (`/docs`) give the frontend, and anyone extending the API, a live contract for free.

## Why Next.js

The frontend needed server-rendered routing (deep-linkable, refresh-safe workspace URLs — `/workspace/[documentId]` — rather than everything living behind client-side-only routing), while still being a highly interactive, stateful single-page-app-like experience once loaded. Next.js's App Router gives both: real routes backed by the filesystem, and full client-component interactivity where the workspace actually needs it (conversation panel, PDF viewer, resizable panels). Turbopack was adopted for the faster local iteration loop during development, which matters more for a project without a hosted deployment to fall back on for previewing changes.

## Why Ollama

Ollama was chosen over calling model weights directly (e.g. via `transformers` or `llama.cpp` bindings) because it provides a stable local HTTP API (`/api/chat`, `/api/tags`) that behaves like any other backend dependency the system already talks to (Qdrant, Neo4j) — the `OllamaProvider` is a thin, swappable adapter, not a process-management or GPU-memory-management concern the application itself has to own. It also makes the vision model (`gemma3:4b`) and the text model (`qwen2.5:7b-instruct`) interchangeable at the same interface, which is what let figure analysis (§10 of ARCHITECTURE.md) be added as a second model without touching the generation service's structure.

`qwen2.5:7b-instruct` specifically was selected for a practical reason: it's a 7B-class model capable of following the strict grounding/citation instructions in the system prompt reliably, while still being small enough to run with reasonable latency on consumer hardware — the entire point of local-first only holds if the default model is actually usable without a GPU cluster.

## Why Qdrant

*(The equivalent decision to "why FAISS" for this codebase — FAISS is not used here.)*

FAISS is an excellent in-process similarity search library, but it is a library, not a service: it has no notion of a "collection," no payload filtering, and no persistence model beyond what the calling process manages itself. This system needed per-document collections with metadata filtering (searches are always scoped to one document's vectors) and a stable process boundary between "the thing that computes embeddings" and "the thing that serves vector search" — the same reason Neo4j was chosen over an in-process graph library (below). Qdrant provides collection-level isolation, payload-based filtering, and persistence as a running service with its own lifecycle, addressable exactly like the other infrastructure dependencies (Neo4j, Ollama) via a URL in settings — one consistent operational model for every piece of local infrastructure, rather than one thing embedded in-process and everything else external.

## Why Knowledge Units

The alternative to a uniform "Knowledge Unit" abstraction would be treating text, tables, and figures as fundamentally different data types with separate retrieval, ranking, and citation code paths for each. That would have made two things much harder: (1) fusing evidence of different modalities into one ranked list — Reciprocal Rank Fusion only works cleanly when every candidate, regardless of modality, exposes the same score-relevant fields — and (2) citation resolution, which needs to point at "the thing that was cited" without caring whether that thing was a paragraph, a table, or a figure. Making every modality collapse into the same `Chunk` shape (with a `modality` field for where they *do* need different treatment — embedding input, vision analysis eligibility) is what lets retrieval, ranking, and citation code stay modality-agnostic, while modality-specific logic stays isolated to exactly the two places it's needed: the chunking strategies (§5) and the generation-time figure analyst (§10).

## Why Graph-Based Retrieval

Pure vector search treats every Knowledge Unit as an independent point in embedding space, which is a poor model for a scientific paper: sections continue across chunk boundaries, methods sections reference figures defined elsewhere, and claims cite specific bibliography entries. A citation-graph traversal (Neo4j, `CITES`/`REFERENCES`/`CONTINUES` edges) recovers exactly the structure vector search throws away — it can pull in a cited reference that is not textually similar to the question at all, because it's connected to something that is.

A real graph database was chosen over an in-process graph library for the same reason Qdrant was chosen over an in-process vector library: the graph needs to persist across requests, be queried with a real query language (Cypher) for bounded traversals (`retrieval_max_expansion_depth`, `retrieval_max_neighbors_per_node`), and be operable as its own service with its own lifecycle — not rebuilt in memory on every process start.

The design also deliberately keeps the graph from ever being a source of new text: expansion only ever returns node *ids*, which are then re-fetched from Qdrant. This guarantees the graph can never introduce a hallucinated or stale piece of evidence — it can only ever help *find* real, already-indexed content faster.

## State Management Decisions

**Backend: fully stateless per request.** There is no session concept and no server-side conversation memory (see ARCHITECTURE.md §12) — deliberately. A stateless backend is trivially horizontally scalable and trivially cacheable per document, and it keeps "what does the model know about this conversation" fully auditable: it's exactly what's in the current request, nothing implicit.

**Frontend: client state and server state kept in separate systems.** Zustand owns UI/selection state (which document is open, panel sizes, conversation history) with `persist` middleware to `localStorage`, so a refresh doesn't lose a session. TanStack Query owns anything that's actually fetched from the backend, with its own cache invalidation. Mixing these two into one store is a common source of bugs (stale server data treated as UI state, or UI state accidentally sent back to a server that doesn't expect it) that this separation avoids by construction.

## Storage Decisions

Every pipeline stage writes to its own directory under `data/{stage}/{document_id}/`, mirroring the backend's own package boundaries (`backend/storage`, one `WorkspaceStorage` instance per stage). This was chosen over one shared document store for the same inspectability reason as the seven-stage pipeline itself: if retrieval quality degrades for one document, the exact JSON that stage produced is on disk, per document, per stage — there's no need to reconstruct intermediate state from a single monolithic record. The trade-off is more files on disk than a single database would produce; that trade-off was accepted deliberately in exchange for every stage's output being directly readable without a client library or query.

## Scalability Considerations

The current design scales along the axis it was built for — **more documents, more questions per document** — cleanly: each document has its own Qdrant collection and its own graph subtree, so retrieval cost for one document's questions does not grow with the total number of documents in the system, and a stateless backend can run multiple instances behind a load balancer with no shared in-process state to coordinate. It does **not** currently scale along a different axis — **cross-document retrieval** — because that was explicitly out of scope for the first version (see the README's Limitations and Future Work sections); answering "compare this claim across my whole library" would require a retrieval design that spans collections and graphs, which the current per-document isolation does not provide.

## Performance Considerations

The generation pipeline's context budget (`context_window - max_tokens - 512` reserved overhead, ~4 chars/token estimate) is deliberately conservative — it trades a small amount of usable context for eliminating truncation failures at the Ollama call boundary. Retrieval's evidence caps (5 groups, 2 primaries per section) exist for the same reason on the input side: bounding the prompt size keeps latency predictable regardless of how large or heavily-cited the source paper is, rather than latency scaling with document size. Graph expansion's traversal cost budget (`retrieval_max_traversal_cost`) exists specifically to prevent a densely-connected paper (many citations, many cross-references) from turning a 2-hop BFS into an unbounded query — the depth limit alone isn't sufficient protection against a high-fan-out graph.

## Future Migration Path to Cloud Inference

The design that makes local-first possible — a thin, interface-bound provider per external dependency (`interfaces/generation_provider.py`, `interfaces/embedding_provider.py`, `interfaces/vector_store.py`, `interfaces/knowledge_graph_store.py`) — is the same design that makes migrating to hosted infrastructure a swap, not a rewrite. `OllamaProvider` is the only file that imports the `ollama` package; nothing above it (the generation service, the prompt composer, the citation resolver) knows or cares that generation happens locally. Replacing it with an equivalent provider for OpenAI, Anthropic, Azure OpenAI, or a self-hosted inference server (vLLM, TGI, Triton) means implementing the same `GenerationProvider` interface against a different SDK and updating `backend/api/dependencies.py`'s wiring — the retrieval pipeline, the citation system, and the frontend do not change. The same applies to `EmbeddingProvider` for a hosted embedding API, and `VectorStore`/`KnowledgeGraphStore` for a managed Qdrant Cloud or Neo4j Aura instance instead of a local container. Local-first was chosen as the default for the reasons above, not because the architecture is incapable of anything else.
