<div align="center">

# Knowledge-Infused Multimodal RAG

**Research Workspace** — evidence-grounded question answering over scientific PDFs, with retrieval fused across text, tables, figures, and a citation graph.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?logo=fastapi&logoColor=white)](backend/api/app.py)
[![Next.js 15](https://img.shields.io/badge/Next.js-15.5-000000.svg?logo=nextdotjs&logoColor=white)](frontend/package.json)
[![React 19](https://img.shields.io/badge/React-19.1-61DAFB.svg?logo=react&logoColor=black)](frontend/package.json)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6.svg?logo=typescript&logoColor=white)](frontend/tsconfig.json)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector%20search-DC244C.svg)](docker-compose.yml)
[![Neo4j](https://img.shields.io/badge/Neo4j-knowledge%20graph-008CC1.svg?logo=neo4j&logoColor=white)](docker-compose.yml)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-000000.svg)](backend/config/settings.py)
[![Local-first](https://img.shields.io/badge/deployment-local--first-6f42c1.svg)](#deployment)

[Overview](#overview) · [Features](#key-features) · [Architecture](#architecture-overview) · [Pipeline](#end-to-end-pipeline) · [Getting Started](#getting-started) · [Deployment](#deployment)

</div>

---

## Overview

**Knowledge-Infused Multimodal RAG** (shown to end users as **Research Workspace**) is a full-stack application for asking questions about scientific papers and getting answers you can actually verify.

Upload a PDF. The system parses it into text, tables, and figures; builds a graph of how those pieces relate to one another (citations, references, section continuations); embeds and indexes everything; and answers questions using only what the paper actually says — with every claim traceable back to a specific sentence, table, or figure, at a specific page and location in the original document.

Every stage runs against real infrastructure — Qdrant, Neo4j, and a local Ollama model — with no mocking, and is covered by over 100 automated tests, including full pipeline integration tests and a retrieval/generation benchmark suite.

## Motivation

General-purpose LLMs are fluent, but not naturally honest about scientific documents: they will answer from prior training data instead of the paper in front of them, they routinely ignore tables and figures in favor of surrounding text, and they rarely show *where* in the document a claim actually came from.

This project exists to address three specific problems:

1. **Trustworthy answers.** Every sentence that states a fact is required to carry a citation back to a specific knowledge unit — a paragraph, a table, or a figure — that a reader can verify immediately, in the original PDF.
2. **Figures and tables as first-class evidence.** Tables keep their row/column structure through parsing instead of being flattened into prose, and figures are analyzed by a vision model when a question is actually about them.
3. **Retrieval beyond a single similarity score.** Scientific papers are structured documents — sections continue, works cite each other, references get reused. Backing vector search with a real graph traversal surfaces evidence that nearest-neighbor search alone would miss.

## Key Features

- **Evidence-grounded answers** — the model answers only from retrieved evidence, never outside knowledge, and every factual sentence must carry a citation.
- **Multimodal retrieval** — text, tables, and figures are all first-class citizens in the index, not text-only with images bolted on.
- **Knowledge-graph-backed retrieval** — vector search is expanded through a citation/reference graph in Neo4j, surfacing cited references and continuing sections that pure top-K search would miss.
- **Hybrid ranking** — Reciprocal Rank Fusion over four independent signals (dense similarity, lexical overlap, graph proximity, relationship confidence).
- **Clickable citations with exact PDF navigation** — click a citation and the PDF viewer jumps to the exact page and highlights the exact region it came from.
- **On-demand visual reasoning** — figure-specific questions trigger a local vision-language model to describe what's actually visible in the figure image, layered on top of its caption.
- **Per-document conversation history** — every uploaded paper keeps its own independent conversation thread.
- **Full generation traceability** — every answer carries a phase-by-phase trace (planning, retrieval, grounding, citation resolution).
- **A real evaluation harness** — retrieval metrics (recall, precision, MRR, nDCG, hit rate), generation metrics (citation accuracy, grounding accuracy, unsupported-claim rate), and operational metrics (latency, throughput, CPU, memory), run against a real benchmark dataset.
- **Local-first by design** — parsing, embedding, and generation all run locally; a document never has to leave the user's machine to be processed.

## Screenshots

> Screenshots pending capture. Once available, they will be added below and referenced from this section.

| | |
|---|---|
| Upload & document library | _placeholder_ |
| Conversation with citations | _placeholder_ |
| Citation → PDF navigation | _placeholder_ |
| Evidence panel | _placeholder_ |
| Figure explanation | _placeholder_ |

## Demo

> A recorded walkthrough will be linked here once available.

## Architecture Overview

The application is two independently deployable services plus a small set of local infrastructure:

```mermaid
flowchart LR
    subgraph Client["Browser"]
        UI["Research Workspace UI<br/>(Next.js 15 / React 19)"]
    end

    subgraph Backend["FastAPI Backend"]
        API["REST API<br/>(backend/api)"]
        Pipeline["Document Pipeline<br/>parse → represent → embed → index → graph"]
        RAG["Retrieval + Generation"]
    end

    subgraph Infra["Local Infrastructure"]
        Qdrant[("Qdrant<br/>vector search")]
        Neo4j[("Neo4j<br/>knowledge graph")]
        Ollama["Ollama<br/>qwen2.5:7b-instruct + gemma3:4b vision"]
        FS[("Local filesystem<br/>data/*")]
    end

    UI <-->|REST / JSON| API
    API --> Pipeline
    API --> RAG
    Pipeline --> FS
    Pipeline --> Qdrant
    Pipeline --> Neo4j
    RAG --> Qdrant
    RAG --> Neo4j
    RAG --> Ollama
    RAG --> FS
```

- **Frontend** (`frontend/`) — a Next.js 15 / React 19 research workspace: a document sidebar, a conversation panel, an evidence panel, and a PDF viewer, kept in sync by a small set of Zustand stores and TanStack Query.
- **Backend** (`backend/`) — a FastAPI service structured as one vertical package per pipeline stage (ingestion, parsing, chunking, embeddings, search, graph, retrieval, generation, evaluation), each following the same internal shape: `interfaces/` (ports) → `providers/` (the only place a vendor SDK is imported) → `services/` (orchestration) → `repository/` (per-stage storage) → `validator/` (structural checks at every phase boundary).
- **Infrastructure** — Qdrant and Neo4j run in Docker; Ollama runs natively on the host and serves both the text generation model and the vision model used for figure analysis.

A complete technical breakdown — full data flow, backend and frontend architecture, every pipeline in detail, storage layout, and error handling, with Mermaid diagrams throughout — lives in **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**. For the engineering reasoning behind these choices, including why each major technology was selected over its alternatives, see **[`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md)**.

## Technology Stack

| Layer | Technology | Why |
|---|---|---|
| Backend | Python 3.12 · FastAPI · Pydantic v2 · Uvicorn | Pydantic-native validation shared with the domain model, `Depends()`-based DI as the single wiring point, and async support for a service whose real bottleneck is I/O to Qdrant/Neo4j/Ollama. |
| Document parsing | [Docling](https://github.com/DS4SD/docling) — layout analysis, table structure extraction, figure rendering | Preserves real table cell structure and renders figures to image assets in one library, instead of flattening tables to text and needing a separate tool for figures. |
| Embeddings | `sentence-transformers`, default model `BAAI/bge-m3` | A strong, long-context, locally-runnable embedding model — no hosted embedding API required. |
| Vector search | [Qdrant](https://qdrant.tech/) (cosine similarity) | A running service with native per-document collections and payload filtering, not an in-process library (e.g. FAISS) that would need collection isolation and persistence hand-rolled. |
| Knowledge graph | [Neo4j](https://neo4j.com/) 5 Community | A persistent, independently queryable graph (Cypher, bounded traversals) — the citation/reference graph must survive and be queried across requests, which an in-memory graph library (e.g. NetworkX) isn't built for. |
| Generation | [Ollama](https://ollama.com/) — `qwen2.5:7b-instruct` for text answers, `gemma3:4b` for figure vision | A stable local HTTP API turns "the model" into just another networked dependency, like Qdrant and Neo4j, with no in-process model/GPU-memory management. `qwen2.5:7b` follows strict grounding instructions while staying light enough for consumer hardware. |
| Frontend | Next.js 15 (App Router, Turbopack) · React 19 · TypeScript 5 | Deep-linkable, refresh-safe workspace routes alongside a fully interactive client experience; static types at the API/view-model boundary catch integration bugs at compile time. |
| UI | Tailwind CSS v4 · shadcn/ui on Base UI primitives · `react-resizable-panels` · `react-pdf` · Framer Motion · `cmdk` | shadcn generates owned component source rather than an opaque dependency, and `react-pdf` gives the page/coordinate-level control the citation-highlighting feature depends on. |
| State / data | Zustand (client state, persisted) · TanStack Query (server state) · Axios · Zod + React Hook Form | Client state (selection, panel sizes, conversation history) and server state (fetched data) are deliberately kept in separate systems with different caching models. |
| Testing | Pytest (backend, 95 test files) · Vitest + Testing Library (frontend unit/component) · Playwright (frontend E2E) · `vitest-axe` (accessibility) | Playwright's auto-waiting model suits a UI backed by genuinely asynchronous, real-infrastructure operations (document processing, LLM generation) without flaky sleeps. |
| Infra | Docker Compose (Qdrant + Neo4j) · local filesystem artifact storage per pipeline stage | One command gives every contributor identical, disposable local infrastructure. |

The full reasoning behind each of these choices — including the alternatives considered and the trade-offs accepted — is in [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md).

## Repository Structure

```
knowledge-infused-multimodal-rag/
├── backend/                     FastAPI service — one vertical package per pipeline stage
│   ├── api/                     routes, request/response schemas, DI wiring (app.py, dependencies.py)
│   ├── domain/                  vendor-independent entities: Chunk, Paper, Relationship, BoundingBox...
│   ├── ingestion/                upload handling, raw PDF storage
│   ├── parser/                   Docling-backed parsing → structured Paper (text, tables, figures)
│   ├── chunking/                 Paper → Knowledge Units ("Chunk"s) + relationships
│   ├── embeddings/                Knowledge Units → vector embeddings
│   ├── search/                    embeddings → Qdrant index
│   ├── graph/                     Knowledge Units → Neo4j knowledge graph
│   ├── retrieval/                 question → ranked, graph-expanded evidence bundle
│   ├── generation/                evidence bundle → grounded, cited answer (via Ollama)
│   ├── evaluation/                real benchmark suite (retrieval + generation + ops metrics)
│   └── storage/                   shared local-filesystem storage abstraction
├── frontend/                    Next.js 15 "Research Workspace" UI
│   ├── app/                       App Router pages (landing, workspace/[documentId], settings)
│   ├── components/                 conversation, evidence, pdf, navigation, workspace, ui (shadcn)
│   ├── store/                      Zustand: conversation, document library, workspace, accessibility
│   ├── services/                   API client + TanStack Query hooks
│   ├── hooks/, types/, utils/, constants/, lib/, providers/
│   └── tests/                      unit, component, hooks, integration, e2e
├── data/                         per-document pipeline artifacts (raw → parsed → knowledge → embeddings → index → graph → retrieval → generation)
├── docs/                         architecture documentation
├── tests/                        backend pytest suite
├── docker-compose.yml            Qdrant + Neo4j
└── pyproject.toml                backend package + tooling config
```

## End-to-End Pipeline

Uploading a paper runs it through seven backend stages, each a REST call, each persisting its own artifact under `data/`:

```mermaid
flowchart TD
    A["1 · Ingest<br/>POST /documents"] --> B["2 · Parse<br/>POST /documents/{id}/parse"]
    B --> C["3 · Represent<br/>POST /documents/{id}/represent"]
    C --> D["4 · Embed<br/>POST /documents/{id}/embed"]
    D --> E["5 · Index<br/>POST /documents/{id}/index"]
    E --> F["6 · Graph<br/>POST /documents/{id}/graph"]
    F --> G["Ready for questions"]
    G --> H["7 · Generate<br/>POST /documents/{id}/generate"]

    A -.-> A1[("data/raw/{id}")]
    B -.-> B1[("data/parsed/{id}")]
    C -.-> C1[("data/knowledge/{id}")]
    D -.-> D1[("data/embeddings/{id}")]
    E -.-> E1[("Qdrant collection")]
    F -.-> F1[("Neo4j graph")]
    H -.-> H1[("data/generation/{id}")]
```

| Stage | Endpoint | What happens |
|---|---|---|
| **1. Ingest** | `POST /documents` | The uploaded PDF is validated (size limit, type) and persisted as-is with its upload metadata. |
| **2. Parse** | `POST /documents/{id}/parse` | [Docling](https://github.com/DS4SD/docling) runs full layout analysis: text blocks, tables (with cell-level row/column structure), and figures (rendered to PNG), each carrying page-level bounding boxes. |
| **3. Represent** | `POST /documents/{id}/represent` | The parsed paper is walked in reading order and split into **Knowledge Units** — see below. |
| **4. Embed** | `POST /documents/{id}/embed` | Each Knowledge Unit is embedded with `sentence-transformers` (`BAAI/bge-m3` by default), prefixed with its structural label (e.g. `"Figure 2: ..."`) so retrieval understands *what kind* of thing it's matching, not just its raw text. |
| **5. Index** | `POST /documents/{id}/index` | Embeddings are upserted into a per-document Qdrant collection (cosine similarity). |
| **6. Graph** | `POST /documents/{id}/graph` | Knowledge Units and their relationships (citations, references, section continuations) are written into Neo4j as a real property graph. |
| **7. Generate** | `POST /documents/{id}/generate` | A question triggers retrieval, then generation — see the dedicated sections below. |

## Knowledge Unit Pipeline

A **Knowledge Unit** is the atomic, citable piece of evidence in this system — internally a `Chunk` (`backend/domain/chunk.py`), one of `text`, `table`, or `figure`. Every Knowledge Unit carries:

- its **modality** (`text` / `table` / `figure`),
- its **reading-order position** in the document (used for neighbor expansion),
- a **structural label** (`retrieval_context`) like `"Section: III. Methodology"`, `"Figure 2"`, or `"Authors and affiliations (title page)"`,
- and one or more **bounding boxes** (page number + coordinates) — the exact region of the original PDF it came from.

```mermaid
flowchart LR
    P["Parsed Paper<br/>(text blocks, tables, figures)"] --> W["Walk in reading order"]
    W --> S1["ParagraphStrategy"]
    W --> S2["TableStrategy"]
    W --> S3["FigureStrategy"]
    S1 --> KU["Knowledge Unit<br/>(Chunk)"]
    S2 --> KU
    S3 --> KU
    KU --> REL["Relationships<br/>CITES · REFERENCES · CONTINUES"]
    KU --> OUT[("knowledge_units.json")]
    REL --> OUT2[("relationships.json")]
```

- **Paragraphs** are split with a configurable ceiling (250 words) and floor (4 words), so a Knowledge Unit is neither a whole page nor a single orphaned sentence.
- **Tables** keep their caption fused with a markdown export of their actual cell structure (rows, columns, spans, headers) — a table is never flattened into a paragraph of prose.
- **Figures** carry their caption as text and a pointer (`asset_uri`) to the rendered PNG, so the figure image itself is available later for vision analysis.

## Retrieval Pipeline

Retrieval is a four-phase pipeline that combines dense search with a real graph traversal, not a single vector-search call:

```mermaid
flowchart TD
    Q["Question"] --> C1["Phase 1 · Candidate Generation<br/>embed query → Qdrant top-K (K=20, cosine)"]
    C1 --> C2["Phase 2 · Evidence Expansion<br/>graph BFS in Neo4j (≤2 hops)<br/>CITES · REFERENCES · CONTINUES"]
    C2 --> C3["Phase 3 · Evidence Evaluation<br/>Reciprocal Rank Fusion (k=60)<br/>over 4 signals"]
    C3 --> C4["Phase 4 · Evidence Assembly<br/>group into ≤5 evidence groups<br/>≤2 primaries per section"]
    C4 --> B["EvidenceBundle"]
```

1. **Candidate Generation** — the question is embedded with the same model used for indexing and matched against Qdrant (`top_k = 20`, cosine similarity), scoped to the current document.
2. **Evidence Expansion** — a budgeted breadth-first traversal from those seed results through Neo4j, following `CITES` / `REFERENCES` / `CONTINUES` edges up to 2 hops deep, so a directly-cited reference or a continuing section can surface even if it wasn't itself a top vector match. Anything the graph discovers is re-fetched from Qdrant — the graph never invents text of its own.
3. **Evidence Evaluation** — the combined pool is ranked with **Reciprocal Rank Fusion** (`score = Σ 1 / (60 + rank)`) across four independent signals: dense similarity, lexical term overlap, graph proximity, and relationship-type confidence.
4. **Evidence Assembly** — the ranked pool is grouped into evidence groups (a primary Knowledge Unit plus its directly-connected supporting units), capped for diversity so one section can't dominate the whole answer.

## Figure and Table Reasoning

Every modality is retrievable, but each is prepared differently:

- **Text** is embedded directly, with its structural label prefixed.
- **Tables** keep cell-level structure from Docling (rows, columns, spans, headers) and are embedded as caption + structured markdown together — so a question about a specific column or comparison can match the table's actual content, not a lossy text summary of it.
- **Figures** are embedded by their **caption only** at index time. The extra reasoning happens **at answer time, on demand**: when a question is classified as figure-centric, a local vision-language model (`gemma3:4b` via Ollama) is shown the actual rendered figure image and asked to describe, concretely, what's visually in it — components, layout, arrows, axes, labels. That description is explicitly layered onto the caption and clearly marked as automated visual analysis, never presented as the paper's own words, and the whole step degrades gracefully (falls back to caption-only) if the image or model is unavailable.

## Citation System

Every answer is required to cite its evidence, and every citation is independently verified before it reaches the client:

```mermaid
sequenceDiagram
    participant LLM as Ollama (qwen2.5:7b-instruct)
    participant CR as CitationResolver
    participant FE as Frontend
    participant PDF as PDF Viewer

    LLM->>CR: "...as shown in [KU3]..."
    CR->>CR: regex-extract every [KUn] / (KUn)
    CR->>CR: resolve strictly against this prompt's label→evidence map
    CR-->>FE: ResolvedCitation { knowledge_unit_id, page_numbers, bounding_boxes, modality }
    FE->>FE: render citation as a clickable label in the answer
    FE->>PDF: on click — jump to page, highlight bounding box
```

- Each evidence item shown to the model is tagged with a short label (`KU1`, `KU2`, ...) plus its structural identity, e.g. `[KU3] (Figure 2) ...`.
- After generation, the `CitationResolver` regex-extracts every citation the model actually wrote and resolves it **strictly against the exact label map built for that specific prompt** — a citation to a label that was never shown is marked unresolved rather than trusted.
- Resolved citations carry the originating Knowledge Unit's page numbers and bounding boxes straight through to the frontend, which is what lets clicking a citation jump the PDF viewer to the exact page and highlight the exact region — not just "somewhere in this document."

## Conversation Memory

Conversation history is **per-document and entirely client-side**. The backend has no concept of a session or a conversation — every `/documents/{id}/generate` call is a single, independent question with no server-side memory. The frontend (`store/conversation-store.ts`, Zustand + persisted to `localStorage`) keeps one running thread of question/answer turns per uploaded document, so switching between papers shows each paper's own conversation, and a hard refresh doesn't lose history.

## Example Usage Workflow

A typical session, end to end:

1. **Upload** a PDF from the landing page. It appears in the sidebar immediately with a "Preparing" status while the backend runs it through the seven-stage pipeline.
2. **Wait** for the status to change to "Ready" — this covers parsing, Knowledge Unit extraction, embedding, indexing, and graph construction.
3. **Ask a general question**, e.g. *"What is this paper about?"* — the answer arrives with inline citations like `(Section: I. Introduction)`.
4. **Click a citation** in the answer. The right-hand panel switches to the PDF tab and jumps directly to the page and highlighted region the citation came from.
5. **Ask about a figure specifically**, e.g. *"Explain Figure 1 in detail."* — this triggers on-demand vision analysis of the actual figure image, not just its caption.
6. **Ask a follow-up**, e.g. *"What did you just say about Figure 2 specifically?"* — the conversation panel retains context for this document, so the question doesn't need to restate what it's referring to.
7. **Upload a second paper** and switch between the two in the sidebar. Each document keeps its own independent conversation history, PDF view, and evidence panel.

## Limitations

- **Figures are retrieved by caption text only.** There is no dedicated image embedding model yet, so a figure with a sparse caption can be harder to retrieve on visual similarity alone — vision analysis only kicks in once a figure-centric question has already surfaced it.
- **Conversation context is not sent to the backend.** Each question is answered independently; the model does not see prior turns, so pronoun-heavy or highly ambiguous follow-up questions can occasionally be misinterpreted.
- **Retrieval and the knowledge graph are scoped per document.** There is no cross-document search — each uploaded paper has its own isolated Qdrant collection and graph.
- **No continuous integration pipeline.** The test suite (95 backend files, 12 frontend files) currently runs locally only.

## Future Work

- Add a dedicated image embedding model so figures can be matched on visual similarity, not just caption text.
- Support cross-document retrieval and citation for users working across a whole library of papers.
- Introduce server-side conversation context so follow-up questions can resolve references across turns.
- Wire the existing test suite into a CI pipeline.

## Getting Started

This project is **local-first**: parsing, embedding, retrieval, and answer generation all run on your own machine, using a local Ollama model rather than a hosted API. See [Deployment](#deployment) for why.

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | **3.12** (not 3.13+, not 3.11) | `pyproject.toml` pins `>=3.12,<3.13` |
| Node.js | 20+ | with npm |
| Git | any recent version | to clone the repository |
| Docker | any recent version | runs Qdrant + Neo4j |
| [Ollama](https://ollama.com) | any recent version | runs the local LLM and vision model |

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/nocapgaurav/knowledge-infused-multimodal-rag.git
cd knowledge-infused-multimodal-rag
```

**2. Backend setup**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
deactivate
```

**3. Frontend setup**

```bash
cd frontend
npm install
cd ..
```

**4. Environment variables (optional)**

Both halves read configuration from environment variables, with working defaults for local development — copying these files is only needed if you want to change something:

```bash
cp .env.example .env                       # backend: host/port, Qdrant/Neo4j/Ollama URLs, CORS
cp frontend/.env.example frontend/.env.local  # frontend: NEXT_PUBLIC_API_BASE_URL
```

### Ollama Setup

```bash
# Install Ollama — see https://ollama.com/download for your platform

# Start the Ollama service (skip if it's already running as a background service)
ollama serve

# Pull the models this project uses
ollama pull qwen2.5:7b-instruct   # text generation
ollama pull gemma3:4b             # figure vision

# Verify both models are available
ollama list
curl http://localhost:11434/api/tags
```

### Running the Project

Each of the following runs in its own terminal.

**1. Infrastructure — Qdrant + Neo4j**

```bash
docker compose up -d
```

**2. Ollama** (if not already running as a service)

```bash
ollama serve
# → http://localhost:11434
```

**3. Backend**

```bash
source .venv/bin/activate
uvicorn backend.api.app:create_app --factory --host 127.0.0.1 --port 8000
# → http://localhost:8000
# → http://localhost:8000/docs   (interactive OpenAPI docs)
```

**4. Frontend**

```bash
cd frontend
npm run dev
# → http://localhost:3000
```

Open **http://localhost:3000**. The status indicator in the footer should read **"Backend connected."**

```bash
# Sanity-check the backend directly
curl http://localhost:8000/health     # {"status":"ok"}
```

<details>
<summary><strong>Troubleshooting</strong></summary>

<br>

**Frontend shows "Backend unavailable" / uploads fail with "Could not reach the server."**
1. Confirm the backend process is actually running and healthy: `curl http://localhost:8000/health`.
2. If that fails, the backend isn't running. The most common cause is an incompatible or missing Python environment: `which uvicorn` should point inside `.venv/bin/`.
3. If `curl` succeeds but the browser still can't reach it, check the browser console for a CORS error and confirm `cors_allowed_origins` in `backend/config/settings.py` includes the frontend's actual origin.

**`pip install -e ".[dev]"` fails or `ModuleNotFoundError` on startup.**
Check `python --version` inside the activated venv — it must be 3.12.x.

**Uploads succeed but questions never get answered.**
Confirm Ollama is running and has the configured models pulled: `curl http://localhost:11434/api/tags`.

</details>

## Deployment

This project is intentionally **not deployed publicly**, and that is a design decision, not a limitation.

Research Workspace performs multimodal retrieval and answer generation entirely with local models — a local embedding model, a local vector database, a local knowledge graph, and a local LLM served through Ollama. That architecture is what makes three things possible at once:

- **Privacy** — an uploaded paper, and everything derived from it (embeddings, graph, generated answers), never leaves the machine it was processed on.
- **Reproducibility** — every result comes from a fixed, versioned local model rather than a hosted API that can change behavior without notice.
- **Zero API cost** — there is no third-party inference bill; the only requirement is the compute to run Ollama locally.

Running the system therefore means running it on your own machine, following the [Getting Started](#getting-started) section above — there is no hosted demo, because the entire point of the design is that inference happens locally, not against a remote service.

**This is a starting point, not a ceiling.** Every external dependency — the generation model, the embedding model, the vector store, the graph store — sits behind its own interface (`backend/*/interfaces/`), with exactly one file per dependency (`providers/`) allowed to know which concrete implementation is in use. Moving to hosted infrastructure is therefore a matter of implementing that same interface against a different provider, not a rewrite of the retrieval, citation, or generation logic:

- **Generation** — swap `OllamaProvider` for an equivalent provider backed by OpenAI, Anthropic, Azure OpenAI, or Amazon Bedrock, or for a self-hosted inference server such as vLLM, TGI, or Triton.
- **Embeddings** — swap the local `sentence-transformers` provider for a hosted embedding API.
- **Vector search / knowledge graph** — swap the local Qdrant/Neo4j containers for managed equivalents (e.g. Qdrant Cloud, Neo4j Aura).

Local-first is the default because it best serves this project's actual goals — privacy, reproducibility, and zero marginal inference cost — not because the architecture is limited to it.

## License

Licensed under the [MIT License](LICENSE).
