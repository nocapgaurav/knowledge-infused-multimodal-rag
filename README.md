# Knowledge-Infused Multimodal RAG

A knowledge-infused multimodal question-answering system for scientific research papers: upload a PDF, ask questions, and get answers grounded in the document's own text, tables, and figures, with a knowledge-graph-backed "related content" view.

The system has two parts:
- **`backend/`** — a FastAPI service (Python 3.12) that parses, embeds, indexes, and answers questions over uploaded papers, backed by Qdrant (vector search), Neo4j (knowledge graph), and a local Ollama model (generation).
- **`frontend/`** — a Next.js 15 research workspace UI that talks to the backend.

## Prerequisites

- Python **3.12** (not 3.13+, not 3.11 — `pyproject.toml` pins `>=3.12,<3.13`). On macOS with Homebrew: `brew install python@3.12`.
- Node.js 20+ and npm, for the frontend.
- Docker, for Qdrant and Neo4j.
- [Ollama](https://ollama.com), running locally with a chat-capable model pulled, e.g.:
  ```bash
  ollama pull qwen2.5:7b-instruct
  ```
  (The default model name is configurable — see `backend/config/settings.py`'s `generation_model` / `.env`.)

## One-time setup

From the repository root:

```bash
# Backend: create a venv with the pinned Python version and install the package + dev tools
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
deactivate

# Frontend: install JS dependencies
cd frontend
npm install
cd ..
```

This creates `.venv/` (git-ignored — re-run this step after a clean clone or if the venv is ever deleted) and `frontend/node_modules/`.

## Starting the project (every time)

Run these in order, each in its own terminal (or backgrounded):

**1. Start Qdrant and Neo4j:**
```bash
docker compose up -d
```

**2. Start Ollama**, if it isn't already running as a background service:
```bash
ollama serve
```

**3. Start the backend** (from the repo root, with the venv activated):
```bash
source .venv/bin/activate
uvicorn backend.api.app:create_app --factory --host 127.0.0.1 --port 8000
```
Equivalently, once installed, the console script `kimrag-api` does the same thing using the host/port from settings.

**4. Start the frontend** (from `frontend/`, in a separate terminal):
```bash
cd frontend
npm run dev
```

**5. Open the app**: [http://localhost:3000](http://localhost:3000). The status indicator in the bottom-left corner should read **"Backend connected"** within a couple of seconds. If it reads "Backend unavailable," see Troubleshooting below.

## Verifying the backend directly

```bash
curl http://localhost:8000/health          # {"status":"ok"}
curl http://localhost:8000/docs            # interactive OpenAPI docs
```

## Configuration

Both halves read configuration from environment variables, with working defaults for local development — copying the `.env.example` files is optional unless you need to change something:

- `backend/.env` (see `.env.example` at the repo root): host/port, Qdrant/Neo4j/Ollama URLs, CORS-allowed origins, log level.
- `frontend/.env.local` (see `frontend/.env.example`): `NEXT_PUBLIC_API_BASE_URL`, defaulting to `http://localhost:8000`.

## Stopping / cleaning up

```bash
# Frontend and backend: Ctrl+C in their respective terminals, or:
pkill -f "next dev"; pkill -f "next start"; pkill -f "uvicorn backend.api.app"

# Docker services (add -v to also wipe Qdrant/Neo4j data):
docker compose down
```

## Troubleshooting

**Frontend shows "Backend unavailable" / uploads fail with "Could not reach the server."**
1. Confirm the backend process is actually running and healthy: `curl http://localhost:8000/health`.
2. If that fails, the backend isn't running — see step 3 under Starting the project. The most common cause is an incompatible or missing Python environment: `which uvicorn` should point inside `.venv/bin/`; if it doesn't, the venv either doesn't exist yet (run One-time setup) or isn't activated (`source .venv/bin/activate`).
3. If `curl http://localhost:8000/health` succeeds but the browser still can't reach it, check the browser console for a CORS error and confirm `backend/config/settings.py`'s `cors_allowed_origins` includes the frontend's actual origin (default `http://localhost:3000`).

**`pip install -e ".[dev]"` fails or `ModuleNotFoundError` on startup.**
Check `python --version` inside the activated venv — it must be 3.12.x. If your system's default `python3`/`python3.12` resolves to a different version, create the venv explicitly with the 3.12 binary's full path, e.g. `/opt/homebrew/bin/python3.12 -m venv .venv`.

**Uploads succeed but questions never get answered.**
Confirm Ollama is running and has the configured model pulled: `curl http://localhost:11434/api/tags`.
