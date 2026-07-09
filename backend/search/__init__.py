"""Search index infrastructure: transforms embedding artifacts into a
production-grade, searchable vector index.

Consumes only `backend.embeddings`' persisted artifacts (plus, for payload
enrichment, `backend.chunking`'s knowledge representation artifacts) --
never regenerates knowledge units or embeddings, never parses PDFs.
Business logic (planner, payload builder, service, validator) depends only
on `VectorStore`, never on a specific vector database;
`providers/qdrant_provider.py` is the only concrete implementation and the
only file permitted to import `qdrant_client`.
"""
