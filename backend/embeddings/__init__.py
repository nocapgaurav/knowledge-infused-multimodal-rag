"""Embedding infrastructure: turns the knowledge representation into a
reusable, versioned, production-ready embedding layer.

Consumes only `backend.domain.Chunk` objects read from Module 5's
persisted output. Never depends on PDFs, Docling, OCR, or parser
internals. Business logic (planner, service, validator) depends only on
`EmbeddingProvider`/`ImageEmbeddingProvider` -- never on a specific model
or library; `providers/sentence_transformers_provider.py` is the only
concrete implementation and the only file permitted to import
`sentence_transformers`.
"""
