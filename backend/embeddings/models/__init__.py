"""Embedding infrastructure's own data models -- deliberately not part of
`backend.domain`. See `artifact.py` for why.
"""

from backend.embeddings.models.artifact import EmbeddingArtifact, EmbeddingId, EmbeddingTarget
from backend.embeddings.models.manifest import EmbeddingManifest

__all__ = [
    "EmbeddingArtifact",
    "EmbeddingId",
    "EmbeddingManifest",
    "EmbeddingTarget",
]
