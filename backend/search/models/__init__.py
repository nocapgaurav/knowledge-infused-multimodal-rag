"""Search index infrastructure's own data models -- not part of `backend.domain`
for the same reason Module 6's `EmbeddingArtifact` isn't: this is versioned,
backend-dependent infrastructure output, not a permanent fact about a paper.
"""

from backend.search.models.manifest import IndexManifest
from backend.search.models.vector_point import (
    CollectionInfo,
    DistanceMetric,
    EqualityFilter,
    IndexedField,
    PayloadFieldType,
    SearchResult,
    VectorPoint,
)

__all__ = [
    "CollectionInfo",
    "DistanceMetric",
    "EqualityFilter",
    "IndexManifest",
    "IndexedField",
    "PayloadFieldType",
    "SearchResult",
    "VectorPoint",
]
