"""Qdrant-based implementation of `VectorRetriever`.

This is the only file in this module permitted to import `qdrant_client`.
Owns its own client connection rather than wrapping Module 7's
`QdrantProvider` -- composing over a write-capable object would let a
future change reach its `upsert`/`ensure_collection` methods through this
module, defeating the point of a narrower, read-only interface.
"""

from collections.abc import Sequence
from uuid import UUID

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.retrieval.exceptions import VectorRetrieverError
from backend.retrieval.interfaces.vector_retriever import VectorRetriever
from backend.search.models import EqualityFilter, SearchResult, VectorPoint


class QdrantRetriever(VectorRetriever):
    """Read-only vector retrieval backed by a Qdrant instance."""

    def __init__(self, url: str, timeout_seconds: int = 10) -> None:
        """Connect to a Qdrant instance.

        Args:
            url: Qdrant HTTP API URL (e.g. "http://localhost:6333").
            timeout_seconds: Request timeout for the underlying client.
        """
        self._client = QdrantClient(url=url, timeout=timeout_seconds)

    def search(
        self,
        collection: str,
        query_vector: Sequence[float],
        limit: int,
        filters: Sequence[EqualityFilter] = (),
    ) -> list[SearchResult]:
        try:
            response = self._client.query_points(
                collection_name=collection,
                query=list(query_vector),
                limit=limit,
                query_filter=_build_filter(filters),
                with_payload=True,
            )
        except UnexpectedResponse as exc:
            raise VectorRetrieverError(reason=str(exc)) from exc
        return [
            SearchResult(
                id=UUID(str(point.id)), score=point.score, payload=dict(point.payload or {})
            )
            for point in response.points
        ]

    def retrieve_by_ids(self, collection: str, ids: Sequence[UUID]) -> list[VectorPoint]:
        if not ids:
            return []
        try:
            records = self._client.retrieve(
                collection_name=collection,
                ids=[str(point_id) for point_id in ids],
                with_payload=True,
                with_vectors=True,
            )
        except UnexpectedResponse as exc:
            raise VectorRetrieverError(reason=str(exc)) from exc
        return [
            VectorPoint(
                id=UUID(str(record.id)),
                vector=_as_dense_vector(record.vector),
                payload=dict(record.payload or {}),
            )
            for record in records
        ]


def _build_filter(filters: Sequence[EqualityFilter]) -> models.Filter | None:
    if not filters:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(key=f.field, match=models.MatchValue(value=f.value))
            for f in filters
        ]
    )


def _as_dense_vector(vector: object) -> list[float]:
    """Narrow a retrieved vector to a plain dense vector.

    This application only ever creates single-dense-vector collections, so
    a retrieved point's vector is always a flat `list[float]` in practice --
    but Qdrant's own type also allows sparse, multi-vector, and named-vector
    shapes, which this application never produces or expects.
    """
    if not isinstance(vector, list) or not all(isinstance(item, (int, float)) for item in vector):
        raise VectorRetrieverError(
            reason="retrieved point has an unsupported (non-dense) vector shape"
        )
    return [float(item) for item in vector]
