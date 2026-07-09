"""Qdrant-based implementation of `VectorStore`.

This is the only file in the application permitted to import
`qdrant_client`.
"""

import logging
from collections.abc import Sequence
from uuid import UUID

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.search.exceptions import VectorStoreError
from backend.search.interfaces.vector_store import VectorStore
from backend.search.models import (
    CollectionInfo,
    DistanceMetric,
    EqualityFilter,
    IndexedField,
    PayloadFieldType,
    SearchResult,
    VectorPoint,
)

logger = logging.getLogger(__name__)

_DISTANCE_MAP = {
    DistanceMetric.COSINE: models.Distance.COSINE,
    DistanceMetric.DOT: models.Distance.DOT,
    DistanceMetric.EUCLIDEAN: models.Distance.EUCLID,
}
_REVERSE_DISTANCE_MAP = {value: key for key, value in _DISTANCE_MAP.items()}

_PAYLOAD_SCHEMA_MAP = {
    PayloadFieldType.KEYWORD: models.PayloadSchemaType.KEYWORD,
    PayloadFieldType.INTEGER: models.PayloadSchemaType.INTEGER,
}


class QdrantProvider(VectorStore):
    """Vector store backed by a Qdrant instance."""

    def __init__(self, url: str, timeout_seconds: int = 10) -> None:
        """Connect to a Qdrant instance.

        Args:
            url: Qdrant HTTP API URL (e.g. "http://localhost:6333").
            timeout_seconds: Request timeout for the underlying client.
        """
        self._client = QdrantClient(url=url, timeout=timeout_seconds)

    def collection_exists(self, collection: str) -> bool:
        try:
            return self._client.collection_exists(collection)
        except UnexpectedResponse as exc:
            raise VectorStoreError(reason=str(exc)) from exc

    def ensure_collection(
        self,
        collection: str,
        dimension: int,
        distance: DistanceMetric,
        indexed_fields: Sequence[IndexedField],
    ) -> None:
        try:
            if not self._client.collection_exists(collection):
                self._client.create_collection(
                    collection_name=collection,
                    vectors_config=models.VectorParams(
                        size=dimension, distance=_DISTANCE_MAP[distance]
                    ),
                )
                logger.info(
                    "collection created",
                    extra={"collection": collection, "dimension": dimension},
                )
            for field in indexed_fields:
                self._client.create_payload_index(
                    collection_name=collection,
                    field_name=field.name,
                    field_schema=_PAYLOAD_SCHEMA_MAP[field.field_type],
                )
        except UnexpectedResponse as exc:
            raise VectorStoreError(reason=str(exc)) from exc

    def collection_info(self, collection: str) -> CollectionInfo | None:
        try:
            if not self._client.collection_exists(collection):
                return None
            info = self._client.get_collection(collection)
        except UnexpectedResponse as exc:
            raise VectorStoreError(reason=str(exc)) from exc

        vectors_config = info.config.params.vectors
        if not isinstance(vectors_config, models.VectorParams):
            raise VectorStoreError(
                reason=f"collection '{collection}' uses an unsupported multi-vector configuration"
            )
        return CollectionInfo(
            name=collection,
            dimension=vectors_config.size,
            distance=_REVERSE_DISTANCE_MAP[vectors_config.distance],
            point_count=info.points_count or 0,
        )

    def upsert(self, collection: str, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        try:
            self._client.upsert(
                collection_name=collection,
                points=[
                    models.PointStruct(id=str(point.id), vector=point.vector, payload=point.payload)
                    for point in points
                ],
                wait=True,
            )
        except UnexpectedResponse as exc:
            raise VectorStoreError(reason=str(exc)) from exc

    def count(self, collection: str, filters: Sequence[EqualityFilter] = ()) -> int:
        try:
            result = self._client.count(
                collection_name=collection,
                count_filter=_build_filter(filters),
                exact=True,
            )
        except UnexpectedResponse as exc:
            raise VectorStoreError(reason=str(exc)) from exc
        return result.count

    def retrieve(self, collection: str, point_ids: Sequence[UUID]) -> list[VectorPoint]:
        if not point_ids:
            return []
        try:
            records = self._client.retrieve(
                collection_name=collection,
                ids=[str(point_id) for point_id in point_ids],
                with_payload=True,
                with_vectors=True,
            )
        except UnexpectedResponse as exc:
            raise VectorStoreError(reason=str(exc)) from exc
        return [
            VectorPoint(
                id=UUID(str(record.id)),
                vector=_as_dense_vector(record.vector),
                payload=dict(record.payload or {}),
            )
            for record in records
        ]

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
            raise VectorStoreError(reason=str(exc)) from exc
        return [
            SearchResult(
                id=UUID(str(point.id)), score=point.score, payload=dict(point.payload or {})
            )
            for point in response.points
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
        raise VectorStoreError(reason="retrieved point has an unsupported (non-dense) vector shape")
    return [float(item) for item in vector]
