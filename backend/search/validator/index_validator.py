"""Post-indexing verification against the real vector store.

Distinct from staleness detection (checked *before* indexing, to decide
whether to skip it): this runs *after* indexing, to confirm it actually
succeeded. Every check queries the real `VectorStore`, not inferred state.
"""

import logging

from backend.domain import PaperId
from backend.search.exceptions import (
    CollectionMissingError,
    DimensionMismatchError,
    IndexedCountMismatchError,
    PayloadIntegrityError,
)
from backend.search.interfaces.vector_store import VectorStore
from backend.search.models import EqualityFilter, VectorPoint

logger = logging.getLogger(__name__)

_REQUIRED_PAYLOAD_FIELDS = (
    "knowledge_unit_id",
    "document_id",
    "modality",
    "embedding_target",
    "reading_order",
)


class IndexValidator:
    """Verifies that indexing actually succeeded, against the real vector store."""

    def __init__(self, vector_store: VectorStore) -> None:
        """Initialize the validator.

        Args:
            vector_store: The vector store to verify against.
        """
        self._vector_store = vector_store

    def verify(
        self,
        document_id: PaperId,
        collection: str,
        expected_dimension: int,
        expected_count: int,
        indexed_points: list[VectorPoint],
    ) -> None:
        """Verify a completed indexing run.

        Every indexed point for this document is checked, not a sample --
        the unit of work here is bounded by one document's chunk count,
        not corpus size, so checking everything is cheap.

        Args:
            document_id: Identifier of the document that was indexed.
            collection: Name of the collection it was indexed into.
            expected_dimension: Vector dimension the embeddings were generated at.
            expected_count: Number of vectors expected to be present for
                this document (accounting for any recorded failures).
            indexed_points: The points that were actually upserted, for a
                payload-integrity check.

        Raises:
            CollectionMissingError: The collection does not exist.
            DimensionMismatchError: The collection's configured dimension
                doesn't match what was expected.
            IndexedCountMismatchError: The indexed point count doesn't
                match what was expected.
            PayloadIntegrityError: An indexed point's payload is missing
                required fields or doesn't match what was indexed.
        """
        info = self._vector_store.collection_info(collection)
        if info is None:
            raise CollectionMissingError(document_id=document_id, collection=collection)
        if info.dimension != expected_dimension:
            raise DimensionMismatchError(
                document_id=document_id, expected=expected_dimension, actual=info.dimension
            )

        actual_count = self._vector_store.count(
            collection, filters=[EqualityFilter(field="document_id", value=str(document_id))]
        )
        if actual_count != expected_count:
            raise IndexedCountMismatchError(
                document_id=document_id, expected=expected_count, actual=actual_count
            )

        self._verify_payload_integrity(document_id, collection, indexed_points)

    def _verify_payload_integrity(
        self, document_id: PaperId, collection: str, indexed_points: list[VectorPoint]
    ) -> None:
        if not indexed_points:
            return
        point_ids = [point.id for point in indexed_points]
        retrieved = {
            point.id: point for point in self._vector_store.retrieve(collection, point_ids)
        }

        for point in indexed_points:
            retrieved_point = retrieved.get(point.id)
            if retrieved_point is None:
                raise PayloadIntegrityError(
                    document_id=document_id,
                    reason=f"point {point.id} was not found after indexing",
                )
            missing = [
                field for field in _REQUIRED_PAYLOAD_FIELDS if field not in retrieved_point.payload
            ]
            if missing:
                raise PayloadIntegrityError(
                    document_id=document_id,
                    reason=f"point {point.id} is missing required payload fields: {missing}",
                )
            if retrieved_point.payload.get("knowledge_unit_id") != point.payload.get(
                "knowledge_unit_id"
            ):
                raise PayloadIntegrityError(
                    document_id=document_id,
                    reason=f"point {point.id} payload does not match what was indexed",
                )
