"""Integration tests for post-indexing verification, against a real Qdrant instance."""

from collections.abc import Iterator
from uuid import uuid4

import pytest

from backend.domain import PaperId
from backend.search.exceptions import (
    CollectionMissingError,
    DimensionMismatchError,
    IndexedCountMismatchError,
    PayloadIntegrityError,
)
from backend.search.models import DistanceMetric, VectorPoint
from backend.search.providers.qdrant_provider import QdrantProvider
from backend.search.validator.index_validator import IndexValidator

QDRANT_URL = "http://localhost:6333"


@pytest.fixture
def provider() -> QdrantProvider:
    return QdrantProvider(url=QDRANT_URL)


@pytest.fixture
def collection(provider: QdrantProvider) -> Iterator[str]:
    name = f"kimrag_test_{uuid4().hex}"
    yield name
    if provider.collection_exists(name):
        provider._client.delete_collection(name)  # test-only cleanup


def _valid_point() -> VectorPoint:
    return VectorPoint(
        id=uuid4(),
        vector=[0.1, 0.2, 0.3, 0.4],
        payload={
            "knowledge_unit_id": str(uuid4()),
            "document_id": "doc-a",
            "modality": "text",
            "embedding_target": "text",
            "reading_order": 0,
        },
    )


def test_verify_passes_for_a_correctly_indexed_document(
    provider: QdrantProvider, collection: str
) -> None:
    provider.ensure_collection(
        collection, dimension=4, distance=DistanceMetric.COSINE, indexed_fields=()
    )
    document_id = PaperId(uuid4())
    point = _valid_point()
    point.payload["document_id"] = str(document_id)
    provider.upsert(collection, [point])

    validator = IndexValidator(provider)
    validator.verify(
        document_id=document_id,
        collection=collection,
        expected_dimension=4,
        expected_count=1,
        indexed_points=[point],
    )  # should not raise


def test_verify_raises_when_collection_missing(provider: QdrantProvider) -> None:
    validator = IndexValidator(provider)

    with pytest.raises(CollectionMissingError):
        validator.verify(
            document_id=PaperId(uuid4()),
            collection=f"does_not_exist_{uuid4().hex}",
            expected_dimension=4,
            expected_count=1,
            indexed_points=[],
        )


def test_verify_raises_on_dimension_mismatch(provider: QdrantProvider, collection: str) -> None:
    provider.ensure_collection(
        collection, dimension=4, distance=DistanceMetric.COSINE, indexed_fields=()
    )
    validator = IndexValidator(provider)

    with pytest.raises(DimensionMismatchError):
        validator.verify(
            document_id=PaperId(uuid4()),
            collection=collection,
            expected_dimension=1024,
            expected_count=0,
            indexed_points=[],
        )


def test_verify_raises_on_count_mismatch(provider: QdrantProvider, collection: str) -> None:
    provider.ensure_collection(
        collection, dimension=4, distance=DistanceMetric.COSINE, indexed_fields=()
    )
    document_id = PaperId(uuid4())
    point = _valid_point()
    point.payload["document_id"] = str(document_id)
    provider.upsert(collection, [point])
    validator = IndexValidator(provider)

    with pytest.raises(IndexedCountMismatchError):
        validator.verify(
            document_id=document_id,
            collection=collection,
            expected_dimension=4,
            expected_count=5,  # only 1 was actually indexed
            indexed_points=[point],
        )


def test_verify_raises_when_payload_missing_required_fields(
    provider: QdrantProvider, collection: str
) -> None:
    provider.ensure_collection(
        collection, dimension=4, distance=DistanceMetric.COSINE, indexed_fields=()
    )
    document_id = PaperId(uuid4())
    incomplete_point = VectorPoint(
        id=uuid4(), vector=[0.1, 0.2, 0.3, 0.4], payload={"document_id": str(document_id)}
    )
    provider.upsert(collection, [incomplete_point])
    validator = IndexValidator(provider)

    with pytest.raises(PayloadIntegrityError):
        validator.verify(
            document_id=document_id,
            collection=collection,
            expected_dimension=4,
            expected_count=1,
            indexed_points=[incomplete_point],
        )
