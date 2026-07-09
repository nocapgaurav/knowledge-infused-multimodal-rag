"""Integration tests for the real Qdrant provider.

Runs against a real Qdrant instance (see `docker-compose.yml` -- `docker
compose up -d qdrant`), not a fake. Every test uses a unique,
randomly-named collection and cleans it up afterward, so tests don't
interfere with each other or leave state behind.
"""

from collections.abc import Iterator
from uuid import uuid4

import pytest

from backend.search.exceptions import VectorStoreError
from backend.search.models import (
    DistanceMetric,
    EqualityFilter,
    IndexedField,
    PayloadFieldType,
    VectorPoint,
)
from backend.search.providers.qdrant_provider import QdrantProvider

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


def _point(vector: list[float] | None = None, **payload: object) -> VectorPoint:
    return VectorPoint(id=uuid4(), vector=vector or [0.1, 0.2, 0.3, 0.4], payload=payload)


def test_collection_does_not_exist_before_creation(
    provider: QdrantProvider, collection: str
) -> None:
    assert provider.collection_exists(collection) is False


def test_ensure_collection_creates_it_with_correct_configuration(
    provider: QdrantProvider, collection: str
) -> None:
    provider.ensure_collection(
        collection, dimension=4, distance=DistanceMetric.COSINE, indexed_fields=()
    )

    assert provider.collection_exists(collection) is True
    info = provider.collection_info(collection)
    assert info is not None
    assert info.dimension == 4
    assert info.distance is DistanceMetric.COSINE
    assert info.point_count == 0


def test_ensure_collection_is_idempotent(provider: QdrantProvider, collection: str) -> None:
    provider.ensure_collection(
        collection, dimension=4, distance=DistanceMetric.COSINE, indexed_fields=()
    )
    provider.ensure_collection(
        collection, dimension=4, distance=DistanceMetric.COSINE, indexed_fields=()
    )  # no error

    assert provider.collection_exists(collection) is True


def test_ensure_collection_creates_payload_indexes(
    provider: QdrantProvider, collection: str
) -> None:
    fields = (
        IndexedField(name="document_id", field_type=PayloadFieldType.KEYWORD),
        IndexedField(name="reading_order", field_type=PayloadFieldType.INTEGER),
    )

    provider.ensure_collection(
        collection, dimension=4, distance=DistanceMetric.COSINE, indexed_fields=fields
    )

    point = _point(document_id="doc-a", reading_order=0)
    provider.upsert(collection, [point])
    assert (
        provider.count(collection, filters=[EqualityFilter(field="document_id", value="doc-a")])
        == 1
    )


def test_collection_info_returns_none_for_nonexistent_collection(provider: QdrantProvider) -> None:
    assert provider.collection_info(f"does_not_exist_{uuid4().hex}") is None


def test_upsert_and_retrieve_round_trips(provider: QdrantProvider, collection: str) -> None:
    provider.ensure_collection(
        collection, dimension=4, distance=DistanceMetric.COSINE, indexed_fields=()
    )
    # Qdrant normalizes stored vectors for COSINE collections, so a
    # round-trip only preserves the exact values for an already-unit-norm
    # input -- which is what real embeddings (Module 6 normalizes BGE-M3's
    # output) always are in practice.
    point = _point(vector=[1.0, 0.0, 0.0, 0.0], document_id="doc-a", text="hello world")

    provider.upsert(collection, [point])
    retrieved = provider.retrieve(collection, [point.id])

    assert len(retrieved) == 1
    assert retrieved[0].id == point.id
    assert retrieved[0].vector == point.vector
    assert retrieved[0].payload["text"] == "hello world"


def test_upsert_is_idempotent(provider: QdrantProvider, collection: str) -> None:
    provider.ensure_collection(
        collection, dimension=4, distance=DistanceMetric.COSINE, indexed_fields=()
    )
    point = _point()

    provider.upsert(collection, [point])
    provider.upsert(collection, [point])  # re-upsert the same point id

    assert provider.count(collection) == 1


def test_count_with_and_without_filters(provider: QdrantProvider, collection: str) -> None:
    provider.ensure_collection(
        collection,
        dimension=4,
        distance=DistanceMetric.COSINE,
        indexed_fields=(IndexedField(name="document_id", field_type=PayloadFieldType.KEYWORD),),
    )
    provider.upsert(
        collection,
        [_point(document_id="doc-a"), _point(document_id="doc-a"), _point(document_id="doc-b")],
    )

    assert provider.count(collection) == 3
    assert (
        provider.count(collection, filters=[EqualityFilter(field="document_id", value="doc-a")])
        == 2
    )
    assert (
        provider.count(collection, filters=[EqualityFilter(field="document_id", value="doc-b")])
        == 1
    )


def test_retrieve_omits_missing_ids_without_error(
    provider: QdrantProvider, collection: str
) -> None:
    provider.ensure_collection(
        collection, dimension=4, distance=DistanceMetric.COSINE, indexed_fields=()
    )
    point = _point()
    provider.upsert(collection, [point])

    retrieved = provider.retrieve(collection, [point.id, uuid4()])

    assert len(retrieved) == 1
    assert retrieved[0].id == point.id


def test_search_returns_similar_points_ordered_by_score(
    provider: QdrantProvider, collection: str
) -> None:
    provider.ensure_collection(
        collection, dimension=4, distance=DistanceMetric.COSINE, indexed_fields=()
    )
    close_point = _point(vector=[1.0, 0.0, 0.0, 0.0], label="close")
    far_point = _point(vector=[0.0, 1.0, 0.0, 0.0], label="far")
    provider.upsert(collection, [close_point, far_point])

    results = provider.search(collection, query_vector=[1.0, 0.0, 0.0, 0.0], limit=2)

    assert results[0].id == close_point.id
    assert results[0].payload["label"] == "close"
    assert results[0].score >= results[1].score


def test_search_respects_filters(provider: QdrantProvider, collection: str) -> None:
    provider.ensure_collection(
        collection,
        dimension=4,
        distance=DistanceMetric.COSINE,
        indexed_fields=(IndexedField(name="document_id", field_type=PayloadFieldType.KEYWORD),),
    )
    provider.upsert(
        collection,
        [_point(document_id="doc-a"), _point(document_id="doc-b")],
    )

    results = provider.search(
        collection,
        query_vector=[0.1, 0.2, 0.3, 0.4],
        limit=10,
        filters=[EqualityFilter(field="document_id", value="doc-a")],
    )

    assert all(result.payload["document_id"] == "doc-a" for result in results)


def test_dimension_mismatch_raises_vector_store_error(
    provider: QdrantProvider, collection: str
) -> None:
    provider.ensure_collection(
        collection, dimension=4, distance=DistanceMetric.COSINE, indexed_fields=()
    )

    with pytest.raises(VectorStoreError):
        provider.upsert(collection, [_point(vector=[0.1, 0.2])])


def test_querying_a_nonexistent_collection_raises_vector_store_error(
    provider: QdrantProvider,
) -> None:
    with pytest.raises(VectorStoreError):
        provider.count(f"does_not_exist_{uuid4().hex}")
