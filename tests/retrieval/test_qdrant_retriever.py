"""Integration tests for the real Qdrant retriever.

Runs against a real Qdrant instance (see `docker-compose.yml` -- `docker
compose up -d qdrant`), not a fake. Seeds data via Module 7's own
`QdrantProvider` (the write side), then reads it back through
`QdrantRetriever` (this module's read-only side) -- proving the narrower
interface reads the exact same data the write-capable one produces.
"""

from collections.abc import Iterator
from uuid import uuid4

import pytest

from backend.retrieval.providers.qdrant_retriever import QdrantRetriever
from backend.search.models import DistanceMetric, EqualityFilter, VectorPoint
from backend.search.providers.qdrant_provider import QdrantProvider

QDRANT_URL = "http://localhost:6333"


@pytest.fixture
def writer() -> QdrantProvider:
    return QdrantProvider(url=QDRANT_URL)


@pytest.fixture
def retriever() -> QdrantRetriever:
    return QdrantRetriever(url=QDRANT_URL)


@pytest.fixture
def collection(writer: QdrantProvider) -> Iterator[str]:
    name = f"kimrag_test_{uuid4().hex}"
    yield name
    if writer.collection_exists(name):
        writer._client.delete_collection(name)  # test-only cleanup


def _point(vector: list[float] | None = None, **payload: object) -> VectorPoint:
    return VectorPoint(id=uuid4(), vector=vector or [0.1, 0.2, 0.3, 0.4], payload=payload)


def test_search_finds_the_closest_point(
    writer: QdrantProvider, retriever: QdrantRetriever, collection: str
) -> None:
    writer.ensure_collection(
        collection, dimension=4, distance=DistanceMetric.COSINE, indexed_fields=()
    )
    close_point = _point(vector=[1.0, 0.0, 0.0, 0.0], text="close")
    far_point = _point(vector=[0.0, 1.0, 0.0, 0.0], text="far")
    writer.upsert(collection, [close_point, far_point])

    results = retriever.search(collection, query_vector=[1.0, 0.0, 0.0, 0.0], limit=1)

    assert len(results) == 1
    assert results[0].id == close_point.id
    assert results[0].payload["text"] == "close"


def test_search_respects_filters(
    writer: QdrantProvider, retriever: QdrantRetriever, collection: str
) -> None:
    writer.ensure_collection(
        collection, dimension=4, distance=DistanceMetric.COSINE, indexed_fields=()
    )
    writer.upsert(collection, [_point(document_id="doc-a"), _point(document_id="doc-b")])

    results = retriever.search(
        collection,
        query_vector=[0.1, 0.2, 0.3, 0.4],
        limit=10,
        filters=[EqualityFilter(field="document_id", value="doc-a")],
    )

    assert all(result.payload["document_id"] == "doc-a" for result in results)


def test_retrieve_by_ids_returns_full_payload(
    writer: QdrantProvider, retriever: QdrantRetriever, collection: str
) -> None:
    writer.ensure_collection(
        collection, dimension=4, distance=DistanceMetric.COSINE, indexed_fields=()
    )
    point = _point(text="hello evidence", modality="text")
    writer.upsert(collection, [point])

    retrieved = retriever.retrieve_by_ids(collection, [point.id])

    assert len(retrieved) == 1
    assert retrieved[0].id == point.id
    assert retrieved[0].payload["text"] == "hello evidence"


def test_retrieve_by_ids_omits_missing_ids_without_error(
    writer: QdrantProvider, retriever: QdrantRetriever, collection: str
) -> None:
    writer.ensure_collection(
        collection, dimension=4, distance=DistanceMetric.COSINE, indexed_fields=()
    )
    point = _point()
    writer.upsert(collection, [point])

    retrieved = retriever.retrieve_by_ids(collection, [point.id, uuid4()])

    assert len(retrieved) == 1
    assert retrieved[0].id == point.id


def test_retrieve_by_ids_with_empty_list_returns_empty(
    retriever: QdrantRetriever, collection: str
) -> None:
    assert retriever.retrieve_by_ids(collection, []) == []
