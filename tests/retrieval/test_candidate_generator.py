"""Tests for Phase 1: candidate generation, using fakes -- no real services needed."""

from collections.abc import Sequence
from uuid import uuid4

import pytest

from backend.embeddings.interfaces.embedding_provider import EmbeddingProvider
from backend.retrieval.candidate.candidate_generator import CandidateGenerator, normalize_query
from backend.retrieval.exceptions import QueryEmbeddingError
from backend.retrieval.interfaces.vector_retriever import VectorRetriever
from backend.retrieval.models import DiscoveryMethod
from backend.search.models import EqualityFilter, SearchResult, VectorPoint


class _FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.embedded_texts: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def model_version(self) -> str:
        return "fake-1"

    @property
    def embedding_dimension(self) -> int:
        return 4

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _FakeVectorRetriever(VectorRetriever):
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.last_filters: Sequence[EqualityFilter] = ()

    def search(
        self,
        collection: str,
        query_vector: Sequence[float],
        limit: int,
        filters: Sequence[EqualityFilter] = (),
    ) -> list[SearchResult]:
        self.last_filters = filters
        return self._results[:limit]

    def retrieve_by_ids(self, collection: str, ids: Sequence) -> list[VectorPoint]:
        raise NotImplementedError


def _search_result(**payload_overrides: object) -> SearchResult:
    payload = {
        "knowledge_unit_id": str(uuid4()),
        "document_id": str(uuid4()),
        "section_id": None,
        "modality": "text",
        "text": "some evidence text",
        "asset_uri": None,
        "reading_order": 0,
        "citation_count": 0,
    }
    payload.update(payload_overrides)
    return SearchResult(id=uuid4(), score=0.9, payload=payload)


def test_normalize_query_collapses_whitespace() -> None:
    assert normalize_query("  what   is\n\tthe result?  ") == "what is the result?"


def test_generate_returns_candidates_ordered_by_search_result() -> None:
    document_id = uuid4()
    results = [
        _search_result(document_id=str(document_id), text="first"),
        _search_result(document_id=str(document_id), text="second"),
    ]
    embedding_provider = _FakeEmbeddingProvider()
    vector_retriever = _FakeVectorRetriever(results)
    generator = CandidateGenerator(embedding_provider, vector_retriever, top_k=10)

    candidates = generator.generate(document_id, "what is the result?", "some-collection")

    assert [c.text for c in candidates] == ["first", "second"]
    assert all(c.discovery_method is DiscoveryMethod.DENSE_RETRIEVAL for c in candidates)
    assert all(c.dense_similarity == 0.9 for c in candidates)
    assert all(c.graph_path.depth == 0 for c in candidates)


def test_generate_embeds_the_normalized_query() -> None:
    embedding_provider = _FakeEmbeddingProvider()
    vector_retriever = _FakeVectorRetriever([])
    generator = CandidateGenerator(embedding_provider, vector_retriever, top_k=10)

    generator.generate(uuid4(), "  a   messy\nquery  ", "some-collection")

    assert embedding_provider.embedded_texts == ["a messy query"]


def test_generate_filters_by_document_id() -> None:
    document_id = uuid4()
    embedding_provider = _FakeEmbeddingProvider()
    vector_retriever = _FakeVectorRetriever([])
    generator = CandidateGenerator(embedding_provider, vector_retriever, top_k=10)

    generator.generate(document_id, "query", "some-collection")

    assert list(vector_retriever.last_filters) == [
        EqualityFilter(field="document_id", value=str(document_id))
    ]


def test_generate_raises_on_empty_query() -> None:
    generator = CandidateGenerator(_FakeEmbeddingProvider(), _FakeVectorRetriever([]), top_k=10)

    with pytest.raises(QueryEmbeddingError):
        generator.generate(uuid4(), "    ", "some-collection")


def test_candidate_section_id_is_none_when_payload_has_no_section() -> None:
    document_id = uuid4()
    result = _search_result(document_id=str(document_id), section_id=None)
    generator = CandidateGenerator(
        _FakeEmbeddingProvider(), _FakeVectorRetriever([result]), top_k=10
    )

    candidates = generator.generate(document_id, "query", "some-collection")

    assert candidates[0].section_id is None


def test_normalize_query_appends_author_intent_hints() -> None:
    from backend.retrieval.candidate.candidate_generator import normalize_query

    assert normalize_query("Who wrote this paper?") == (
        "Who wrote this paper? | authors affiliations title page"
    )
    assert normalize_query("Which university conducted this research?") == (
        "Which university conducted this research? | authors affiliations title page"
    )


def test_normalize_query_leaves_plain_questions_untouched() -> None:
    from backend.retrieval.candidate.candidate_generator import normalize_query

    assert normalize_query("What is Figure 2?") == "What is Figure 2?"


def test_normalize_query_skips_hint_already_present_in_query() -> None:
    from backend.retrieval.candidate.candidate_generator import normalize_query

    assert normalize_query("Who are the authors?") == "Who are the authors?"
