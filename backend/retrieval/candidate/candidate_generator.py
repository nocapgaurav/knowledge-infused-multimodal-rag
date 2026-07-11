"""Phase 1: Candidate Generation.

Normalizes the query, embeds it with the same model that produced the
indexed vectors, and retrieves the top-K most similar knowledge units for
one document. Collection selection (which collection to search) and
metadata filtering beyond `document_id` are the caller's concern -- this
class only knows how to turn a query and a collection name into candidates.
"""

import logging
import re
from uuid import UUID

from backend.domain import ChunkId, ChunkModality, PaperId, SectionId
from backend.domain.value_objects import BoundingBox
from backend.embeddings.interfaces.embedding_provider import EmbeddingProvider
from backend.retrieval.exceptions import QueryEmbeddingError
from backend.retrieval.interfaces.vector_retriever import VectorRetriever
from backend.retrieval.models import DiscoveryMethod, RetrievalCandidate
from backend.search.models import EqualityFilter, SearchResult

logger = logging.getLogger(__name__)


_INTENT_EXPANSIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"who\s+(?:wrote|authored|developed|created|conducted)|"
            r"which\s+(?:university|institution|lab|group)|"
            r"\bauthors?\b|\baffiliations?\b",
            re.IGNORECASE,
        ),
        "authors affiliations title page",
    ),
    (
        re.compile(r"\bkey\s?words?\b|\bindex\s+terms\b", re.IGNORECASE),
        "keywords index terms",
    ),
    (
        re.compile(
            r"\bmethodolog(?:y|ies)\b|\bapproach\b|\bhow\s+does\s+it\s+work\b",
            re.IGNORECASE,
        ),
        "method methodology proposed",
    ),
)
"""Deterministic intent hints appended to the embedded query. Each entry
maps researcher phrasings to the canonical vocabulary the paper's own
structural chunks carry (their `retrieval_context`), closing the gap
between how people ask ("who wrote this?", "which university?") and how
the target chunk is labeled ("Authors and affiliations (title page)").
A fixed rule table, not a model -- the same question always expands the
same way, and the user's own words are always kept first."""


def normalize_query(query: str) -> str:
    """Normalize a raw user question into a canonical form for embedding.

    Collapses all whitespace (including newlines) to single spaces and
    strips leading/trailing whitespace. Deliberately does not lowercase or
    strip punctuation -- the embedding model's own tokenizer already
    handles case and punctuation, and altering them further is exactly the
    kind of unjustified transformation that could silently change meaning
    (e.g. "COVID-19" vs "covid 19").

    Deterministic intent hints are appended (never substituted) when the
    question matches a known researcher phrasing -- see
    `_INTENT_EXPANSIONS`.

    Args:
        query: The raw question text.

    Returns:
        The normalized query text, possibly with appended intent hints.
    """
    normalized = " ".join(query.split())
    if not normalized:
        return normalized
    hints = [
        hint
        for pattern, hint in _INTENT_EXPANSIONS
        if pattern.search(normalized) and hint.split()[0].lower() not in normalized.lower()
    ]
    if hints:
        return normalized + " | " + " ".join(hints)
    return normalized


class CandidateGenerator:
    """Generates Phase 1 retrieval candidates via dense vector search."""

    def __init__(
        self, embedding_provider: EmbeddingProvider, vector_retriever: VectorRetriever, top_k: int
    ) -> None:
        """Initialize the generator.

        Args:
            embedding_provider: Produces the query embedding. Must be the
                same model (and revision) that produced the indexed
                vectors, or similarity scores are meaningless.
            vector_retriever: Read-only access to the vector database.
            top_k: Maximum number of candidates to retrieve.
        """
        self._embedding_provider = embedding_provider
        self._vector_retriever = vector_retriever
        self._top_k = top_k

    def generate(
        self, document_id: PaperId, query: str, collection: str
    ) -> list[RetrievalCandidate]:
        """Generate candidates for a query, scoped to one document's collection.

        Args:
            document_id: Identifier of the document to search within.
            query: The raw user question.
            collection: Name of the collection this document's vectors are indexed in.

        Returns:
            Candidates ordered by descending dense similarity.

        Raises:
            QueryEmbeddingError: The query is empty after normalization,
                or embedding it failed.
            VectorRetrieverError: The search failed.
        """
        normalized = normalize_query(query)
        if not normalized:
            raise QueryEmbeddingError(reason="query is empty after normalization")

        try:
            vectors = self._embedding_provider.embed_texts([normalized])
        except Exception as exc:  # provider-specific exceptions vary by backend
            raise QueryEmbeddingError(reason=str(exc)) from exc

        results = self._vector_retriever.search(
            collection,
            vectors[0],
            limit=self._top_k,
            filters=[EqualityFilter(field="document_id", value=str(document_id))],
        )
        candidates = [_to_candidate(result) for result in results]
        logger.info(
            "candidates generated",
            extra={"document_id": str(document_id), "count": len(candidates)},
        )
        return candidates


def _to_candidate(result: SearchResult) -> RetrievalCandidate:
    payload = result.payload
    section_id_value = payload.get("section_id")
    return RetrievalCandidate(
        knowledge_unit_id=ChunkId(UUID(str(payload["knowledge_unit_id"]))),
        document_id=PaperId(UUID(str(payload["document_id"]))),
        section_id=SectionId(UUID(str(section_id_value))) if section_id_value else None,
        modality=ChunkModality(payload["modality"]),
        text=payload["text"],
        retrieval_context=payload.get("retrieval_context"),
        page_numbers=tuple(payload.get("page_numbers") or ()),
        bounding_boxes=tuple(
            BoundingBox.model_validate(box) for box in payload.get("bounding_boxes") or ()
        ),
        asset_uri=payload.get("asset_uri"),
        reading_order=payload["reading_order"],
        citation_count=payload.get("citation_count") or 0,
        dense_similarity=result.score,
        discovery_method=DiscoveryMethod.DENSE_RETRIEVAL,
    )
