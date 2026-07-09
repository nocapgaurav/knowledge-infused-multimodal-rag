"""VectorRetriever: the read-only port candidate generation and expansion hydration depend on.

Deliberately narrower than Module 7's `VectorStore`: this module must
never write to Qdrant, and depending on the full `VectorStore` interface
(which includes `upsert`/`ensure_collection`) would make that a matter of
convention rather than a structural guarantee. Every method here is a read.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from uuid import UUID

from backend.search.models import EqualityFilter, SearchResult, VectorPoint


class VectorRetriever(ABC):
    """Read-only access to a vector database's similarity search and point lookup."""

    @abstractmethod
    def search(
        self,
        collection: str,
        query_vector: Sequence[float],
        limit: int,
        filters: Sequence[EqualityFilter] = (),
    ) -> list[SearchResult]:
        """Find the most similar points to a query vector.

        Args:
            collection: Name of the collection to search.
            query_vector: Vector to find similar points to.
            limit: Maximum number of results to return.
            filters: Equality filters results must satisfy.

        Returns:
            Matches ordered by descending similarity score.

        Raises:
            VectorRetrieverError: The search failed.
        """

    @abstractmethod
    def retrieve_by_ids(self, collection: str, ids: Sequence[UUID]) -> list[VectorPoint]:
        """Retrieve specific points by id, to hydrate graph-discovered candidates with content.

        Args:
            collection: Name of the collection to read from.
            ids: Identifiers of the points to retrieve.

        Returns:
            The points that exist, in no particular order. Missing ids are
            silently omitted, not errored on.

        Raises:
            VectorRetrieverError: The retrieval failed.
        """
