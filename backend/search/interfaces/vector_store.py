"""VectorStore: the port every concrete vector database backend implements.

Business logic (planner, service, validator) depends only on this
interface, never on a concrete backend. Swapping Qdrant for Milvus,
Weaviate, or pgvector later means writing one new provider file -- zero
changes anywhere else. `search()` is defined and tested here even though
this module never calls it itself: `VectorStore` is the same interface
Module 9 (hybrid retrieval) will depend on, and extending the interface
after Module 9 already exists would be the wrong order of operations.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from uuid import UUID

from backend.search.models import (
    CollectionInfo,
    DistanceMetric,
    EqualityFilter,
    IndexedField,
    SearchResult,
    VectorPoint,
)


class VectorStore(ABC):
    """A vector database capable of storing, filtering, and searching embeddings."""

    @abstractmethod
    def collection_exists(self, collection: str) -> bool:
        """Check whether a collection already exists.

        Args:
            collection: Name of the collection to check.

        Returns:
            `True` if the collection exists.
        """

    @abstractmethod
    def ensure_collection(
        self,
        collection: str,
        dimension: int,
        distance: DistanceMetric,
        indexed_fields: Sequence[IndexedField],
    ) -> None:
        """Create a collection if it does not already exist.

        Idempotent: calling this repeatedly with the same arguments is safe.

        Args:
            collection: Name of the collection to create.
            dimension: Vector dimension the collection is configured for.
            distance: Similarity metric the collection is configured for.
            indexed_fields: Payload fields to build a filter index on.

        Raises:
            VectorStoreError: The collection could not be created.
        """

    @abstractmethod
    def collection_info(self, collection: str) -> CollectionInfo | None:
        """Return a collection's configuration and current state.

        Args:
            collection: Name of the collection to describe.

        Returns:
            The collection's info, or `None` if it does not exist.
        """

    @abstractmethod
    def upsert(self, collection: str, points: Sequence[VectorPoint]) -> None:
        """Insert or update a batch of points.

        Idempotent: upserting the same point id again overwrites it with
        identical data, producing no duplicate.

        Args:
            collection: Name of the collection to write to.
            points: Points to upsert.

        Raises:
            VectorStoreError: The upsert failed.
        """

    @abstractmethod
    def count(self, collection: str, filters: Sequence[EqualityFilter] = ()) -> int:
        """Count points in a collection, optionally matching filters.

        Args:
            collection: Name of the collection to count in.
            filters: Equality filters to apply. Empty means count everything.

        Returns:
            The number of matching points.
        """

    @abstractmethod
    def retrieve(self, collection: str, point_ids: Sequence[UUID]) -> list[VectorPoint]:
        """Retrieve specific points by id.

        Args:
            collection: Name of the collection to read from.
            point_ids: Identifiers of the points to retrieve.

        Returns:
            The points that exist, in no particular order. Missing ids are
            silently omitted, not errored on.
        """

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
        """
