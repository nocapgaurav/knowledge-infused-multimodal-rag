"""Database-agnostic vector store data shapes.

These are what business logic (planner, service, validator) actually
manipulates -- never a Qdrant-specific type. A concrete provider translates
between these and whatever shape its backend needs.

Plain `dataclasses`, not Pydantic: this is trusted, internal plumbing
constructed by this module's own code and consumed by a provider in the
same process, with no external boundary to validate against.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID


class DistanceMetric(StrEnum):
    """Similarity metric a collection is configured with."""

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


class PayloadFieldType(StrEnum):
    """Type of a payload field, for building a filterable index on it."""

    KEYWORD = "keyword"
    INTEGER = "integer"


@dataclass(frozen=True)
class IndexedField:
    """One payload field a collection should build a filter index on.

    Attributes:
        name: Payload field name.
        field_type: Type of the field, determining what kind of index to build.
    """

    name: str
    field_type: PayloadFieldType


@dataclass(frozen=True)
class VectorPoint:
    """One vector plus its metadata, ready to be upserted.

    Attributes:
        id: Unique identifier for this point. Reused as the vector store's
            native point id.
        vector: The embedding vector.
        payload: Metadata accompanying this vector, for filtering and display.
    """

    id: UUID
    vector: list[float]
    payload: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    """One match from a similarity search.

    Attributes:
        id: Identifier of the matched point.
        score: Similarity score, in the matched collection's own metric.
        payload: The matched point's stored metadata.
    """

    id: UUID
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class CollectionInfo:
    """Describes an existing collection's configuration and state.

    Attributes:
        name: Collection name.
        dimension: Configured vector dimension.
        distance: Configured similarity metric.
        point_count: Number of points currently stored.
    """

    name: str
    dimension: int
    distance: DistanceMetric
    point_count: int


@dataclass(frozen=True)
class EqualityFilter:
    """A single "field equals value" filter condition.

    The only filter shape this module needs -- its own verification step
    only ever filters by `document_id`. Range and boolean-combinator
    filters are a real future need (Module 9's), not built speculatively here.

    Attributes:
        field: Payload field to filter on.
        value: Value the field must equal.
    """

    field: str
    value: str | int | bool
