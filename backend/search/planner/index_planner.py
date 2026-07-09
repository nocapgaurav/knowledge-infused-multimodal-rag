"""Groups embedding artifacts into per-collection indexing plans."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from backend.embeddings.models import EmbeddingArtifact
from backend.search.collection_naming import build_collection_name
from backend.search.models import DistanceMetric


@dataclass(frozen=True)
class CollectionPlan:
    """One collection's worth of embedding artifacts to index.

    Attributes:
        collection_name: Deterministic name of the target collection.
        dimension: Vector dimension all artifacts in this plan share.
        distance: Similarity metric to configure the collection with.
        artifacts: Embedding artifacts routed to this collection.
    """

    collection_name: str
    dimension: int
    distance: DistanceMetric
    artifacts: list[EmbeddingArtifact]


class IndexPlanner:
    """Groups embedding artifacts by (model, version, target) into per-collection plans.

    A collection is the unit Qdrant enforces one fixed vector configuration
    on -- grouping here, once, is what lets every downstream step (payload
    building, upserting, verification) operate one collection at a time
    without re-deriving this grouping.
    """

    def __init__(
        self, collection_prefix: str, distance: DistanceMetric = DistanceMetric.COSINE
    ) -> None:
        """Initialize the planner.

        Args:
            collection_prefix: Namespace prefix for collection names.
            distance: Similarity metric new collections are configured with.
        """
        self._collection_prefix = collection_prefix
        self._distance = distance

    def plan(self, artifacts: Sequence[EmbeddingArtifact]) -> list[CollectionPlan]:
        """Group embedding artifacts into per-collection plans.

        Args:
            artifacts: Embedding artifacts to plan indexing for.

        Returns:
            One `CollectionPlan` per distinct (model, version, target)
            combination present in `artifacts`.
        """
        groups: dict[tuple[str, str, str], list[EmbeddingArtifact]] = defaultdict(list)
        for artifact in artifacts:
            key = (artifact.model_name, artifact.model_version, artifact.target.value)
            groups[key].append(artifact)

        return [
            CollectionPlan(
                collection_name=build_collection_name(
                    self._collection_prefix, model_name, model_version, target
                ),
                dimension=group[0].embedding_dimension,
                distance=self._distance,
                artifacts=group,
            )
            for (model_name, model_version, target), group in groups.items()
        ]
