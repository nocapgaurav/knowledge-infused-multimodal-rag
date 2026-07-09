"""Phase 2: Evidence Expansion.

Budgeted breadth-first traversal from Phase 1's seed candidates through
the knowledge graph, discovering additional candidates via deterministic
relationships. Cycle prevention and duplicate elimination both fall out of
a single global visited-set: a node, once seen (as a seed or as a
discovery), is never explored again and never added twice.
"""

import logging
from dataclasses import dataclass
from uuid import UUID

from backend.domain import ChunkId, ChunkModality, PaperId, SectionId
from backend.retrieval.expansion.relationship_policy import (
    RELATIONSHIP_TYPES,
    directions_for,
)
from backend.retrieval.interfaces.graph_retriever import GraphRetriever
from backend.retrieval.interfaces.vector_retriever import VectorRetriever
from backend.retrieval.models import (
    DiscoveryMethod,
    GraphPath,
    RetrievalCandidate,
    TraversalHop,
)
from backend.search.models import VectorPoint

logger = logging.getLogger(__name__)

_KNOWLEDGE_UNIT_LABEL = "KnowledgeUnit"


@dataclass(frozen=True)
class ExpansionBudget:
    """Hard limits on how much traversal one expansion call may perform.

    Attributes:
        max_depth: Maximum number of hops from any seed candidate.
        max_neighbors_per_node: Maximum neighbors explored from any single
            node, across all relationship types combined -- protects
            against a single hub node (e.g. a large section) causing
            unbounded fan-out.
        max_total_evidence: Maximum number of new candidates expansion may
            discover, across the whole traversal.
        max_traversal_cost: Maximum total neighbor-edges examined (before
            truncation), across the whole traversal -- protects against
            pathological fan-out even when individual per-node caps are respected.
    """

    max_depth: int = 2
    max_neighbors_per_node: int = 10
    max_total_evidence: int = 50
    max_traversal_cost: int = 500


@dataclass(frozen=True)
class ExpansionResult:
    """The outcome of one expansion call.

    Attributes:
        candidates: Newly discovered candidates (never includes the seeds
            themselves), each carrying the path that discovered it.
        traversal_cost: Total neighbor-edges examined, for reporting
            against `ExpansionBudget.max_traversal_cost`.
        budget_exhausted: Which budget dimension, if any, stopped
            traversal early -- `None` if traversal completed naturally
            (frontier exhausted before any limit was hit).
    """

    candidates: list[RetrievalCandidate]
    traversal_cost: int
    budget_exhausted: str | None


class GraphExpander:
    """Expands a seed candidate pool through the knowledge graph."""

    def __init__(self, graph_retriever: GraphRetriever, vector_retriever: VectorRetriever) -> None:
        """Initialize the expander.

        Args:
            graph_retriever: Read-only single-hop neighbor lookup.
            vector_retriever: Read-only vector database access, used to
                hydrate graph-discovered node ids with their actual content.
        """
        self._graph_retriever = graph_retriever
        self._vector_retriever = vector_retriever

    def expand(
        self,
        seed_candidates: list[RetrievalCandidate],
        collection: str,
        budget: ExpansionBudget,
    ) -> ExpansionResult:
        """Expand a seed candidate pool through the graph.

        Args:
            seed_candidates: Phase 1's candidates, the traversal's starting points.
            collection: Name of the collection to hydrate discovered node
                ids' content from.
            budget: Hard limits on traversal depth, breadth, and cost.

        Returns:
            Newly discovered candidates plus traversal accounting.

        Raises:
            GraphRetrieverError: A neighbor lookup failed.
            VectorRetrieverError: Hydrating discovered nodes' content failed.
        """
        visited: set[str] = {str(candidate.knowledge_unit_id) for candidate in seed_candidates}
        frontier: list[tuple[str, GraphPath]] = [
            (str(candidate.knowledge_unit_id), GraphPath()) for candidate in seed_candidates
        ]
        discovered_paths: dict[str, GraphPath] = {}
        traversal_cost = 0
        budget_exhausted: str | None = None

        for _ in range(budget.max_depth):
            if not frontier:
                break
            if len(discovered_paths) >= budget.max_total_evidence:
                budget_exhausted = "max_total_evidence"
                break
            if traversal_cost >= budget.max_traversal_cost:
                budget_exhausted = "max_traversal_cost"
                break

            next_frontier: list[tuple[str, GraphPath]] = []
            for node_id, path in frontier:
                if len(discovered_paths) >= budget.max_total_evidence:
                    budget_exhausted = "max_total_evidence"
                    break
                if traversal_cost >= budget.max_traversal_cost:
                    budget_exhausted = "max_traversal_cost"
                    break

                node_neighbors = []
                for relationship_type in RELATIONSHIP_TYPES:
                    for direction in directions_for(relationship_type):
                        found = self._graph_retriever.neighbors(
                            [node_id], [relationship_type], direction
                        )
                        traversal_cost += len(found)
                        node_neighbors.extend(found)
                node_neighbors = node_neighbors[: budget.max_neighbors_per_node]

                for neighbor in node_neighbors:
                    if neighbor.neighbor_id in visited:
                        continue
                    if len(discovered_paths) >= budget.max_total_evidence:
                        budget_exhausted = "max_total_evidence"
                        break
                    visited.add(neighbor.neighbor_id)
                    new_path = GraphPath(
                        hops=(
                            *path.hops,
                            TraversalHop(
                                source_id=neighbor.source_id,
                                target_id=neighbor.neighbor_id,
                                relationship_type=neighbor.relationship_type,
                                direction=neighbor.direction,
                            ),
                        )
                    )
                    if _KNOWLEDGE_UNIT_LABEL in neighbor.neighbor_labels:
                        discovered_paths[neighbor.neighbor_id] = new_path
                    next_frontier.append((neighbor.neighbor_id, new_path))

            frontier = next_frontier

        candidates = self._hydrate(discovered_paths, collection)
        logger.info(
            "evidence expansion complete",
            extra={
                "discovered": len(candidates),
                "traversal_cost": traversal_cost,
                "budget_exhausted": budget_exhausted,
            },
        )
        return ExpansionResult(
            candidates=candidates, traversal_cost=traversal_cost, budget_exhausted=budget_exhausted
        )

    def _hydrate(
        self, discovered_paths: dict[str, GraphPath], collection: str
    ) -> list[RetrievalCandidate]:
        if not discovered_paths:
            return []
        points = self._vector_retriever.retrieve_by_ids(
            collection, [UUID(node_id) for node_id in discovered_paths]
        )
        if len(points) != len(discovered_paths):
            logger.warning(
                "some graph-discovered nodes had no corresponding vector store point",
                extra={"discovered": len(discovered_paths), "hydrated": len(points)},
            )
        return [_to_candidate(point, discovered_paths[str(point.id)]) for point in points]


def _to_candidate(point: VectorPoint, graph_path: GraphPath) -> RetrievalCandidate:
    payload = point.payload
    section_id_value = payload.get("section_id")
    return RetrievalCandidate(
        knowledge_unit_id=ChunkId(point.id),
        document_id=PaperId(UUID(str(payload["document_id"]))),
        section_id=SectionId(UUID(str(section_id_value))) if section_id_value else None,
        modality=ChunkModality(payload["modality"]),
        text=payload["text"],
        asset_uri=payload.get("asset_uri"),
        reading_order=payload["reading_order"],
        citation_count=payload.get("citation_count") or 0,
        dense_similarity=None,
        discovery_method=DiscoveryMethod.GRAPH_EXPANSION,
        graph_path=graph_path,
    )
