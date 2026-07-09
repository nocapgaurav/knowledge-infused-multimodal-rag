"""Orchestrates graph construction.

Check staleness (skip if unchanged, unless forced) -> read the knowledge
representation -> plan the graph -> validate it in memory -> ensure the
store's schema -> replace the stored graph -> verify against the real
store -> persist the graph manifest. Each step (repository, planner,
validator, provider) is independently testable; this class's only job is
calling them in the right order.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from backend.domain import PaperId
from backend.graph.exceptions import GraphStoreError
from backend.graph.interfaces.knowledge_graph_store import KnowledgeGraphStore
from backend.graph.models import GraphManifest, KnowledgeGraph
from backend.graph.planner.graph_planner import GraphPlanner
from backend.graph.repository.graph_repository import GraphRepository
from backend.graph.validator.graph_validator import GraphValidator

logger = logging.getLogger(__name__)

ARTIFACT_SCHEMA_VERSION = "1.0"
GRAPH_CONSTRUCTION_VERSION = "1.0"
"""Version of this module's own node/edge derivation rules. Bumped when
those rules change, independently of the knowledge representation --
see the Phase 1 architectural review's answer to graph versioning."""


@dataclass(frozen=True)
class GraphBuildResult:
    """The outcome of one `build_graph` call.

    Attributes:
        manifest: The manifest describing this run (freshly generated, or
            the existing one if construction was skipped as not stale).
        newly_built: Whether this call actually (re)built and persisted the
            graph, as opposed to skipping because it was already fresh.
    """

    manifest: GraphManifest
    newly_built: bool


class GraphService:
    """Builds and persists a document's knowledge graph."""

    def __init__(self, repository: GraphRepository, store: KnowledgeGraphStore) -> None:
        """Initialize the service.

        Args:
            repository: Reads knowledge representation artifacts and
                persists graph manifests.
            store: The graph store to persist into.
        """
        self._repository = repository
        self._store = store
        self._planner = GraphPlanner()
        self._validator = GraphValidator()

    def build_graph(self, document_id: PaperId, force: bool = False) -> GraphBuildResult:
        """Build a document's knowledge graph, verifying the result.

        Idempotent by default: if the existing graph is already fresh
        (the knowledge representation is unchanged and this module's
        construction rules haven't changed version), no work is done and
        the existing manifest is returned unchanged.

        Args:
            document_id: Identifier of a document with an existing
                knowledge representation (Module 5's output).
            force: Rebuild even if the existing graph is already fresh.

        Returns:
            The manifest and whether the graph was actually rebuilt.

        Raises:
            KnowledgeRepresentationNotFoundError: No knowledge
                representation exists for this document.
            GraphValidationError: The freshly constructed graph failed
                pre-persistence verification.
            GraphStoreError: Persisting to the graph store failed.
            GraphStorageError: A storage failure prevented the manifest
                from being persisted.
        """
        representation_hash = self._repository.compute_representation_hash(document_id)
        if not force:
            existing = self._repository.load_graph_manifest(document_id)
            if existing is not None and self._is_fresh(existing, representation_hash):
                logger.info(
                    "graph already up to date, skipping construction",
                    extra={"document_id": str(document_id)},
                )
                return GraphBuildResult(manifest=existing, newly_built=False)

        chunks = self._repository.read_chunks(document_id)
        relationships = self._repository.read_relationships(document_id)

        graph = self._planner.plan(document_id, chunks, relationships)
        self._validator.validate(document_id, graph, chunks, relationships)

        self._store.ensure_schema()
        try:
            self._store.replace_graph(graph)
        except GraphStoreError:
            logger.error(
                "graph store replacement failed",
                exc_info=True,
                extra={"document_id": str(document_id)},
            )
            raise
        self._verify_store_matches(document_id, graph)

        manifest = GraphManifest(
            document_id=document_id,
            artifact_version=ARTIFACT_SCHEMA_VERSION,
            graph_version=GRAPH_CONSTRUCTION_VERSION,
            node_count=graph.node_count,
            relationship_count=graph.relationship_count,
            checksum=_hash_graph(graph),
            source_representation_version=representation_hash,
            created_at=datetime.now(UTC),
        )
        self._repository.save_graph_manifest(document_id, manifest)

        logger.info(
            "document graph built",
            extra={
                "document_id": str(document_id),
                "nodes": graph.node_count,
                "relationships": graph.relationship_count,
            },
        )
        return GraphBuildResult(manifest=manifest, newly_built=True)

    def _is_fresh(self, existing: GraphManifest, representation_hash: str) -> bool:
        return (
            existing.source_representation_version == representation_hash
            and existing.graph_version == GRAPH_CONSTRUCTION_VERSION
        )

    def _verify_store_matches(self, document_id: PaperId, graph: KnowledgeGraph) -> None:
        summary = self._store.graph_summary(document_id)
        if (
            summary.node_count != graph.node_count
            or summary.relationship_count != graph.relationship_count
        ):
            raise GraphStoreError(
                reason=(
                    f"post-write verification failed for {document_id}: "
                    f"expected {graph.node_count} nodes / {graph.relationship_count} "
                    f"relationships, store reports {summary.node_count} / "
                    f"{summary.relationship_count}"
                )
            )


def _hash_graph(graph: KnowledgeGraph) -> str:
    payload = {
        "nodes": sorted(
            (node.model_dump(mode="json") for node in graph.nodes),
            key=lambda item: item["id"],
        ),
        "edges": sorted(
            (edge.model_dump(mode="json") for edge in graph.edges),
            key=lambda item: (item["source_id"], item["target_id"], item["relationship_type"]),
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
