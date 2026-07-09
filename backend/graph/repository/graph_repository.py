"""Reads the knowledge representation and persists graph manifests.

Reading `knowledge_units.json`/`relationships.json` here is the only input
this module ever consumes -- it never parses PDFs, regenerates knowledge
units, regenerates embeddings, or queries Qdrant.
"""

import hashlib
import json
import logging
from typing import Any

from backend.domain import Chunk, PaperId, Relationship
from backend.graph.exceptions import GraphStorageError, KnowledgeRepresentationNotFoundError
from backend.graph.models import GraphManifest
from backend.storage.exceptions import StorageError
from backend.storage.interfaces import WorkspaceStorage

logger = logging.getLogger(__name__)

_KNOWLEDGE_UNITS_FILENAME = "knowledge_units.json"
_RELATIONSHIPS_FILENAME = "relationships.json"
_GRAPH_MANIFEST_FILENAME = "graph_manifest.json"


class GraphRepository:
    """Reads knowledge representation artifacts and persists graph manifests."""

    def __init__(
        self, knowledge_storage: WorkspaceStorage, graph_storage: WorkspaceStorage
    ) -> None:
        """Initialize the repository.

        Args:
            knowledge_storage: Storage backend holding knowledge
                representation artifacts (Module 5's output).
            graph_storage: Storage backend to persist graph manifests into.
        """
        self._knowledge_storage = knowledge_storage
        self._graph_storage = graph_storage

    def read_chunks(self, document_id: PaperId) -> list[Chunk]:
        """Return the document's knowledge units.

        Args:
            document_id: Identifier of the document to read.

        Returns:
            The document's knowledge units, in no particular order.

        Raises:
            KnowledgeRepresentationNotFoundError: No knowledge
                representation exists for this document.
        """
        payload = self._read_knowledge_units_payload(document_id)
        return [Chunk.model_validate(item) for item in payload["chunks"]]

    def read_relationships(self, document_id: PaperId) -> list[Relationship]:
        """Return the document's relationships.

        Args:
            document_id: Identifier of the document to read.

        Returns:
            The document's relationships.

        Raises:
            KnowledgeRepresentationNotFoundError: No knowledge
                representation exists for this document.
        """
        payload = self._read_relationships_payload(document_id)
        return [Relationship.model_validate(item) for item in payload["relationships"]]

    def compute_representation_hash(self, document_id: PaperId) -> str:
        """Compute a content hash of the current knowledge representation.

        Used as `source_representation_version`: comparing this against a
        persisted graph manifest's recorded value is how upstream
        staleness is detected.

        Args:
            document_id: Identifier of the document to hash.

        Returns:
            A SHA-256 hex digest of the combined knowledge units and
            relationships content.
        """
        combined = {
            "chunks": self._read_knowledge_units_payload(document_id),
            "relationships": self._read_relationships_payload(document_id),
        }
        return _hash_payload(combined)

    def load_graph_manifest(self, document_id: PaperId) -> GraphManifest | None:
        """Return the persisted graph manifest for a document, if one exists.

        Args:
            document_id: Identifier of the document to look up.

        Returns:
            The existing `GraphManifest`, or `None` if this document's
            graph has never been built.
        """
        if not self._graph_storage.workspace_exists(document_id):
            return None
        try:
            payload = self._graph_storage.read_json(document_id, _GRAPH_MANIFEST_FILENAME)
        except StorageError:
            return None
        return GraphManifest.model_validate(payload)

    def save_graph_manifest(self, document_id: PaperId, manifest: GraphManifest) -> None:
        """Persist a graph manifest.

        Args:
            document_id: Identifier of the document the graph was built for.
            manifest: The manifest to persist.

        Raises:
            GraphStorageError: A storage failure prevented persistence.
        """
        try:
            if not self._graph_storage.workspace_exists(document_id):
                self._graph_storage.create_workspace(document_id)
            self._graph_storage.write_json(
                document_id, _GRAPH_MANIFEST_FILENAME, manifest.model_dump(mode="json")
            )
        except StorageError as exc:
            raise GraphStorageError(document_id=document_id) from exc

    def _read_knowledge_units_payload(self, document_id: PaperId) -> dict[str, Any]:
        if not self._knowledge_storage.workspace_exists(document_id):
            raise KnowledgeRepresentationNotFoundError(document_id=document_id)
        try:
            return self._knowledge_storage.read_json(document_id, _KNOWLEDGE_UNITS_FILENAME)
        except StorageError as exc:
            raise KnowledgeRepresentationNotFoundError(document_id=document_id) from exc

    def _read_relationships_payload(self, document_id: PaperId) -> dict[str, Any]:
        if not self._knowledge_storage.workspace_exists(document_id):
            raise KnowledgeRepresentationNotFoundError(document_id=document_id)
        try:
            return self._knowledge_storage.read_json(document_id, _RELATIONSHIPS_FILENAME)
        except StorageError as exc:
            raise KnowledgeRepresentationNotFoundError(document_id=document_id) from exc


def _hash_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
