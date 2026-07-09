"""Reads the knowledge representation and persists embedding artifacts.

Owns the artifact-specific logic that doesn't belong in the generic
`WorkspaceStorage` abstraction: computing the representation-version hash
used for staleness detection, and typed load/save of the embedding and
manifest shapes. Unlike Module 5's rejected `chunking/storage/`, this is
not a hollow wrapper -- it has real behavior of its own.
"""

import hashlib
import json
import logging
from typing import Any

from backend.domain import PaperId
from backend.embeddings.exceptions import EmbeddingStorageError, RepresentationNotFoundError
from backend.embeddings.models import EmbeddingArtifact, EmbeddingManifest
from backend.storage.exceptions import StorageError
from backend.storage.interfaces import WorkspaceStorage

logger = logging.getLogger(__name__)

_KNOWLEDGE_UNITS_FILENAME = "knowledge_units.json"
_EMBEDDINGS_FILENAME = "embeddings.json"
_MANIFEST_FILENAME = "manifest.json"


class EmbeddingRepository:
    """Reads knowledge representations and persists embedding artifacts."""

    def __init__(
        self, knowledge_storage: WorkspaceStorage, embeddings_storage: WorkspaceStorage
    ) -> None:
        """Initialize the repository.

        Args:
            knowledge_storage: Storage backend holding knowledge
                representation artifacts (Module 5's output).
            embeddings_storage: Storage backend to persist embedding
                artifacts into.
        """
        self._knowledge_storage = knowledge_storage
        self._embeddings_storage = embeddings_storage

    def read_knowledge_units_payload(self, document_id: PaperId) -> dict[str, Any]:
        """Return the raw `knowledge_units.json` payload for a document.

        Args:
            document_id: Identifier of the document to read.

        Returns:
            The parsed JSON payload.

        Raises:
            RepresentationNotFoundError: No knowledge representation exists
                for this document.
        """
        if not self._knowledge_storage.workspace_exists(document_id):
            raise RepresentationNotFoundError(document_id=document_id)
        try:
            return self._knowledge_storage.read_json(document_id, _KNOWLEDGE_UNITS_FILENAME)
        except StorageError as exc:
            raise RepresentationNotFoundError(document_id=document_id) from exc

    def compute_representation_version(self, document_id: PaperId) -> str:
        """Compute a content hash of the current knowledge representation.

        Used as `source_representation_version`: comparing this against a
        persisted manifest's recorded value is how document-level
        staleness is detected, without inspecting individual chunks.

        Args:
            document_id: Identifier of the document to hash.

        Returns:
            A SHA-256 hex digest of the current representation's content.
        """
        payload = self.read_knowledge_units_payload(document_id)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def load_manifest(self, document_id: PaperId) -> EmbeddingManifest | None:
        """Return the persisted manifest for a document, if one exists.

        Args:
            document_id: Identifier of the document to look up.

        Returns:
            The existing `EmbeddingManifest`, or `None` if no embeddings
            have ever been generated for this document.
        """
        if not self._embeddings_storage.workspace_exists(document_id):
            return None
        try:
            payload = self._embeddings_storage.read_json(document_id, _MANIFEST_FILENAME)
        except StorageError:
            return None
        return EmbeddingManifest.model_validate(payload)

    def is_stale(
        self, document_id: PaperId, current_model_name: str, current_model_version: str
    ) -> bool:
        """Determine whether existing embeddings for a document are stale.

        Compares the persisted manifest (if any) against the current
        knowledge representation's content hash and the currently
        configured model -- using only metadata, never re-embedding.

        Args:
            document_id: Identifier of the document to check.
            current_model_name: Name of the currently configured embedding model.
            current_model_version: Resolved revision of the currently configured model.

        Returns:
            `True` if no manifest exists yet, the configured model has
            changed, or the knowledge representation has changed since the
            manifest was generated.
        """
        manifest = self.load_manifest(document_id)
        if manifest is None:
            return True
        if (
            manifest.model_name != current_model_name
            or manifest.model_version != current_model_version
        ):
            return True
        return manifest.source_representation_version != self.compute_representation_version(
            document_id
        )

    def save(
        self, document_id: PaperId, artifacts: list[EmbeddingArtifact], manifest: EmbeddingManifest
    ) -> None:
        """Persist embedding artifacts and their manifest.

        Args:
            document_id: Identifier of the document being embedded.
            artifacts: Embedding artifacts to persist.
            manifest: Manifest describing this embedding run.

        Raises:
            EmbeddingStorageError: A storage failure prevented persistence.
        """
        try:
            if not self._embeddings_storage.workspace_exists(document_id):
                self._embeddings_storage.create_workspace(document_id)
            self._embeddings_storage.write_json(
                document_id,
                _EMBEDDINGS_FILENAME,
                {
                    "document_id": str(document_id),
                    "count": len(artifacts),
                    "embeddings": [artifact.model_dump(mode="json") for artifact in artifacts],
                },
            )
            self._embeddings_storage.write_json(
                document_id, _MANIFEST_FILENAME, manifest.model_dump(mode="json")
            )
        except StorageError as exc:
            raise EmbeddingStorageError(document_id=document_id) from exc
