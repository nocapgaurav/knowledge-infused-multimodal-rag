"""Reads embedding artifacts and the knowledge representation, and persists index manifests.

Reading `knowledge_units.json`/`relationships.json` here is payload
enrichment (joining back to get `section_id`, `reading_order`, citation
counts), not "regenerating knowledge units" -- nothing here recomputes
anything Module 5 already produced.
"""

import hashlib
import json
import logging
from typing import Any

from backend.domain import Chunk, ChunkId, PaperId, Relationship
from backend.embeddings.models import EmbeddingArtifact, EmbeddingManifest
from backend.search.exceptions import EmbeddingArtifactsNotFoundError, IndexStorageError
from backend.search.models import IndexManifest
from backend.storage.exceptions import StorageError
from backend.storage.interfaces import WorkspaceStorage

logger = logging.getLogger(__name__)

_EMBEDDINGS_FILENAME = "embeddings.json"
_EMBEDDING_MANIFEST_FILENAME = "manifest.json"
_KNOWLEDGE_UNITS_FILENAME = "knowledge_units.json"
_RELATIONSHIPS_FILENAME = "relationships.json"
_INDEX_MANIFEST_FILENAME = "index_manifest.json"


class IndexRepository:
    """Reads embedding/knowledge artifacts and persists index manifests."""

    def __init__(
        self,
        embeddings_storage: WorkspaceStorage,
        knowledge_storage: WorkspaceStorage,
        index_storage: WorkspaceStorage,
    ) -> None:
        """Initialize the repository.

        Args:
            embeddings_storage: Storage backend holding embedding artifacts
                (Module 6's output).
            knowledge_storage: Storage backend holding knowledge
                representation artifacts (Module 5's output).
            index_storage: Storage backend to persist index manifests into.
        """
        self._embeddings_storage = embeddings_storage
        self._knowledge_storage = knowledge_storage
        self._index_storage = index_storage

    def read_embedding_artifacts(self, document_id: PaperId) -> list[EmbeddingArtifact]:
        """Return the embedding artifacts for a document.

        Args:
            document_id: Identifier of the document to read.

        Returns:
            The document's embedding artifacts.

        Raises:
            EmbeddingArtifactsNotFoundError: No embedding artifacts exist
                for this document.
        """
        payload = self._read_embeddings_payload(document_id)
        return [EmbeddingArtifact.model_validate(item) for item in payload["embeddings"]]

    def read_embedding_manifest(self, document_id: PaperId) -> EmbeddingManifest:
        """Return the embedding manifest for a document.

        Args:
            document_id: Identifier of the document to read.

        Returns:
            The document's embedding manifest.

        Raises:
            EmbeddingArtifactsNotFoundError: No embedding manifest exists
                for this document.
        """
        return EmbeddingManifest.model_validate(self._read_embedding_manifest_payload(document_id))

    def read_chunks(self, document_id: PaperId) -> dict[ChunkId, Chunk]:
        """Return the document's knowledge units, keyed by id.

        Args:
            document_id: Identifier of the document to read.

        Returns:
            A mapping of chunk id to chunk.
        """
        payload = self._knowledge_storage.read_json(document_id, _KNOWLEDGE_UNITS_FILENAME)
        chunks = [Chunk.model_validate(item) for item in payload["chunks"]]
        return {chunk.id: chunk for chunk in chunks}

    def read_relationships(self, document_id: PaperId) -> list[Relationship]:
        """Return the document's relationships.

        Args:
            document_id: Identifier of the document to read.

        Returns:
            The document's relationships.
        """
        payload = self._knowledge_storage.read_json(document_id, _RELATIONSHIPS_FILENAME)
        return [Relationship.model_validate(item) for item in payload["relationships"]]

    def compute_embedding_manifest_hash(self, document_id: PaperId) -> str:
        """Compute a content hash of the current embedding manifest.

        Used as `source_embedding_manifest`: comparing this against a
        persisted index manifest's recorded value is how staleness is detected.

        Args:
            document_id: Identifier of the document to hash.

        Returns:
            A SHA-256 hex digest of the embedding manifest's content.
        """
        return _hash_payload(self._read_embedding_manifest_payload(document_id))

    def compute_embeddings_checksum(self, document_id: PaperId) -> str:
        """Compute a content hash of the current embedding artifacts.

        Distinct from `compute_embedding_manifest_hash`: this hashes the
        vector data itself, not the summary metadata file.

        Args:
            document_id: Identifier of the document to hash.

        Returns:
            A SHA-256 hex digest of the embedding artifacts' content.
        """
        return _hash_payload(self._read_embeddings_payload(document_id))

    def load_index_manifest(self, document_id: PaperId) -> IndexManifest | None:
        """Return the persisted index manifest for a document, if one exists.

        Args:
            document_id: Identifier of the document to look up.

        Returns:
            The existing `IndexManifest`, or `None` if this document has
            never been indexed.
        """
        if not self._index_storage.workspace_exists(document_id):
            return None
        try:
            payload = self._index_storage.read_json(document_id, _INDEX_MANIFEST_FILENAME)
        except StorageError:
            return None
        return IndexManifest.model_validate(payload)

    def is_stale(self, document_id: PaperId) -> bool:
        """Determine whether an existing index for a document is stale.

        Args:
            document_id: Identifier of the document to check.

        Returns:
            `True` if the document has never been indexed, or if the
            embedding manifest has changed since it was last indexed.
        """
        manifest = self.load_index_manifest(document_id)
        if manifest is None:
            return True
        return manifest.source_embedding_manifest != self.compute_embedding_manifest_hash(
            document_id
        )

    def save_index_manifest(self, document_id: PaperId, manifest: IndexManifest) -> None:
        """Persist an index manifest.

        Args:
            document_id: Identifier of the document that was indexed.
            manifest: The manifest to persist.

        Raises:
            IndexStorageError: A storage failure prevented persistence.
        """
        try:
            if not self._index_storage.workspace_exists(document_id):
                self._index_storage.create_workspace(document_id)
            self._index_storage.write_json(
                document_id, _INDEX_MANIFEST_FILENAME, manifest.model_dump(mode="json")
            )
        except StorageError as exc:
            raise IndexStorageError(document_id=document_id) from exc

    def _read_embeddings_payload(self, document_id: PaperId) -> dict[str, Any]:
        if not self._embeddings_storage.workspace_exists(document_id):
            raise EmbeddingArtifactsNotFoundError(document_id=document_id)
        try:
            return self._embeddings_storage.read_json(document_id, _EMBEDDINGS_FILENAME)
        except StorageError as exc:
            raise EmbeddingArtifactsNotFoundError(document_id=document_id) from exc

    def _read_embedding_manifest_payload(self, document_id: PaperId) -> dict[str, Any]:
        if not self._embeddings_storage.workspace_exists(document_id):
            raise EmbeddingArtifactsNotFoundError(document_id=document_id)
        try:
            return self._embeddings_storage.read_json(document_id, _EMBEDDING_MANIFEST_FILENAME)
        except StorageError as exc:
            raise EmbeddingArtifactsNotFoundError(document_id=document_id) from exc


def _hash_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
