"""Reads upstream manifests (Modules 6, 7, 8) for provenance and collection
resolution, and persists this module's own retrieval manifest.

Reads nothing else: no knowledge units, no relationships, no parsed
`Paper` -- every fact this module needs about a chunk's content lives in
Qdrant's own payload (fetched through `VectorRetriever`), and every fact
about graph structure lives in Neo4j (fetched through `GraphRetriever`).
This repository only ever touches small, already-persisted metadata files.
"""

import logging

from backend.domain import PaperId
from backend.embeddings.models import EmbeddingManifest
from backend.graph.models import GraphManifest
from backend.retrieval.exceptions import (
    DocumentNotGraphedError,
    DocumentNotIndexedError,
    RetrievalStorageError,
)
from backend.retrieval.models import RetrievalManifest
from backend.search.models import IndexManifest
from backend.storage.exceptions import StorageError
from backend.storage.interfaces import WorkspaceStorage

logger = logging.getLogger(__name__)

_EMBEDDING_MANIFEST_FILENAME = "manifest.json"
_INDEX_MANIFEST_FILENAME = "index_manifest.json"
_GRAPH_MANIFEST_FILENAME = "graph_manifest.json"
_RETRIEVAL_MANIFEST_FILENAME = "retrieval_manifest.json"


class RetrievalRepository:
    """Reads upstream manifests and persists retrieval manifests."""

    def __init__(
        self,
        embeddings_storage: WorkspaceStorage,
        index_storage: WorkspaceStorage,
        graph_storage: WorkspaceStorage,
        retrieval_storage: WorkspaceStorage,
    ) -> None:
        """Initialize the repository.

        Args:
            embeddings_storage: Storage backend holding embedding
                manifests (Module 6's output).
            index_storage: Storage backend holding index manifests
                (Module 7's output).
            graph_storage: Storage backend holding graph manifests
                (Module 8's output).
            retrieval_storage: Storage backend to persist retrieval
                manifests into.
        """
        self._embeddings_storage = embeddings_storage
        self._index_storage = index_storage
        self._graph_storage = graph_storage
        self._retrieval_storage = retrieval_storage

    def resolve_collection(self, document_id: PaperId) -> str:
        """Return the name of the collection this document's vectors are indexed in.

        Args:
            document_id: Identifier of the document to look up.

        Returns:
            The collection name recorded in Module 7's index manifest.

        Raises:
            DocumentNotIndexedError: No index manifest exists for this document.
        """
        return self.read_index_manifest(document_id).collection_name

    def read_index_manifest(self, document_id: PaperId) -> IndexManifest:
        """Return the index manifest for a document.

        Args:
            document_id: Identifier of the document to look up.

        Returns:
            The document's index manifest.

        Raises:
            DocumentNotIndexedError: No index manifest exists for this document.
        """
        if not self._index_storage.workspace_exists(document_id):
            raise DocumentNotIndexedError(document_id=document_id)
        try:
            payload = self._index_storage.read_json(document_id, _INDEX_MANIFEST_FILENAME)
        except StorageError as exc:
            raise DocumentNotIndexedError(document_id=document_id) from exc
        return IndexManifest.model_validate(payload)

    def read_embedding_manifest(self, document_id: PaperId) -> EmbeddingManifest:
        """Return the embedding manifest for a document.

        Args:
            document_id: Identifier of the document to look up.

        Returns:
            The document's embedding manifest.

        Raises:
            DocumentNotIndexedError: No embedding manifest exists for this
                document -- reused here rather than a separate exception,
                since a document with no embeddings also has no index.
        """
        if not self._embeddings_storage.workspace_exists(document_id):
            raise DocumentNotIndexedError(document_id=document_id)
        try:
            payload = self._embeddings_storage.read_json(document_id, _EMBEDDING_MANIFEST_FILENAME)
        except StorageError as exc:
            raise DocumentNotIndexedError(document_id=document_id) from exc
        return EmbeddingManifest.model_validate(payload)

    def read_graph_manifest(self, document_id: PaperId) -> GraphManifest:
        """Return the graph manifest for a document.

        Args:
            document_id: Identifier of the document to look up.

        Returns:
            The document's graph manifest.

        Raises:
            DocumentNotGraphedError: No graph manifest exists for this document.
        """
        if not self._graph_storage.workspace_exists(document_id):
            raise DocumentNotGraphedError(document_id=document_id)
        try:
            payload = self._graph_storage.read_json(document_id, _GRAPH_MANIFEST_FILENAME)
        except StorageError as exc:
            raise DocumentNotGraphedError(document_id=document_id) from exc
        return GraphManifest.model_validate(payload)

    def save_retrieval_manifest(self, document_id: PaperId, manifest: RetrievalManifest) -> None:
        """Persist a retrieval manifest, overwriting any previous run's for this document.

        Retrieval is a per-query operation, never skipped as "already
        fresh" -- unlike prior modules' manifests, this is a
        reproducibility record of the most recent run, not a staleness
        cache-check target.

        Args:
            document_id: Identifier of the document that was queried.
            manifest: The manifest to persist.

        Raises:
            RetrievalStorageError: A storage failure prevented persistence.
        """
        try:
            if not self._retrieval_storage.workspace_exists(document_id):
                self._retrieval_storage.create_workspace(document_id)
            self._retrieval_storage.write_json(
                document_id, _RETRIEVAL_MANIFEST_FILENAME, manifest.model_dump(mode="json")
            )
        except StorageError as exc:
            raise RetrievalStorageError(document_id=document_id) from exc
