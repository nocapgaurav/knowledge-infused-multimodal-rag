"""Persists generation manifests.

Reads nothing: every fact this module needs is either passed in directly
(the `EvidenceBundle` argument) or produced by its own phases. Unlike
every other module's repository, this one has no upstream artifacts to
read -- it exists solely to persist this module's own reproducibility record.
"""

import logging

from backend.domain import PaperId
from backend.generation.exceptions import GenerationStorageError
from backend.generation.models.generation_manifest import GenerationManifest
from backend.storage.exceptions import StorageError
from backend.storage.interfaces import WorkspaceStorage

logger = logging.getLogger(__name__)

_GENERATION_MANIFEST_FILENAME = "generation_manifest.json"


class GenerationRepository:
    """Persists generation manifests."""

    def __init__(self, generation_storage: WorkspaceStorage) -> None:
        """Initialize the repository.

        Args:
            generation_storage: Storage backend to persist generation manifests into.
        """
        self._generation_storage = generation_storage

    def save_generation_manifest(self, document_id: PaperId, manifest: GenerationManifest) -> None:
        """Persist a generation manifest, overwriting any previous run's for this document.

        Generation is a per-query operation, never skipped as "already
        answered" -- this is a reproducibility record of the most recent
        run, not a staleness cache-check target.

        Args:
            document_id: Identifier of the document the question was asked against.
            manifest: The manifest to persist.

        Raises:
            GenerationStorageError: A storage failure prevented persistence.
        """
        try:
            if not self._generation_storage.workspace_exists(document_id):
                self._generation_storage.create_workspace(document_id)
            self._generation_storage.write_json(
                document_id, _GENERATION_MANIFEST_FILENAME, manifest.model_dump(mode="json")
            )
        except StorageError as exc:
            raise GenerationStorageError(document_id=document_id) from exc
