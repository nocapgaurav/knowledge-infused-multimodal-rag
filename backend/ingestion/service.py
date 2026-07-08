"""Orchestrates document ingestion: validate, assign identity, persist."""

import hashlib
import logging
from datetime import UTC, datetime
from uuid import uuid4

from backend.domain import PaperId
from backend.ingestion.exceptions import IngestionStorageError, UploadJobNotFoundError
from backend.ingestion.identifiers import UploadJobId
from backend.ingestion.models import UploadJob, UploadMetadata, UploadStatus
from backend.ingestion.validation import validate_upload
from backend.storage.exceptions import StorageError
from backend.storage.interfaces import WorkspaceStorage

logger = logging.getLogger(__name__)

_PDF_FILENAME = "paper.pdf"
_METADATA_FILENAME = "upload.json"
_STATUS_FILENAME = "status.json"


class DocumentIngestionService:
    """Accepts, validates, and persists uploaded documents.

    Depends only on the `WorkspaceStorage` interface, not a concrete
    backend, and never touches a framework-specific upload type -- both
    are what keep this class unit-testable and swappable.

    Attributes:
        max_upload_size_bytes: Maximum permitted upload size, in bytes.
    """

    def __init__(self, storage: WorkspaceStorage, max_upload_size_bytes: int) -> None:
        """Initialize the service.

        Args:
            storage: Workspace storage backend to persist documents into.
            max_upload_size_bytes: Maximum permitted upload size, in bytes.
        """
        self._storage = storage
        self.max_upload_size_bytes = max_upload_size_bytes

    def ingest(self, *, filename: str, content_type: str, content: bytes) -> UploadJob:
        """Validate and persist a newly uploaded document.

        Args:
            filename: Filename as supplied by the client.
            content_type: Content type as declared by the client.
            content: Raw file content.

        Returns:
            The `UploadJob` created for this upload.

        Raises:
            UnsupportedFileTypeError: The extension or declared content type is not PDF.
            EmptyFileError: The file contains no bytes.
            FileTooLargeError: The file exceeds the configured size limit.
            InvalidPdfContentError: The file's content does not start with a PDF header.
            IngestionStorageError: A storage failure prevented the document
                from being persisted.
        """
        validate_upload(
            filename=filename,
            content_type=content_type,
            content=content,
            max_size_bytes=self.max_upload_size_bytes,
        )

        document_id = PaperId(uuid4())
        upload_job_id = UploadJobId(uuid4())
        now = datetime.now(UTC)

        metadata = UploadMetadata(
            document_id=document_id,
            upload_job_id=upload_job_id,
            original_filename=filename,
            content_type=content_type,
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            created_at=now,
        )
        job = UploadJob(
            id=upload_job_id,
            document_id=document_id,
            status=UploadStatus.UPLOADED,
            created_at=now,
            updated_at=now,
        )

        try:
            self._storage.create_workspace(document_id)
            self._storage.write_bytes(document_id, _PDF_FILENAME, content)
            self._storage.write_json(
                document_id, _METADATA_FILENAME, metadata.model_dump(mode="json")
            )
            self._storage.write_json(document_id, _STATUS_FILENAME, job.model_dump(mode="json"))
        except StorageError as exc:
            logger.error(
                "failed to persist ingested document",
                exc_info=True,
                extra={"document_id": str(document_id)},
            )
            self._mark_failed_best_effort(document_id, job, exc)
            raise IngestionStorageError(document_id=document_id) from exc

        logger.info(
            "document ingested",
            extra={"document_id": str(document_id), "upload_job_id": str(upload_job_id)},
        )
        return job

    def get_status(self, document_id: PaperId) -> UploadJob:
        """Return the current upload job for a document.

        Args:
            document_id: Identifier of the document to look up.

        Returns:
            The document's current `UploadJob`.

        Raises:
            UploadJobNotFoundError: No upload job exists for this document.
        """
        if not self._storage.workspace_exists(document_id):
            raise UploadJobNotFoundError(document_id=document_id)
        payload = self._storage.read_json(document_id, _STATUS_FILENAME)
        return UploadJob.model_validate(payload)

    def _mark_failed_best_effort(
        self, document_id: PaperId, job: UploadJob, exc: Exception
    ) -> None:
        """Attempt to persist a FAILED status after a storage error.

        Best-effort: a secondary failure here is logged, not raised, so it
        never masks the original error that triggered this call.
        """
        failed_job = job.model_copy(
            update={
                "status": UploadStatus.FAILED,
                "error_message": str(exc),
                "updated_at": datetime.now(UTC),
            }
        )
        try:
            self._storage.write_json(
                document_id, _STATUS_FILENAME, failed_job.model_dump(mode="json")
            )
        except StorageError:
            logger.error(
                "failed to persist FAILED status after a prior storage error",
                exc_info=True,
                extra={"document_id": str(document_id)},
            )
