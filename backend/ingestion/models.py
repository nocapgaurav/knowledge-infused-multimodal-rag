"""Models for the document ingestion pipeline.

These are deliberately not part of `backend.domain`: an `UploadJob`
represents the lifecycle of *accepting* a document, not a fact extracted
from one. A `Paper` does not exist yet at this stage -- only the parser
(Module 4) is permitted to construct one.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from backend.domain import PaperId
from backend.ingestion.identifiers import UploadJobId


class UploadStatus(StrEnum):
    """Lifecycle state of an ingested document.

    `VALIDATING` and `READY_FOR_PARSING` are modeled now for forward
    compatibility -- e.g. once validation becomes asynchronous, or once a
    document needs to explicitly signal "the parser may now pick this up"
    -- but this module's current synchronous implementation only ever
    produces `UPLOADED` (validation passed, fully persisted) or `FAILED`
    (a storage error occurred after validation passed).
    """

    UPLOADED = "UPLOADED"
    VALIDATING = "VALIDATING"
    READY_FOR_PARSING = "READY_FOR_PARSING"
    FAILED = "FAILED"


class UploadMetadata(BaseModel):
    """Immutable facts about a single upload, persisted once as `upload.json`.

    Attributes:
        document_id: Identifier of the ingested document.
        upload_job_id: Identifier of the upload job created for this upload.
        original_filename: Filename as supplied by the client.
        content_type: Content type as declared by the client.
        size_bytes: Size of the uploaded file, in bytes.
        sha256: SHA-256 checksum of the uploaded file's content, computed
            purely as an opaque fingerprint -- this module does not
            interpret the file's content, only hash it as bytes.
        created_at: Timestamp the upload was accepted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: PaperId
    upload_job_id: UploadJobId
    original_filename: str
    content_type: str
    size_bytes: int = Field(ge=0)
    sha256: str
    created_at: datetime


class UploadJob(BaseModel):
    """The current lifecycle state of an ingested document, persisted as `status.json`.

    Attributes:
        id: Unique identifier for this upload job.
        document_id: Identifier of the document this job tracks.
        status: Current ingestion status.
        error_message: Reason ingestion failed, set only when `status` is `FAILED`.
        created_at: Timestamp the job was created.
        updated_at: Timestamp `status` was last changed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: UploadJobId
    document_id: PaperId
    status: UploadStatus
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
