"""Request and response schemas for the document ingestion API."""

from datetime import datetime

from pydantic import BaseModel

from backend.domain import PaperId
from backend.ingestion.identifiers import UploadJobId
from backend.ingestion.models import UploadStatus


class DocumentUploadResponse(BaseModel):
    """Response returned after successfully accepting a document upload.

    Attributes:
        document_id: Identifier assigned to the uploaded document. This is
            the same identifier the parser will later assign to the
            resulting `Paper`.
        upload_job_id: Identifier of the upload job tracking this
            document's ingestion lifecycle.
        status: Current ingestion status.
    """

    document_id: PaperId
    upload_job_id: UploadJobId
    status: UploadStatus


class DocumentStatusResponse(BaseModel):
    """Response returned when querying a document's ingestion status.

    Attributes:
        document_id: Identifier of the document.
        upload_job_id: Identifier of the upload job.
        status: Current ingestion status.
        error_message: Reason ingestion failed, set only when `status` is `FAILED`.
        updated_at: Timestamp this status was last updated.
    """

    document_id: PaperId
    upload_job_id: UploadJobId
    status: UploadStatus
    error_message: str | None
    updated_at: datetime
