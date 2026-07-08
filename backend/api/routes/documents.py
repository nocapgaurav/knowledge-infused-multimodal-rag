"""API routes for the document ingestion pipeline.

Handlers only translate between HTTP and the ingestion service -- no
validation, persistence, or status logic lives here.
"""

from fastapi import APIRouter, Depends, File, UploadFile, status

from backend.api.dependencies import get_ingestion_service
from backend.api.schemas.documents import DocumentStatusResponse, DocumentUploadResponse
from backend.domain import PaperId
from backend.ingestion.service import DocumentIngestionService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentIngestionService = Depends(get_ingestion_service),
) -> DocumentUploadResponse:
    """Accept and ingest an uploaded PDF document.

    Args:
        file: Uploaded file.
        service: Document ingestion service, injected.

    Returns:
        The identity and initial status of the ingested document.
    """
    content = await file.read()
    job = service.ingest(
        filename=file.filename or "upload.pdf",
        content_type=file.content_type or "application/octet-stream",
        content=content,
    )
    return DocumentUploadResponse(
        document_id=job.document_id, upload_job_id=job.id, status=job.status
    )


@router.get("/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(
    document_id: PaperId,
    service: DocumentIngestionService = Depends(get_ingestion_service),
) -> DocumentStatusResponse:
    """Return a document's current ingestion status.

    Args:
        document_id: Identifier of the document to look up.
        service: Document ingestion service, injected.

    Returns:
        The document's current ingestion status. Does not reflect parsing
        status, since parsing is not implemented by this module.
    """
    job = service.get_status(document_id)
    return DocumentStatusResponse(
        document_id=job.document_id,
        upload_job_id=job.id,
        status=job.status,
        error_message=job.error_message,
        updated_at=job.updated_at,
    )
