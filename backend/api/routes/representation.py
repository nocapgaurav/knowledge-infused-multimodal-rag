"""API routes for the knowledge representation engine.

Handlers only translate between HTTP and the representation service -- no
building, validation, or persistence logic lives here.
"""

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_knowledge_representation_service
from backend.api.schemas.representation import RepresentDocumentResponse
from backend.chunking.services.knowledge_representation_service import (
    KnowledgeRepresentationService,
)
from backend.domain import PaperId

router = APIRouter(prefix="/documents", tags=["representation"])


@router.post("/{document_id}/represent", response_model=RepresentDocumentResponse)
async def represent_document(
    document_id: PaperId,
    service: KnowledgeRepresentationService = Depends(get_knowledge_representation_service),
) -> RepresentDocumentResponse:
    """Build and persist the knowledge representation for a parsed document.

    Processing is synchronous: the response is only returned once building,
    validation, and artifact persistence have completed.

    Args:
        document_id: Identifier of a document already parsed by Module 4.
        service: Knowledge representation service, injected.

    Returns:
        Confirmation that the document was represented, with unit and
        relationship counts.
    """
    chunks, relationships = service.represent_document(document_id)
    return RepresentDocumentResponse(
        document_id=document_id,
        knowledge_units=len(chunks),
        relationships=len(relationships),
    )
