"""API routes for the document parsing engine.

Handlers only translate between HTTP and the parser service -- no parsing,
mapping, validation, or persistence logic lives here.
"""

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_parser_service
from backend.api.schemas.parsing import ParseDocumentResponse
from backend.domain import PaperId
from backend.parser.services.parser_service import ParserService

router = APIRouter(prefix="/documents", tags=["parsing"])


@router.post("/{document_id}/parse", response_model=ParseDocumentResponse)
async def parse_document(
    document_id: PaperId,
    service: ParserService = Depends(get_parser_service),
) -> ParseDocumentResponse:
    """Parse a previously ingested document into a structured `Paper`.

    Processing is synchronous: the response is only returned once parsing
    and artifact persistence have completed.

    Args:
        document_id: Identifier of a document already accepted by the
            ingestion pipeline.
        service: Parser service, injected.

    Returns:
        Confirmation that the document was parsed.
    """
    service.parse_document(document_id)
    return ParseDocumentResponse(document_id=document_id)
