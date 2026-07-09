"""API routes for the hybrid evidence retrieval engine.

The handler only translates between HTTP and the retrieval service -- no
candidate generation, expansion, evaluation, or assembly logic lives here.
"""

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_retrieval_service
from backend.api.schemas.retrieval import RetrieveEvidenceRequest
from backend.domain import PaperId
from backend.retrieval.models import EvidenceBundle
from backend.retrieval.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/documents", tags=["retrieval"])


@router.post("/{document_id}/retrieve", response_model=EvidenceBundle)
async def retrieve_evidence(
    document_id: PaperId,
    request: RetrieveEvidenceRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> EvidenceBundle:
    """Retrieve the highest-quality evidence for a question against one document.

    Read-only: never parses documents, regenerates knowledge units or
    embeddings, modifies Qdrant or Neo4j, or generates an answer. Returns
    evidence, not an answer -- only Module 10 may consume this bundle to
    produce one.

    Args:
        document_id: Identifier of the document to retrieve evidence from.
        request: The user's question.
        service: Retrieval service, injected.

    Returns:
        The complete evidence bundle.
    """
    return service.retrieve(document_id, request.query)
