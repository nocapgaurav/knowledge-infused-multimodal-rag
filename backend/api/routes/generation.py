"""API routes for the grounded answer generation engine.

The handler composes Module 9's retrieval and Module 10's generation for
one convenient HTTP call -- this composition lives only here, in the thin
API layer, never inside `GenerationService` itself. `GenerationService`
always receives an already-built `EvidenceBundle` as a plain argument and
never reaches back into retrieval on its own; that independence is what
this route's two separate `Depends()` calls make concrete.
"""

from fastapi import APIRouter, Depends

from backend.api.dependencies import (
    get_generation_config,
    get_generation_service,
    get_retrieval_service,
)
from backend.api.schemas.generation import GenerateAnswerRequest
from backend.domain import PaperId
from backend.generation.models.generation_config import GenerationConfig
from backend.generation.models.grounded_response import GroundedResponse
from backend.generation.services.generation_service import GenerationService
from backend.retrieval.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/documents", tags=["generation"])


@router.post("/{document_id}/generate", response_model=GroundedResponse)
async def generate_answer(
    document_id: PaperId,
    request: GenerateAnswerRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    generation_service: GenerationService = Depends(get_generation_service),
    config: GenerationConfig = Depends(get_generation_config),
) -> GroundedResponse:
    """Retrieve evidence and generate a grounded answer for one question.

    Read-only with respect to every prior module's own storage: retrieval
    reads Qdrant and Neo4j exactly as Module 9 already does; generation
    never touches either, consuming only the resulting `EvidenceBundle`.

    Args:
        document_id: Identifier of the document to answer a question about.
        request: The user's question.
        retrieval_service: Retrieval service, injected.
        generation_service: Generation service, injected.
        config: Generation configuration, injected.

    Returns:
        The complete, evidence-grounded response.
    """
    bundle = retrieval_service.retrieve(document_id, request.query)
    return generation_service.generate(bundle, config)
