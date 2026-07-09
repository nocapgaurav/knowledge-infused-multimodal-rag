"""API routes for the embedding infrastructure.

Handlers only translate between HTTP and the embedding service -- no
planning, provider calls, validation, or persistence logic lives here.
"""

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_embedding_service
from backend.api.schemas.embedding import EmbedDocumentResponse
from backend.domain import PaperId
from backend.embeddings.services.embedding_service import EmbeddingService

router = APIRouter(prefix="/documents", tags=["embedding"])


@router.post("/{document_id}/embed", response_model=EmbedDocumentResponse)
async def embed_document(
    document_id: PaperId,
    force: bool = False,
    service: EmbeddingService = Depends(get_embedding_service),
) -> EmbedDocumentResponse:
    """Generate and persist embeddings for a document's knowledge representation.

    Idempotent by default: if existing embeddings are already fresh (same
    model, same representation content), no work is done and the existing
    count is returned unchanged.

    Args:
        document_id: Identifier of a document with an existing knowledge
            representation (Module 5's output).
        force: Regenerate even if existing embeddings are already fresh.
        service: Embedding service, injected.

    Returns:
        Confirmation that the document was embedded, with the current
        total embedding count and the model used.
    """
    result = service.embed_document(document_id, force=force)
    return EmbedDocumentResponse(
        document_id=document_id,
        embeddings_generated=result.manifest.embedding_count,
        model=result.manifest.model_name,
    )
