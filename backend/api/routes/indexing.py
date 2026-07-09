"""API routes for the search index infrastructure.

Handlers only translate between HTTP and the indexing service -- no
planning, payload building, vector store calls, or verification logic
lives here.
"""

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_indexing_service
from backend.api.schemas.indexing import IndexDocumentResponse
from backend.domain import PaperId
from backend.search.services.indexing_service import IndexingService

router = APIRouter(prefix="/documents", tags=["indexing"])


@router.post("/{document_id}/index", response_model=IndexDocumentResponse)
async def index_document(
    document_id: PaperId,
    force: bool = False,
    service: IndexingService = Depends(get_indexing_service),
) -> IndexDocumentResponse:
    """Index a document's embedding artifacts into the vector store.

    Idempotent by default: if the existing index is already fresh (the
    embedding manifest hasn't changed), no work is done and the existing
    counts are returned unchanged.

    Args:
        document_id: Identifier of a document with existing embedding
            artifacts (Module 6's output).
        force: Reindex even if the existing index is already fresh.
        service: Indexing service, injected.

    Returns:
        Confirmation that the document was indexed, with the collection
        name and current total indexed vector count.
    """
    result = service.index_document(document_id, force=force)
    return IndexDocumentResponse(
        document_id=document_id,
        collection=result.manifest.collection_name,
        indexed_vectors=result.manifest.indexed_vectors,
    )
