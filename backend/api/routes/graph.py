"""API routes for the knowledge graph infrastructure.

Handlers only translate between HTTP and the graph service -- no planning,
building, validation, or store calls live here.
"""

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_graph_service
from backend.api.schemas.graph import BuildGraphResponse
from backend.domain import PaperId
from backend.graph.services.graph_service import GraphService

router = APIRouter(prefix="/documents", tags=["graph"])


@router.post("/{document_id}/graph", response_model=BuildGraphResponse)
async def build_graph(
    document_id: PaperId,
    force: bool = False,
    service: GraphService = Depends(get_graph_service),
) -> BuildGraphResponse:
    """Build a document's knowledge graph from its knowledge representation.

    Idempotent by default: if the existing graph is already fresh (the
    knowledge representation hasn't changed and this module's construction
    rules haven't changed version), no work is done and the existing
    counts are returned unchanged.

    Args:
        document_id: Identifier of a document with an existing knowledge
            representation (Module 5's output).
        force: Rebuild even if the existing graph is already fresh.
        service: Graph service, injected.

    Returns:
        Confirmation that the graph was built, with its current node and
        relationship counts.
    """
    result = service.build_graph(document_id, force=force)
    return BuildGraphResponse(
        document_id=document_id,
        nodes=result.manifest.node_count,
        relationships=result.manifest.relationship_count,
    )
