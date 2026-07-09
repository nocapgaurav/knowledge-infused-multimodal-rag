"""Request and response schemas for the knowledge graph API."""

from typing import Literal

from pydantic import BaseModel

from backend.domain import PaperId


class BuildGraphResponse(BaseModel):
    """Response returned after successfully building a document's knowledge graph.

    Attributes:
        document_id: Identifier of the document the graph was built for.
        nodes: Number of nodes currently in the document's graph.
        relationships: Number of edges currently in the document's graph.
        status: Always `"GRAPH_CREATED"` -- a failure raises instead of
            returning this schema with a different status value.
    """

    document_id: PaperId
    nodes: int
    relationships: int
    status: Literal["GRAPH_CREATED"] = "GRAPH_CREATED"
