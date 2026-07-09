"""Request and response schemas for the knowledge representation API."""

from typing import Literal

from pydantic import BaseModel

from backend.domain import PaperId


class RepresentDocumentResponse(BaseModel):
    """Response returned after successfully representing a document.

    Attributes:
        document_id: Identifier of the represented document.
        knowledge_units: Number of knowledge units (chunks) produced.
        relationships: Number of relationships produced between them.
        status: Always `"REPRESENTED"` -- a failure raises instead of
            returning this schema with a different status value.
    """

    document_id: PaperId
    knowledge_units: int
    relationships: int
    status: Literal["REPRESENTED"] = "REPRESENTED"
