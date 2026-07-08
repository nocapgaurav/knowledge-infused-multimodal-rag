"""Request and response schemas for the document parsing API."""

from typing import Literal

from pydantic import BaseModel

from backend.domain import PaperId


class ParseDocumentResponse(BaseModel):
    """Response returned after successfully parsing a document.

    Attributes:
        document_id: Identifier of the parsed document.
        status: Always `"PARSED"` -- a failed parse raises instead of
            returning this schema with a different status value.
    """

    document_id: PaperId
    status: Literal["PARSED"] = "PARSED"
