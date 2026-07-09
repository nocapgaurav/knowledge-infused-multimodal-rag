"""Request and response schemas for the embedding API."""

from typing import Literal

from pydantic import BaseModel

from backend.domain import PaperId


class EmbedDocumentResponse(BaseModel):
    """Response returned after successfully embedding a document.

    Attributes:
        document_id: Identifier of the embedded document.
        embeddings_generated: Number of embedding artifacts produced by
            this call. `0` if existing embeddings were already fresh and
            regeneration was skipped.
        model: Name of the text embedding model used.
        status: Always `"EMBEDDED"` -- a failure raises instead of
            returning this schema with a different status value.
    """

    document_id: PaperId
    embeddings_generated: int
    model: str
    status: Literal["EMBEDDED"] = "EMBEDDED"
