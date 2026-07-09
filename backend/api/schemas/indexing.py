"""Request and response schemas for the search index API."""

from typing import Literal

from pydantic import BaseModel

from backend.domain import PaperId


class IndexDocumentResponse(BaseModel):
    """Response returned after successfully indexing a document.

    Attributes:
        document_id: Identifier of the indexed document.
        collection: Name of the collection the document's vectors were
            indexed into.
        indexed_vectors: Number of vectors currently indexed for this
            document. `0` new work is done if the existing index was
            already fresh, but this still reports the current total.
        status: Always `"INDEXED"` -- a failure raises instead of
            returning this schema with a different status value.
    """

    document_id: PaperId
    collection: str
    indexed_vectors: int
    status: Literal["INDEXED"] = "INDEXED"
