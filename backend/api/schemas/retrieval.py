"""Request schema for the hybrid evidence retrieval API.

No response schema is defined here: the route's response is
`backend.retrieval.models.EvidenceBundle` directly. Every other module's
API wraps its service result in a curated response schema because the
route exposes a small summary distinct from the full internal state --
here the full `EvidenceBundle` *is* the contract (per the module spec:
"Return EvidenceBundle, NOT an answer"), so wrapping it in an empty
subclass would add a type with no fields and no behavior of its own.
"""

from pydantic import BaseModel, Field


class RetrieveEvidenceRequest(BaseModel):
    """Request body for a retrieval call.

    Attributes:
        query: The user's question.
    """

    query: str = Field(min_length=1)
