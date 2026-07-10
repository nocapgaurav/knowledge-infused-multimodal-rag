"""Request schema for the grounded answer generation API.

No response schema is defined here, for the same reason Module 9's
retrieval route has none: the route's response is
`backend.generation.models.GroundedResponse` directly -- the full,
structured response *is* the contract, not a curated summary of it.
"""

from pydantic import BaseModel, Field


class GenerateAnswerRequest(BaseModel):
    """Request body for a generation call.

    Attributes:
        query: The user's question.
    """

    query: str = Field(min_length=1)
