"""Strongly typed identifiers for domain entities.

Every domain entity is identified by a dedicated `NewType` over `UUID`
rather than a bare `UUID` or `str`. This lets the type checker catch a
whole class of mistakes -- for example, passing a `SectionId` where a
`FigureId` is expected -- that would otherwise only surface as a runtime
lookup failure, often far from the point of the mistake.
"""

from typing import NewType
from uuid import UUID, uuid4

PaperId = NewType("PaperId", UUID)
MetadataId = NewType("MetadataId", UUID)
AuthorId = NewType("AuthorId", UUID)
SectionId = NewType("SectionId", UUID)
ParagraphId = NewType("ParagraphId", UUID)
FigureId = NewType("FigureId", UUID)
TableId = NewType("TableId", UUID)
CaptionId = NewType("CaptionId", UUID)
ReferenceId = NewType("ReferenceId", UUID)
ChunkId = NewType("ChunkId", UUID)
EvidenceId = NewType("EvidenceId", UUID)
QueryId = NewType("QueryId", UUID)
AnswerId = NewType("AnswerId", UUID)


def generate_id() -> UUID:
    """Generate a new random identifier.

    Returns:
        A version 4 UUID suitable for use as any domain entity's identifier.
    """
    return uuid4()
