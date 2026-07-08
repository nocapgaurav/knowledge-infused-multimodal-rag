"""Domain layer: canonical, framework-agnostic data contracts shared by every module.

Every future module -- parsing, chunking, embeddings, retrieval, the
knowledge graph, and generation -- produces or consumes these types rather
than defining its own. Import from this package, not from its submodules:
`from backend.domain import Paper`, not `from backend.domain.paper import Paper`.
"""

from backend.domain.base import DomainModel
from backend.domain.chunk import Chunk, ChunkModality
from backend.domain.identifiers import (
    AnswerId,
    AuthorId,
    CaptionId,
    ChunkId,
    EvidenceId,
    FigureId,
    MetadataId,
    PaperId,
    ParagraphId,
    QueryId,
    ReferenceId,
    SectionId,
    TableId,
)
from backend.domain.paper import Author, Metadata, Paper
from backend.domain.qa import Answer, Evidence, Query
from backend.domain.reference import Reference
from backend.domain.structure import Paragraph, Section
from backend.domain.value_objects import BoundingBox
from backend.domain.visuals import Caption, CaptionSubjectType, Figure, Table, TableCell

__all__ = [
    "Answer",
    "AnswerId",
    "Author",
    "AuthorId",
    "BoundingBox",
    "Caption",
    "CaptionId",
    "CaptionSubjectType",
    "Chunk",
    "ChunkId",
    "ChunkModality",
    "DomainModel",
    "Evidence",
    "EvidenceId",
    "Figure",
    "FigureId",
    "Metadata",
    "MetadataId",
    "Paper",
    "PaperId",
    "Paragraph",
    "ParagraphId",
    "Query",
    "QueryId",
    "Reference",
    "ReferenceId",
    "Section",
    "SectionId",
    "Table",
    "TableCell",
    "TableId",
]
