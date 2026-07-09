"""Orchestrates knowledge representation: read parsed Paper, build, validate, persist."""

import logging

from backend.chunking.builder.knowledge_builder import KnowledgeBuilder
from backend.chunking.exceptions import PaperNotParsedError, RepresentationStorageError
from backend.chunking.validator.knowledge_validator import validate_representation
from backend.domain import Chunk, Paper, PaperId, Relationship
from backend.storage.exceptions import StorageError
from backend.storage.interfaces import WorkspaceStorage

logger = logging.getLogger(__name__)

_PAPER_FILENAME = "paper.json"
_KNOWLEDGE_UNITS_FILENAME = "knowledge_units.json"
_RELATIONSHIPS_FILENAME = "relationships.json"


class KnowledgeRepresentationService:
    """Builds and persists the knowledge representation for a parsed document.

    Reads the already-persisted `Paper` artifact from parsed storage
    (Module 4's output) rather than requiring a fresh parse -- this and the
    parse step are independently triggerable operations.
    """

    def __init__(
        self,
        parsed_storage: WorkspaceStorage,
        knowledge_storage: WorkspaceStorage,
        builder: KnowledgeBuilder,
    ) -> None:
        """Initialize the service.

        Args:
            parsed_storage: Storage backend holding parsed `Paper` artifacts.
            knowledge_storage: Storage backend to persist knowledge
                representation artifacts into.
            builder: Builds chunks and relationships from a `Paper`.
        """
        self._parsed_storage = parsed_storage
        self._knowledge_storage = knowledge_storage
        self._builder = builder

    def represent_document(self, document_id: PaperId) -> tuple[list[Chunk], list[Relationship]]:
        """Build and persist the knowledge representation for a parsed document.

        Args:
            document_id: Identifier of a document already parsed by Module 4.

        Returns:
            The knowledge units and relationships produced.

        Raises:
            PaperNotParsedError: No parsed `Paper` artifact exists for this id.
            EmptyRepresentationError: The paper produced no knowledge units.
            DuplicateChunkOrderError: The builder produced inconsistent ordering.
            InvalidChunkReferenceError: A chunk's paper/section reference is inconsistent.
            DanglingRelationshipError: A relationship references a nonexistent chunk.
            RepresentationStorageError: A storage failure prevented artifacts
                from being persisted.
        """
        paper = self._read_parsed_paper(document_id)

        result = self._builder.build(paper)
        validate_representation(paper, result.chunks, result.relationships)

        self._persist(document_id, result.chunks, result.relationships)

        logger.info(
            "document represented",
            extra={
                "document_id": str(document_id),
                "chunks": len(result.chunks),
                "relationships": len(result.relationships),
            },
        )
        return result.chunks, result.relationships

    def _read_parsed_paper(self, document_id: PaperId) -> Paper:
        if not self._parsed_storage.workspace_exists(document_id):
            raise PaperNotParsedError(document_id=document_id)
        try:
            payload = self._parsed_storage.read_json(document_id, _PAPER_FILENAME)
        except StorageError as exc:
            raise PaperNotParsedError(document_id=document_id) from exc
        return Paper.model_validate(payload)

    def _persist(
        self, document_id: PaperId, chunks: list[Chunk], relationships: list[Relationship]
    ) -> None:
        try:
            if not self._knowledge_storage.workspace_exists(document_id):
                self._knowledge_storage.create_workspace(document_id)
            self._knowledge_storage.write_json(
                document_id,
                _KNOWLEDGE_UNITS_FILENAME,
                {
                    "document_id": str(document_id),
                    "count": len(chunks),
                    "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
                },
            )
            self._knowledge_storage.write_json(
                document_id,
                _RELATIONSHIPS_FILENAME,
                {
                    "document_id": str(document_id),
                    "count": len(relationships),
                    "relationships": [
                        relationship.model_dump(mode="json") for relationship in relationships
                    ],
                },
            )
        except StorageError as exc:
            raise RepresentationStorageError(document_id=document_id) from exc
