"""Orchestrates document parsing: read, parse, map, validate, persist.

This is the only place these five steps are wired together. Each step
(provider, mapper, validator, storage) is independently testable; this
class's only job is calling them in the right order and translating
failures into the parser's own exception hierarchy.
"""

import logging
from collections import defaultdict
from typing import Any

from backend.domain import BoundingBox, FigureId, Paper, PaperId
from backend.ingestion.models import UploadMetadata
from backend.parser.exceptions import DocumentNotIngestedError, ParserStorageError
from backend.parser.interfaces.document_parser import DocumentParser
from backend.parser.mapper.domain_mapper import DomainMapper
from backend.parser.validator.document_validator import validate_document
from backend.storage.exceptions import StorageError
from backend.storage.interfaces import WorkspaceStorage

logger = logging.getLogger(__name__)

_RAW_PDF_FILENAME = "paper.pdf"
_UPLOAD_METADATA_FILENAME = "upload.json"
_PAPER_FILENAME = "paper.json"
_METADATA_FILENAME = "metadata.json"
_LAYOUT_FILENAME = "layout.json"


class ParserService:
    """Parses an ingested document and persists the resulting structured artifacts.

    Depends only on the `WorkspaceStorage` and `DocumentParser` interfaces,
    never on concrete backends, so it can be tested with fakes for either.
    """

    def __init__(
        self,
        raw_storage: WorkspaceStorage,
        parsed_storage: WorkspaceStorage,
        document_parser: DocumentParser,
        mapper: DomainMapper,
    ) -> None:
        """Initialize the service.

        Args:
            raw_storage: Storage backend holding ingested (unparsed) documents.
            parsed_storage: Storage backend to persist parsed artifacts into.
            document_parser: Parsing engine used to extract document content.
            mapper: Mapper from extracted content into the domain model.
        """
        self._raw_storage = raw_storage
        self._parsed_storage = parsed_storage
        self._document_parser = document_parser
        self._mapper = mapper

    def parse_document(self, document_id: PaperId) -> Paper:
        """Parse a previously ingested document and persist its structured artifacts.

        Args:
            document_id: Identifier of a document already accepted by the
                ingestion pipeline.

        Returns:
            The parsed `Paper`.

        Raises:
            DocumentNotIngestedError: No ingested document exists for this id.
            UnreadablePdfError: The stored PDF could not be parsed.
            MissingRequiredMetadataError: No usable title could be determined.
            EmptyDocumentError: The parsed document has no content.
            InvalidSectionHierarchyError: The document's section hierarchy is broken.
            MissingFigureReferenceError: A caption references a nonexistent figure or table.
            ParserStorageError: A storage failure prevented artifacts from being persisted.
        """
        pdf_bytes, original_filename = self._read_ingested_document(document_id)

        extracted = self._document_parser.parse(pdf_bytes)
        mapping_result = self._mapper.to_paper(
            document_id=document_id, source_filename=original_filename, extracted=extracted
        )
        paper = mapping_result.paper

        validate_document(paper)

        self._persist(paper, mapping_result.figure_images)

        logger.info(
            "document parsed",
            extra={
                "document_id": str(document_id),
                "sections": len(paper.sections),
                "paragraphs": len(paper.paragraphs),
                "figures": len(paper.figures),
                "tables": len(paper.tables),
                "references": len(paper.references),
            },
        )
        return paper

    def _read_ingested_document(self, document_id: PaperId) -> tuple[bytes, str]:
        if not self._raw_storage.workspace_exists(document_id):
            raise DocumentNotIngestedError(document_id=document_id)
        try:
            pdf_bytes = self._raw_storage.read_bytes(document_id, _RAW_PDF_FILENAME)
            upload_metadata = UploadMetadata.model_validate(
                self._raw_storage.read_json(document_id, _UPLOAD_METADATA_FILENAME)
            )
        except StorageError as exc:
            raise DocumentNotIngestedError(document_id=document_id) from exc
        return pdf_bytes, upload_metadata.original_filename

    def _persist(self, paper: Paper, figure_images: dict[FigureId, bytes]) -> None:
        document_id = paper.id
        try:
            if not self._parsed_storage.workspace_exists(document_id):
                self._parsed_storage.create_workspace(document_id)

            self._parsed_storage.write_json(
                document_id, _PAPER_FILENAME, paper.model_dump(mode="json")
            )
            self._parsed_storage.write_json(
                document_id, _METADATA_FILENAME, paper.metadata.model_dump(mode="json")
            )
            self._parsed_storage.write_json(document_id, _LAYOUT_FILENAME, _build_layout(paper))

            for figure in paper.figures:
                image_bytes = figure_images.get(figure.id)
                if image_bytes is not None and figure.asset_uri is not None:
                    self._parsed_storage.write_bytes(document_id, figure.asset_uri, image_bytes)

            for table in paper.tables:
                if table.markdown:
                    self._parsed_storage.write_bytes(
                        document_id, f"tables/{table.id}.md", table.markdown.encode("utf-8")
                    )
        except StorageError as exc:
            raise ParserStorageError(document_id=document_id) from exc


def _build_layout(paper: Paper) -> dict[str, Any]:
    """Build a page-indexed view of every located element's position.

    Complements `paper.json`'s entity-indexed view: a frontend rendering a
    PDF viewer with highlight overlays wants "what's on page N", not to
    reconstruct that by walking every entity in the full document.
    """
    pages: dict[int, list[dict[str, Any]]] = defaultdict(list)

    def add(kind: str, entity_id: object, boxes: list[BoundingBox]) -> None:
        for box in boxes:
            pages[box.page_number].append(
                {
                    "kind": kind,
                    "id": str(entity_id),
                    "bounding_box": {"x0": box.x0, "y0": box.y0, "x1": box.x1, "y1": box.y1},
                }
            )

    for section in paper.sections:
        add("section", section.id, section.bounding_boxes)
    for paragraph in paper.paragraphs:
        add("paragraph", paragraph.id, paragraph.bounding_boxes)
    for figure in paper.figures:
        add("figure", figure.id, figure.bounding_boxes)
    for table in paper.tables:
        add("table", table.id, table.bounding_boxes)
    for caption in paper.captions:
        add("caption", caption.id, caption.bounding_boxes)

    return {
        "document_id": str(paper.id),
        "page_count": paper.metadata.page_count,
        "pages": [
            {"page_number": page_number, "elements": elements}
            for page_number, elements in sorted(pages.items())
        ],
    }
