"""Tests for the parser service orchestration, using a fake DocumentParser
and real LocalFilesystemStorage (against tmp_path)."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from backend.domain import PaperId
from backend.ingestion.identifiers import UploadJobId
from backend.ingestion.models import UploadMetadata
from backend.parser.exceptions import (
    DocumentNotIngestedError,
    EmptyDocumentError,
    UnreadablePdfError,
)
from backend.parser.interfaces.document_parser import DocumentParser
from backend.parser.interfaces.extracted_document import (
    ExtractedDocument,
    ExtractedFigure,
    ExtractedTable,
    ExtractedTableCell,
    ExtractedTextBlock,
    ExtractedTextRole,
)
from backend.parser.mapper.domain_mapper import DomainMapper
from backend.parser.services.parser_service import ParserService
from backend.storage.local_filesystem import LocalFilesystemStorage


class _FakeDocumentParser(DocumentParser):
    def __init__(
        self, result: ExtractedDocument | None = None, error: Exception | None = None
    ) -> None:
        self._result = result
        self._error = error

    def parse(self, pdf_bytes: bytes) -> ExtractedDocument:
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def _seed_raw_document(raw_storage: LocalFilesystemStorage, document_id: PaperId) -> None:
    raw_storage.create_workspace(document_id)
    raw_storage.write_bytes(document_id, "paper.pdf", b"%PDF-1.4 fake content")
    metadata = UploadMetadata(
        document_id=document_id,
        upload_job_id=UploadJobId(uuid4()),
        original_filename="paper.pdf",
        content_type="application/pdf",
        size_bytes=21,
        sha256="deadbeef",
        created_at=datetime.now(UTC),
    )
    raw_storage.write_json(document_id, "upload.json", metadata.model_dump(mode="json"))


@pytest.fixture
def storages(tmp_path: Path) -> tuple[LocalFilesystemStorage, LocalFilesystemStorage]:
    raw = LocalFilesystemStorage(root=tmp_path / "raw")
    parsed = LocalFilesystemStorage(root=tmp_path / "parsed")
    return raw, parsed


def _rich_extracted_document() -> ExtractedDocument:
    figure = ExtractedFigure(
        caption_text="Figure 1: An example.",
        caption_bounding_boxes=(),
        image_bytes=b"fake-png-bytes",
        image_format="png",
        bounding_boxes=(),
    )
    table = ExtractedTable(
        cells=(
            ExtractedTableCell(
                row=0, column=0, text="x", row_span=1, column_span=1, is_header=True
            ),
        ),
        num_rows=1,
        num_columns=1,
        markdown="| x |\n|---|",
        caption_text="Table 1: An example.",
        caption_bounding_boxes=(),
        bounding_boxes=(),
    )
    return ExtractedDocument(
        title="A Test Paper",
        page_count=2,
        content=(
            ExtractedTextBlock(
                role=ExtractedTextRole.SECTION_HEADER,
                text="1. Introduction",
                level=1,
                bounding_boxes=(),
            ),
            ExtractedTextBlock(
                role=ExtractedTextRole.PARAGRAPH, text="intro body", level=None, bounding_boxes=()
            ),
            table,
            figure,
        ),
    )


def test_parse_document_persists_all_expected_artifacts(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage], tmp_path: Path
) -> None:
    raw_storage, parsed_storage = storages
    document_id = PaperId(uuid4())
    _seed_raw_document(raw_storage, document_id)
    service = ParserService(
        raw_storage=raw_storage,
        parsed_storage=parsed_storage,
        document_parser=_FakeDocumentParser(result=_rich_extracted_document()),
        mapper=DomainMapper(),
    )

    paper = service.parse_document(document_id)

    workspace = tmp_path / "parsed" / str(document_id)
    paper_json = json.loads((workspace / "paper.json").read_text())
    metadata_json = json.loads((workspace / "metadata.json").read_text())
    layout_json = json.loads((workspace / "layout.json").read_text())

    assert paper_json["id"] == str(document_id)
    assert metadata_json["title"] == "A Test Paper"
    assert layout_json["document_id"] == str(document_id)
    assert layout_json["page_count"] == 2

    figure = paper.figures[0]
    assert (workspace / "figures" / f"{figure.id}.png").read_bytes() == b"fake-png-bytes"

    table = paper.tables[0]
    assert (workspace / "tables" / f"{table.id}.md").read_text() == "| x |\n|---|"


def test_parse_document_raises_for_unknown_document_id(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    raw_storage, parsed_storage = storages
    service = ParserService(
        raw_storage=raw_storage,
        parsed_storage=parsed_storage,
        document_parser=_FakeDocumentParser(result=_rich_extracted_document()),
        mapper=DomainMapper(),
    )

    with pytest.raises(DocumentNotIngestedError):
        service.parse_document(PaperId(uuid4()))


def test_parse_document_propagates_unreadable_pdf_error(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    raw_storage, parsed_storage = storages
    document_id = PaperId(uuid4())
    _seed_raw_document(raw_storage, document_id)
    service = ParserService(
        raw_storage=raw_storage,
        parsed_storage=parsed_storage,
        document_parser=_FakeDocumentParser(error=UnreadablePdfError(reason="corrupt")),
        mapper=DomainMapper(),
    )

    with pytest.raises(UnreadablePdfError):
        service.parse_document(document_id)


def test_parse_document_propagates_validation_failures(
    storages: tuple[LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    raw_storage, parsed_storage = storages
    document_id = PaperId(uuid4())
    _seed_raw_document(raw_storage, document_id)
    empty_document = ExtractedDocument(title="A Title With No Content", page_count=1, content=())
    service = ParserService(
        raw_storage=raw_storage,
        parsed_storage=parsed_storage,
        document_parser=_FakeDocumentParser(result=empty_document),
        mapper=DomainMapper(),
    )

    with pytest.raises(EmptyDocumentError):
        service.parse_document(document_id)
