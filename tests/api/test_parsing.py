"""End-to-end tests for the document parsing API.

Uses a fake `DocumentParser` (not real Docling) -- this test verifies
routing, dependency wiring, and status-code mapping, which is a distinct
concern from Docling correctness (covered by
`tests/parser/test_docling_provider.py`).
"""

from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.dependencies import get_document_parser, get_parsed_storage, get_workspace_storage
from backend.parser.exceptions import UnreadablePdfError
from backend.parser.interfaces.document_parser import DocumentParser
from backend.parser.interfaces.extracted_document import (
    ExtractedDocument,
    ExtractedTextBlock,
    ExtractedTextRole,
)
from backend.storage.local_filesystem import LocalFilesystemStorage

_VALID_PDF_CONTENT = b"%PDF-1.4\n...minimal pdf content..."


class _FakeDocumentParser(DocumentParser):
    def __init__(self, should_fail: bool = False) -> None:
        self._should_fail = should_fail

    def parse(self, pdf_bytes: bytes) -> ExtractedDocument:
        if self._should_fail:
            raise UnreadablePdfError(reason="simulated failure")
        return ExtractedDocument(
            title="A Test Paper",
            page_count=1,
            content=(
                ExtractedTextBlock(
                    role=ExtractedTextRole.SECTION_HEADER,
                    text="1. Introduction",
                    level=1,
                    bounding_boxes=(),
                ),
                ExtractedTextBlock(
                    role=ExtractedTextRole.PARAGRAPH,
                    text="body text",
                    level=None,
                    bounding_boxes=(),
                ),
            ),
        )


def _make_client(tmp_path: Path, *, parser_should_fail: bool = False) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_workspace_storage] = lambda: LocalFilesystemStorage(
        root=tmp_path / "raw"
    )
    app.dependency_overrides[get_parsed_storage] = lambda: LocalFilesystemStorage(
        root=tmp_path / "parsed"
    )
    app.dependency_overrides[get_document_parser] = lambda: _FakeDocumentParser(
        should_fail=parser_should_fail
    )
    return TestClient(app)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    with _make_client(tmp_path) as test_client:
        yield test_client


def test_parse_document_returns_parsed_status(client: TestClient) -> None:
    upload_response = client.post(
        "/documents", files={"file": ("paper.pdf", _VALID_PDF_CONTENT, "application/pdf")}
    )
    document_id = upload_response.json()["document_id"]

    response = client.post(f"/documents/{document_id}/parse")

    assert response.status_code == 200
    assert response.json() == {"document_id": document_id, "status": "PARSED"}


def test_parse_document_returns_404_for_unknown_document(client: TestClient) -> None:
    response = client.post(f"/documents/{uuid4()}/parse")

    assert response.status_code == 404


def test_parse_document_returns_422_when_pdf_is_unreadable(tmp_path: Path) -> None:
    with _make_client(tmp_path, parser_should_fail=True) as client:
        upload_response = client.post(
            "/documents", files={"file": ("paper.pdf", _VALID_PDF_CONTENT, "application/pdf")}
        )
        document_id = upload_response.json()["document_id"]

        response = client.post(f"/documents/{document_id}/parse")

        assert response.status_code == 422
