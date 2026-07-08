"""End-to-end tests for the document ingestion API."""

from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.dependencies import get_workspace_storage
from backend.storage.local_filesystem import LocalFilesystemStorage

_VALID_PDF_CONTENT = b"%PDF-1.4\n...minimal pdf content..."


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_workspace_storage] = lambda: LocalFilesystemStorage(
        root=tmp_path / "raw"
    )
    with TestClient(app) as test_client:
        yield test_client


def test_upload_document_returns_201_with_the_expected_shape(client: TestClient) -> None:
    response = client.post(
        "/documents", files={"file": ("paper.pdf", _VALID_PDF_CONTENT, "application/pdf")}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "UPLOADED"
    assert "document_id" in body
    assert "upload_job_id" in body


def test_upload_document_rejects_a_non_pdf_file(client: TestClient) -> None:
    response = client.post("/documents", files={"file": ("notes.txt", b"hello", "text/plain")})

    assert response.status_code == 415


def test_get_document_status_after_upload(client: TestClient) -> None:
    upload_response = client.post(
        "/documents", files={"file": ("paper.pdf", _VALID_PDF_CONTENT, "application/pdf")}
    )
    document_id = upload_response.json()["document_id"]

    status_response = client.get(f"/documents/{document_id}")

    assert status_response.status_code == 200
    assert status_response.json()["status"] == "UPLOADED"


def test_get_document_status_for_unknown_id_returns_404(client: TestClient) -> None:
    response = client.get(f"/documents/{uuid4()}")

    assert response.status_code == 404
