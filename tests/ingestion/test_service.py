"""Tests for the document ingestion service."""

from pathlib import Path
from uuid import uuid4

import pytest

from backend.domain import PaperId
from backend.ingestion.exceptions import UnsupportedFileTypeError, UploadJobNotFoundError
from backend.ingestion.models import UploadStatus
from backend.ingestion.service import DocumentIngestionService
from backend.storage.local_filesystem import LocalFilesystemStorage

_VALID_PDF_CONTENT = b"%PDF-1.4\n...minimal pdf content..."


@pytest.fixture
def service(tmp_path: Path) -> DocumentIngestionService:
    storage = LocalFilesystemStorage(root=tmp_path / "raw")
    return DocumentIngestionService(storage=storage, max_upload_size_bytes=1_000)


def test_ingest_creates_a_workspace_and_persists_all_files(
    service: DocumentIngestionService, tmp_path: Path
) -> None:
    job = service.ingest(
        filename="paper.pdf", content_type="application/pdf", content=_VALID_PDF_CONTENT
    )

    workspace = tmp_path / "raw" / str(job.document_id)
    assert (workspace / "paper.pdf").read_bytes() == _VALID_PDF_CONTENT
    assert (workspace / "upload.json").exists()
    assert (workspace / "status.json").exists()


def test_ingest_returns_an_uploaded_job(service: DocumentIngestionService) -> None:
    job = service.ingest(
        filename="paper.pdf", content_type="application/pdf", content=_VALID_PDF_CONTENT
    )

    assert job.status is UploadStatus.UPLOADED
    assert job.error_message is None


def test_ingest_rejects_an_invalid_file_and_creates_no_workspace(
    service: DocumentIngestionService, tmp_path: Path
) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        service.ingest(filename="notes.txt", content_type="text/plain", content=b"hello")

    assert list((tmp_path / "raw").iterdir()) == []


def test_get_status_returns_the_persisted_job(service: DocumentIngestionService) -> None:
    job = service.ingest(
        filename="paper.pdf", content_type="application/pdf", content=_VALID_PDF_CONTENT
    )

    fetched = service.get_status(job.document_id)

    assert fetched == job


def test_get_status_raises_for_an_unknown_document(service: DocumentIngestionService) -> None:
    with pytest.raises(UploadJobNotFoundError):
        service.get_status(PaperId(uuid4()))
