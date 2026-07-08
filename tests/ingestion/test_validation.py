"""Tests for structural upload validation."""

import pytest

from backend.ingestion.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    InvalidPdfContentError,
    UnsupportedFileTypeError,
)
from backend.ingestion.validation import validate_upload

_VALID_PDF_CONTENT = b"%PDF-1.4\n...minimal pdf content..."


def test_accepts_a_valid_pdf() -> None:
    validate_upload(
        filename="paper.pdf",
        content_type="application/pdf",
        content=_VALID_PDF_CONTENT,
        max_size_bytes=1_000,
    )


def test_rejects_a_non_pdf_extension() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        validate_upload(
            filename="notes.txt",
            content_type="application/pdf",
            content=_VALID_PDF_CONTENT,
            max_size_bytes=1_000,
        )


def test_rejects_a_non_pdf_content_type() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        validate_upload(
            filename="paper.pdf",
            content_type="text/plain",
            content=_VALID_PDF_CONTENT,
            max_size_bytes=1_000,
        )


def test_rejects_an_empty_file() -> None:
    with pytest.raises(EmptyFileError):
        validate_upload(
            filename="paper.pdf",
            content_type="application/pdf",
            content=b"",
            max_size_bytes=1_000,
        )


def test_rejects_a_file_over_the_size_limit() -> None:
    with pytest.raises(FileTooLargeError):
        validate_upload(
            filename="paper.pdf",
            content_type="application/pdf",
            content=_VALID_PDF_CONTENT,
            max_size_bytes=5,
        )


def test_rejects_content_without_a_pdf_header() -> None:
    with pytest.raises(InvalidPdfContentError):
        validate_upload(
            filename="paper.pdf",
            content_type="application/pdf",
            content=b"this is not actually a pdf",
            max_size_bytes=1_000,
        )
