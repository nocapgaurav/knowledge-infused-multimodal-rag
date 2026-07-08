"""Structural validation for uploaded files.

Every check here is content-agnostic: extension, declared content type,
size, and the presence of a PDF magic header. None of it reads or
interprets the document's actual content -- that boundary belongs to the
parser (Module 4).
"""

from pathlib import PurePosixPath

from backend.ingestion.exceptions import (
    EmptyFileError,
    FileTooLargeError,
    InvalidPdfContentError,
    UnsupportedFileTypeError,
)

_ALLOWED_EXTENSIONS = frozenset({".pdf"})
_ALLOWED_CONTENT_TYPES = frozenset({"application/pdf"})
_PDF_MAGIC_HEADER = b"%PDF-"


def validate_upload(
    *, filename: str, content_type: str, content: bytes, max_size_bytes: int
) -> None:
    """Validate that an uploaded file is a plausible, in-limits PDF.

    Args:
        filename: Filename as supplied by the client.
        content_type: Content type as declared by the client.
        content: Raw file content.
        max_size_bytes: Maximum permitted file size, in bytes.

    Raises:
        UnsupportedFileTypeError: The extension or declared content type is not PDF.
        EmptyFileError: The file contains no bytes.
        FileTooLargeError: The file exceeds `max_size_bytes`.
        InvalidPdfContentError: The file's content does not start with a PDF header.
    """
    extension = PurePosixPath(filename).suffix.lower()
    if extension not in _ALLOWED_EXTENSIONS or content_type not in _ALLOWED_CONTENT_TYPES:
        raise UnsupportedFileTypeError(filename=filename, content_type=content_type)

    if not content:
        raise EmptyFileError(filename=filename)

    if len(content) > max_size_bytes:
        raise FileTooLargeError(
            filename=filename, size_bytes=len(content), max_size_bytes=max_size_bytes
        )

    if not content.startswith(_PDF_MAGIC_HEADER):
        raise InvalidPdfContentError(filename=filename)
