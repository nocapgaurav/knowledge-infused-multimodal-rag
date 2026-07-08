"""Exceptions raised by the document ingestion pipeline."""

from backend.domain import PaperId


class IngestionError(Exception):
    """Base class for all document ingestion errors."""


class UnsupportedFileTypeError(IngestionError):
    """Raised when an uploaded file's extension or content type is not PDF."""

    def __init__(self, *, filename: str, content_type: str) -> None:
        self.filename = filename
        self.content_type = content_type
        super().__init__(f"unsupported file type for '{filename}' ({content_type})")


class EmptyFileError(IngestionError):
    """Raised when an uploaded file contains no bytes."""

    def __init__(self, *, filename: str) -> None:
        self.filename = filename
        super().__init__(f"uploaded file '{filename}' is empty")


class FileTooLargeError(IngestionError):
    """Raised when an uploaded file exceeds the configured size limit."""

    def __init__(self, *, filename: str, size_bytes: int, max_size_bytes: int) -> None:
        self.filename = filename
        self.size_bytes = size_bytes
        self.max_size_bytes = max_size_bytes
        super().__init__(
            f"file '{filename}' is {size_bytes} bytes, exceeding the "
            f"{max_size_bytes} byte limit"
        )


class InvalidPdfContentError(IngestionError):
    """Raised when an uploaded file's content does not start with a PDF header."""

    def __init__(self, *, filename: str) -> None:
        self.filename = filename
        super().__init__(f"file '{filename}' does not contain valid PDF content")


class UploadJobNotFoundError(IngestionError):
    """Raised when no upload job exists for a given document id."""

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(f"no upload job found for document {document_id}")


class IngestionStorageError(IngestionError):
    """Raised when a storage failure prevents a document from being ingested."""

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(f"a storage error occurred while ingesting document {document_id}")
