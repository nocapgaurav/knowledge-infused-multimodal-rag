"""Exceptions raised by the document parsing engine."""

from backend.domain import CaptionId, PaperId


class ParsingError(Exception):
    """Base class for all document parsing errors."""


class DocumentNotIngestedError(ParsingError):
    """Raised when no ingested document exists for a given document id."""

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(f"no ingested document found for {document_id}")


class UnreadablePdfError(ParsingError):
    """Raised when a parser could not read PDF content from raw bytes.

    Raised by a `DocumentParser` implementation, which parses raw bytes and
    has no notion of document identity -- callers that know the document id
    are expected to attach it (e.g. via logging) when catching this.
    """

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"content could not be parsed as a PDF: {reason}")


class DocumentValidationError(ParsingError):
    """Base class for structural defects found in a freshly parsed document."""


class EmptyDocumentError(DocumentValidationError):
    """Raised when a parsed document has no textual content at all."""

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(f"document {document_id} has no extractable content")


class MissingRequiredMetadataError(DocumentValidationError):
    """Raised when a required metadata field could not be determined."""

    def __init__(self, *, document_id: PaperId, field: str) -> None:
        self.document_id = document_id
        self.field = field
        super().__init__(f"document {document_id} is missing required metadata: {field}")


class InvalidSectionHierarchyError(DocumentValidationError):
    """Raised when a document's section hierarchy is broken (dangling parent or cycle)."""

    def __init__(self, *, document_id: PaperId, reason: str) -> None:
        self.document_id = document_id
        self.reason = reason
        super().__init__(f"document {document_id} has an invalid section hierarchy: {reason}")


class MissingFigureReferenceError(DocumentValidationError):
    """Raised when a caption references a figure or table that does not exist."""

    def __init__(self, *, document_id: PaperId, caption_id: CaptionId) -> None:
        self.document_id = document_id
        self.caption_id = caption_id
        super().__init__(
            f"document {document_id} has caption {caption_id} referencing a "
            "figure or table that does not exist"
        )


class ParserStorageError(ParsingError):
    """Raised when a storage failure prevents parsed artifacts from being persisted."""

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(f"a storage error occurred while persisting parsed document {document_id}")
