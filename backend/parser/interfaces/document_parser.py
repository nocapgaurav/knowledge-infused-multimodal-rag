"""The document parsing port: what any parsing engine must provide."""

from abc import ABC, abstractmethod

from backend.parser.interfaces.extracted_document import ExtractedDocument


class DocumentParser(ABC):
    """Converts raw PDF bytes into a provider-agnostic `ExtractedDocument`.

    Concrete implementations (e.g. a Docling-based provider) may use any
    underlying library; nothing about that choice may appear in the
    `ExtractedDocument` this method returns.
    """

    @abstractmethod
    def parse(self, pdf_bytes: bytes) -> ExtractedDocument:
        """Parse a PDF's raw bytes into its structural content.

        Args:
            pdf_bytes: Raw bytes of the PDF to parse.

        Returns:
            The document's provider-agnostic extracted structure.

        Raises:
            UnreadablePdfError: The bytes could not be parsed as a PDF.
        """
