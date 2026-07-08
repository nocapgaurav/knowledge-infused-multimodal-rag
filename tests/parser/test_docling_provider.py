"""Integration tests for the real Docling-based parser provider.

Unlike the mapper and service tests (which use hand-built fixtures or a
fake provider), this exercises actual Docling output against a real PDF --
the only test in this suite that verifies the anti-corruption layer holds
up against the real library, not an assumption about its shape. It is
slower than the rest of the suite (Docling loads layout and table-structure
models on first use); the parser fixture is module-scoped so that cost is
paid once, not once per test.
"""

from pathlib import Path
from uuid import uuid4

import pytest

from backend.domain import CaptionSubjectType, PaperId
from backend.parser.exceptions import UnreadablePdfError
from backend.parser.interfaces.extracted_document import (
    ExtractedFigure,
    ExtractedTable,
    ExtractedTextBlock,
    ExtractedTextRole,
)
from backend.parser.mapper.domain_mapper import DomainMapper
from backend.parser.providers.docling_parser import DoclingDocumentParser
from backend.parser.validator.document_validator import validate_document

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_paper.pdf"


@pytest.fixture(scope="module")
def parser() -> DoclingDocumentParser:
    return DoclingDocumentParser(ocr_enabled=False)


@pytest.fixture(scope="module")
def sample_pdf_bytes() -> bytes:
    return _FIXTURE_PATH.read_bytes()


def test_parses_sample_paper_into_expected_structure(
    parser: DoclingDocumentParser, sample_pdf_bytes: bytes
) -> None:
    extracted = parser.parse(sample_pdf_bytes)

    assert extracted.page_count == 2
    assert extracted.title is not None

    headers = [
        item
        for item in extracted.content
        if isinstance(item, ExtractedTextBlock) and item.role is ExtractedTextRole.SECTION_HEADER
    ]
    paragraphs = [
        item
        for item in extracted.content
        if isinstance(item, ExtractedTextBlock) and item.role is ExtractedTextRole.PARAGRAPH
    ]
    tables = [item for item in extracted.content if isinstance(item, ExtractedTable)]
    figures = [item for item in extracted.content if isinstance(item, ExtractedFigure)]

    assert any("Introduction" in header.text for header in headers)
    assert len(paragraphs) > 0
    assert len(tables) == 1
    assert len(figures) == 1


def test_extracts_table_with_resolved_caption(
    parser: DoclingDocumentParser, sample_pdf_bytes: bytes
) -> None:
    extracted = parser.parse(sample_pdf_bytes)
    table = next(item for item in extracted.content if isinstance(item, ExtractedTable))

    assert table.caption_text is not None
    assert "Table 1" in table.caption_text
    assert table.num_rows == 3
    assert table.num_columns == 3
    assert table.markdown is not None


def test_extracts_figure_with_image_bytes_and_caption(
    parser: DoclingDocumentParser, sample_pdf_bytes: bytes
) -> None:
    extracted = parser.parse(sample_pdf_bytes)
    figure = next(item for item in extracted.content if isinstance(item, ExtractedFigure))

    assert figure.caption_text is not None
    assert "Figure 1" in figure.caption_text
    assert figure.image_bytes is not None
    assert figure.image_format == "png"


def test_bounding_boxes_are_normalized_to_top_left_origin(
    parser: DoclingDocumentParser, sample_pdf_bytes: bytes
) -> None:
    extracted = parser.parse(sample_pdf_bytes)
    headers = [
        item
        for item in extracted.content
        if isinstance(item, ExtractedTextBlock) and item.role is ExtractedTextRole.SECTION_HEADER
    ]

    for header in headers:
        for box in header.bounding_boxes:
            assert box.y0 < box.y1  # top-left origin: top edge numerically above bottom edge


def test_raises_unreadable_pdf_error_for_garbage_bytes(parser: DoclingDocumentParser) -> None:
    with pytest.raises(UnreadablePdfError):
        parser.parse(b"this is not a pdf at all")


def test_full_pipeline_parses_and_validates_the_sample_paper(
    parser: DoclingDocumentParser, sample_pdf_bytes: bytes
) -> None:
    """Provider -> mapper -> validator, chained end to end on a real PDF."""
    extracted = parser.parse(sample_pdf_bytes)
    result = DomainMapper().to_paper(
        document_id=PaperId(uuid4()), source_filename="sample_paper.pdf", extracted=extracted
    )

    validate_document(result.paper)  # should not raise

    paper = result.paper
    assert paper.metadata.abstract is not None
    assert len(paper.references) == 2
    assert any(c.subject_type is CaptionSubjectType.TABLE for c in paper.captions)
    assert any(c.subject_type is CaptionSubjectType.FIGURE for c in paper.captions)
