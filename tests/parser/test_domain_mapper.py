"""Tests for ExtractedDocument -> Paper mapping."""

from uuid import uuid4

import pytest

from backend.domain import CaptionSubjectType, PaperId
from backend.parser.exceptions import MissingRequiredMetadataError
from backend.parser.interfaces.extracted_document import (
    ExtractedDocument,
    ExtractedFigure,
    ExtractedTable,
    ExtractedTableCell,
    ExtractedTextBlock,
    ExtractedTextRole,
)
from backend.parser.mapper.domain_mapper import DomainMapper

HEADER = ExtractedTextRole.SECTION_HEADER
PARAGRAPH = ExtractedTextRole.PARAGRAPH


def _text(role: ExtractedTextRole, text: str, level: int | None = None) -> ExtractedTextBlock:
    return ExtractedTextBlock(role=role, text=text, level=level, bounding_boxes=())


def _document(*content, title: str | None = "A Test Paper") -> ExtractedDocument:
    return ExtractedDocument(title=title, page_count=3, content=tuple(content))


@pytest.fixture
def mapper() -> DomainMapper:
    return DomainMapper()


def test_raises_when_no_title_is_available(mapper: DomainMapper) -> None:
    document = _document(_text(PARAGRAPH, "body text"), title=None)

    with pytest.raises(MissingRequiredMetadataError):
        mapper.to_paper(document_id=PaperId(uuid4()), source_filename="p.pdf", extracted=document)


def test_maps_title_and_page_count(mapper: DomainMapper) -> None:
    document = _document(_text(HEADER, "1. Introduction", level=1), _text(PARAGRAPH, "body"))

    result = mapper.to_paper(
        document_id=PaperId(uuid4()), source_filename="p.pdf", extracted=document
    )

    assert result.paper.metadata.title == "A Test Paper"
    assert result.paper.metadata.page_count == 3
    assert result.paper.metadata.source_filename == "p.pdf"


def test_reconstructs_section_hierarchy_from_heading_levels(mapper: DomainMapper) -> None:
    document = _document(
        _text(HEADER, "1. Introduction", level=1),
        _text(PARAGRAPH, "intro body"),
        _text(HEADER, "1.1 Background", level=2),
        _text(PARAGRAPH, "background body"),
        _text(HEADER, "2. Method", level=1),
        _text(PARAGRAPH, "method body"),
    )

    result = mapper.to_paper(
        document_id=PaperId(uuid4()), source_filename="p.pdf", extracted=document
    )
    paper = result.paper

    by_title = {section.title: section for section in paper.sections}
    assert by_title["1. Introduction"].parent_section_id is None
    assert by_title["1.1 Background"].parent_section_id == by_title["1. Introduction"].id
    assert by_title["2. Method"].parent_section_id is None

    background_paragraph = next(p for p in paper.paragraphs if p.text == "background body")
    assert background_paragraph.section_id == by_title["1.1 Background"].id


def test_abstract_section_becomes_metadata_not_a_section(mapper: DomainMapper) -> None:
    document = _document(
        _text(HEADER, "Abstract", level=1),
        _text(PARAGRAPH, "This paper studies things."),
        _text(HEADER, "1. Introduction", level=1),
        _text(PARAGRAPH, "intro body"),
    )

    result = mapper.to_paper(
        document_id=PaperId(uuid4()), source_filename="p.pdf", extracted=document
    )
    paper = result.paper

    assert paper.metadata.abstract == "This paper studies things."
    assert all(section.title != "Abstract" for section in paper.sections)
    assert all(p.text != "This paper studies things." for p in paper.paragraphs)


def test_references_section_splits_bracket_numbered_entries(mapper: DomainMapper) -> None:
    document = _document(
        _text(HEADER, "References", level=1),
        _text(PARAGRAPH, "[1] Smith, J. (2020). A paper. [2] Doe, A. (2021). Another paper."),
    )

    result = mapper.to_paper(
        document_id=PaperId(uuid4()), source_filename="p.pdf", extracted=document
    )
    paper = result.paper

    assert len(paper.references) == 2
    assert paper.references[0].raw_text.startswith("[1] Smith")
    assert paper.references[1].raw_text.startswith("[2] Doe")
    assert paper.references[0].order == 0
    assert paper.references[1].order == 1
    assert all(section.title != "References" for section in paper.sections)


def test_references_section_falls_back_to_one_entry_per_block(mapper: DomainMapper) -> None:
    document = _document(
        _text(HEADER, "References", level=1),
        _text(PARAGRAPH, "Smith, J. (2020). A paper without brackets."),
        _text(PARAGRAPH, "Doe, A. (2021). Another paper without brackets."),
    )

    result = mapper.to_paper(
        document_id=PaperId(uuid4()), source_filename="p.pdf", extracted=document
    )

    assert len(result.paper.references) == 2


def test_maps_table_with_caption(mapper: DomainMapper) -> None:
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
    document = _document(_text(HEADER, "1. Results", level=1), table)

    result = mapper.to_paper(
        document_id=PaperId(uuid4()), source_filename="p.pdf", extracted=document
    )
    paper = result.paper

    assert len(paper.tables) == 1
    assert paper.tables[0].markdown == "| x |\n|---|"
    caption = next(c for c in paper.captions if c.subject_id == paper.tables[0].id)
    assert caption.subject_type is CaptionSubjectType.TABLE
    assert caption.text == "Table 1: An example."


def test_maps_figure_with_image_bytes(mapper: DomainMapper) -> None:
    figure = ExtractedFigure(
        caption_text="Figure 1: A chart.",
        caption_bounding_boxes=(),
        image_bytes=b"fake-png-bytes",
        image_format="png",
        bounding_boxes=(),
    )
    document = _document(_text(HEADER, "1. Results", level=1), figure)

    result = mapper.to_paper(
        document_id=PaperId(uuid4()), source_filename="p.pdf", extracted=document
    )
    paper = result.paper

    assert len(paper.figures) == 1
    mapped_figure = paper.figures[0]
    assert mapped_figure.asset_uri == f"figures/{mapped_figure.id}.png"
    assert result.figure_images[mapped_figure.id] == b"fake-png-bytes"


def test_figure_without_image_has_no_asset_uri(mapper: DomainMapper) -> None:
    figure = ExtractedFigure(
        caption_text=None,
        caption_bounding_boxes=(),
        image_bytes=None,
        image_format=None,
        bounding_boxes=(),
    )
    document = _document(figure)

    result = mapper.to_paper(
        document_id=PaperId(uuid4()), source_filename="p.pdf", extracted=document
    )

    assert result.paper.figures[0].asset_uri is None
    assert result.figure_images == {}
