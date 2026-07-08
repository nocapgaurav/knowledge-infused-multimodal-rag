"""Tests for structural validation of a mapped Paper."""

from uuid import uuid4

import pytest

from backend.domain import (
    Caption,
    CaptionSubjectType,
    Figure,
    FigureId,
    Metadata,
    Paper,
    PaperId,
    Paragraph,
    Section,
    SectionId,
    Table,
)
from backend.parser.exceptions import (
    EmptyDocumentError,
    InvalidSectionHierarchyError,
    MissingFigureReferenceError,
)
from backend.parser.validator.document_validator import validate_document


def _metadata() -> Metadata:
    return Metadata(title="A Paper", source_filename="p.pdf")


def test_valid_document_passes(caplog: pytest.LogCaptureFixture) -> None:
    paper_id = PaperId(uuid4())
    figure = Figure(paper_id=paper_id, order=0)
    caption = Caption(
        paper_id=paper_id,
        subject_type=CaptionSubjectType.FIGURE,
        subject_id=figure.id,
        text="Figure 1.",
    )
    paper = Paper(
        id=paper_id,
        metadata=_metadata(),
        paragraphs=[Paragraph(paper_id=paper_id, order=0, text="body")],
        figures=[figure],
        captions=[caption],
    )

    validate_document(paper)  # should not raise


def test_section_with_unknown_parent_raises() -> None:
    paper_id = PaperId(uuid4())
    orphan = Section(
        paper_id=paper_id,
        parent_section_id=SectionId(uuid4()),
        title="Orphan",
        level=2,
        order=0,
    )
    paper = Paper(id=paper_id, metadata=_metadata(), sections=[orphan])

    with pytest.raises(InvalidSectionHierarchyError):
        validate_document(paper)


def test_section_cycle_raises() -> None:
    paper_id = PaperId(uuid4())
    a = Section(paper_id=paper_id, title="A", level=1, order=0)
    b = Section(paper_id=paper_id, parent_section_id=a.id, title="B", level=2, order=0)
    # Manually break the tree into a cycle: A's parent becomes B.
    a_with_cycle = a.model_copy(update={"parent_section_id": b.id})
    paper = Paper(id=paper_id, metadata=_metadata(), sections=[a_with_cycle, b])

    with pytest.raises(InvalidSectionHierarchyError):
        validate_document(paper)


def test_caption_referencing_unknown_figure_raises() -> None:
    paper_id = PaperId(uuid4())
    dangling_caption = Caption(
        paper_id=paper_id,
        subject_type=CaptionSubjectType.FIGURE,
        subject_id=FigureId(uuid4()),
        text="Figure 1.",
    )
    paper = Paper(
        id=paper_id,
        metadata=_metadata(),
        paragraphs=[Paragraph(paper_id=paper_id, order=0, text="body")],
        captions=[dangling_caption],
    )

    with pytest.raises(MissingFigureReferenceError):
        validate_document(paper)


def test_uncaptioned_figure_logs_warning_but_does_not_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    paper_id = PaperId(uuid4())
    figure = Figure(paper_id=paper_id, order=0)
    paper = Paper(
        id=paper_id,
        metadata=_metadata(),
        paragraphs=[Paragraph(paper_id=paper_id, order=0, text="body")],
        figures=[figure],
    )

    with caplog.at_level("WARNING"):
        validate_document(paper)  # should not raise

    assert any("no caption" in record.message for record in caplog.records)


def test_document_with_only_a_table_is_not_considered_empty() -> None:
    paper_id = PaperId(uuid4())
    table = Table(paper_id=paper_id, order=0, num_rows=1, num_columns=1)
    paper = Paper(id=paper_id, metadata=_metadata(), tables=[table])

    validate_document(paper)  # should not raise


def test_document_with_no_content_at_all_raises_empty_document_error() -> None:
    paper_id = PaperId(uuid4())
    paper = Paper(id=paper_id, metadata=_metadata())

    with pytest.raises(EmptyDocumentError):
        validate_document(paper)
