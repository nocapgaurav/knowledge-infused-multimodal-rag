"""Tests for bounding-box-based reading order reconstruction."""

from uuid import uuid4

from backend.chunking.builder.reading_order import compute_reading_order
from backend.domain import BoundingBox, Figure, Metadata, Paper, PaperId, Paragraph, Section, Table


def _box(page: int = 1, y0: float = 0.0) -> BoundingBox:
    return BoundingBox(page_number=page, x0=0.0, y0=y0, x1=100.0, y1=y0 + 10.0)


def _metadata() -> Metadata:
    return Metadata(title="A Paper", source_filename="p.pdf")


def test_interleaves_figure_between_paragraphs_by_vertical_position() -> None:
    paper_id = PaperId(uuid4())
    section = Section(paper_id=paper_id, title="Results", level=1, order=0)
    para_before = Paragraph(
        paper_id=paper_id,
        section_id=section.id,
        order=0,
        text="before",
        bounding_boxes=[_box(y0=10)],
    )
    figure = Figure(paper_id=paper_id, section_id=section.id, order=0, bounding_boxes=[_box(y0=20)])
    para_after = Paragraph(
        paper_id=paper_id,
        section_id=section.id,
        order=1,
        text="after",
        bounding_boxes=[_box(y0=30)],
    )
    # Deliberately constructed out of visual order to prove sorting, not insertion order, wins.
    paper = Paper(
        id=paper_id,
        metadata=_metadata(),
        sections=[section],
        paragraphs=[para_after, para_before],
        figures=[figure],
    )

    ordered = compute_reading_order(paper)

    assert [o.item.id for o in ordered] == [para_before.id, figure.id, para_after.id]


def test_depth_first_section_order_visits_children_before_next_sibling() -> None:
    paper_id = PaperId(uuid4())
    intro = Section(paper_id=paper_id, title="Introduction", level=1, order=0)
    background = Section(
        paper_id=paper_id, parent_section_id=intro.id, title="Background", level=2, order=0
    )
    method = Section(paper_id=paper_id, title="Method", level=1, order=1)
    intro_para = Paragraph(paper_id=paper_id, section_id=intro.id, order=0, text="intro")
    background_para = Paragraph(paper_id=paper_id, section_id=background.id, order=0, text="bg")
    method_para = Paragraph(paper_id=paper_id, section_id=method.id, order=0, text="method")
    paper = Paper(
        id=paper_id,
        metadata=_metadata(),
        sections=[method, background, intro],  # shuffled on purpose
        paragraphs=[method_para, background_para, intro_para],
    )

    ordered = compute_reading_order(paper)

    assert [o.item.id for o in ordered] == [intro_para.id, background_para.id, method_para.id]


def test_items_with_no_bounding_box_fall_back_to_paragraphs_then_figures_then_tables() -> None:
    paper_id = PaperId(uuid4())
    section = Section(paper_id=paper_id, title="Results", level=1, order=0)
    table = Table(paper_id=paper_id, section_id=section.id, order=0, num_rows=1, num_columns=1)
    figure = Figure(paper_id=paper_id, section_id=section.id, order=0)
    paragraph = Paragraph(paper_id=paper_id, section_id=section.id, order=0, text="text")
    paper = Paper(
        id=paper_id,
        metadata=_metadata(),
        sections=[section],
        paragraphs=[paragraph],
        figures=[figure],
        tables=[table],
    )

    ordered = compute_reading_order(paper)

    assert [o.item.id for o in ordered] == [paragraph.id, figure.id, table.id]


def test_items_with_no_section_come_before_real_sections() -> None:
    paper_id = PaperId(uuid4())
    section = Section(paper_id=paper_id, title="Introduction", level=1, order=0)
    unsectioned = Paragraph(paper_id=paper_id, section_id=None, order=0, text="author line")
    sectioned = Paragraph(paper_id=paper_id, section_id=section.id, order=0, text="body")
    paper = Paper(
        id=paper_id,
        metadata=_metadata(),
        sections=[section],
        paragraphs=[sectioned, unsectioned],
    )

    ordered = compute_reading_order(paper)

    assert [o.item.id for o in ordered] == [unsectioned.id, sectioned.id]
