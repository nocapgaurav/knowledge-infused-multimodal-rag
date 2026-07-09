"""Integration tests for the full knowledge builder pipeline."""

from uuid import uuid4

from backend.chunking.builder.knowledge_builder import KnowledgeBuilder
from backend.domain import (
    Caption,
    CaptionSubjectType,
    ChunkModality,
    Figure,
    Metadata,
    Paper,
    PaperId,
    Paragraph,
    Reference,
    RelationshipType,
    Section,
    Table,
)


def _builder() -> KnowledgeBuilder:
    return KnowledgeBuilder(max_words_per_chunk=250, min_words_per_chunk=4)


def _full_paper() -> Paper:
    paper_id = PaperId(uuid4())
    intro = Section(paper_id=paper_id, title="1. Introduction", level=1, order=0)
    method = Section(paper_id=paper_id, title="2. Method", level=1, order=1)
    sub = Section(
        paper_id=paper_id, parent_section_id=method.id, title="2.1 Details", level=2, order=0
    )

    intro_para = Paragraph(
        paper_id=paper_id,
        section_id=intro.id,
        order=0,
        text="This finding was established previously [1].",
    )
    method_para = Paragraph(
        paper_id=paper_id,
        section_id=sub.id,
        order=0,
        text="See Table 1 and Figure 1 for a full comparison.",
    )

    table = Table(
        paper_id=paper_id,
        section_id=sub.id,
        order=0,
        num_rows=1,
        num_columns=1,
        markdown="| x |\n|---|",
    )
    table_caption = Caption(
        paper_id=paper_id,
        subject_type=CaptionSubjectType.TABLE,
        subject_id=table.id,
        text="Table 1: Results.",
    )
    figure = Figure(paper_id=paper_id, section_id=sub.id, order=0)
    figure_caption = Caption(
        paper_id=paper_id,
        subject_type=CaptionSubjectType.FIGURE,
        subject_id=figure.id,
        text="Figure 1: A chart.",
    )
    reference = Reference(paper_id=paper_id, order=0, raw_text="[1] Smith, J. (2020). A paper.")

    return Paper(
        id=paper_id,
        metadata=Metadata(
            title="A Full Paper", source_filename="p.pdf", abstract="This is the abstract."
        ),
        sections=[intro, method, sub],
        paragraphs=[intro_para, method_para],
        tables=[table],
        figures=[figure],
        captions=[table_caption, figure_caption],
        references=[reference],
    )


def test_abstract_is_the_first_chunk() -> None:
    result = _builder().build(_full_paper())

    first = min(result.chunks, key=lambda c: c.order)
    assert first.text == "This is the abstract."
    assert first.order == 0


def test_order_is_globally_unique_and_contiguous() -> None:
    result = _builder().build(_full_paper())

    orders = sorted(chunk.order for chunk in result.chunks)
    assert orders == list(range(len(result.chunks)))


def test_reference_chunk_has_the_highest_order() -> None:
    paper = _full_paper()
    result = _builder().build(paper)

    reference_chunk = next(
        c for c in result.chunks if c.source_element_ids == [paper.references[0].id]
    )
    assert reference_chunk.order == max(c.order for c in result.chunks)


def test_nested_section_id_is_preserved_on_its_paragraph() -> None:
    paper = _full_paper()
    sub_section = paper.sections[2]
    result = _builder().build(paper)

    method_chunk = next(c for c in result.chunks if "Table 1 and Figure 1" in c.text)
    assert method_chunk.section_id == sub_section.id


def test_citation_relationship_links_paragraph_to_reference() -> None:
    paper = _full_paper()
    result = _builder().build(paper)

    intro_chunk = next(c for c in result.chunks if "established previously" in c.text)
    reference_chunk = next(
        c for c in result.chunks if c.source_element_ids == [paper.references[0].id]
    )

    cites = [r for r in result.relationships if r.relationship_type is RelationshipType.CITES]
    assert any(
        r.source_chunk_id == intro_chunk.id and r.target_chunk_id == reference_chunk.id
        for r in cites
    )


def test_reference_mentions_link_paragraph_to_table_and_figure() -> None:
    paper = _full_paper()
    result = _builder().build(paper)

    method_chunk = next(c for c in result.chunks if "Table 1 and Figure 1" in c.text)
    table_chunk = next(c for c in result.chunks if c.modality is ChunkModality.TABLE)
    figure_chunk = next(c for c in result.chunks if c.modality is ChunkModality.FIGURE)

    references = {
        r.target_chunk_id
        for r in result.relationships
        if r.relationship_type is RelationshipType.REFERENCES
        and r.source_chunk_id == method_chunk.id
    }
    assert references == {table_chunk.id, figure_chunk.id}


def test_figure_heavy_paper_with_no_tables_builds_correctly() -> None:
    paper_id = PaperId(uuid4())
    section = Section(paper_id=paper_id, title="Results", level=1, order=0)
    figures = [Figure(paper_id=paper_id, section_id=section.id, order=i) for i in range(10)]
    paper = Paper(
        id=paper_id,
        metadata=Metadata(title="Figure Heavy", source_filename="p.pdf"),
        sections=[section],
        figures=figures,
    )

    result = _builder().build(paper)

    figure_chunks = [c for c in result.chunks if c.modality is ChunkModality.FIGURE]
    table_chunks = [c for c in result.chunks if c.modality is ChunkModality.TABLE]
    assert len(figure_chunks) == 10
    assert table_chunks == []


def test_table_heavy_paper_with_no_figures_builds_correctly() -> None:
    paper_id = PaperId(uuid4())
    section = Section(paper_id=paper_id, title="Results", level=1, order=0)
    tables = [
        Table(paper_id=paper_id, section_id=section.id, order=i, num_rows=1, num_columns=1)
        for i in range(10)
    ]
    paper = Paper(
        id=paper_id,
        metadata=Metadata(title="Table Heavy", source_filename="p.pdf"),
        sections=[section],
        tables=tables,
    )

    result = _builder().build(paper)

    table_chunks = [c for c in result.chunks if c.modality is ChunkModality.TABLE]
    figure_chunks = [c for c in result.chunks if c.modality is ChunkModality.FIGURE]
    assert len(table_chunks) == 10
    assert figure_chunks == []


def test_paper_with_only_paragraphs_builds_without_error() -> None:
    paper_id = PaperId(uuid4())
    section = Section(paper_id=paper_id, title="Body", level=1, order=0)
    paragraphs = [
        Paragraph(
            paper_id=paper_id,
            section_id=section.id,
            order=i,
            text=f"This is paragraph number {i}, long enough to avoid tiny-paragraph merging.",
        )
        for i in range(5)
    ]
    paper = Paper(
        id=paper_id,
        metadata=Metadata(title="Text Only", source_filename="p.pdf"),
        sections=[section],
        paragraphs=paragraphs,
    )

    result = _builder().build(paper)

    assert len(result.chunks) == 5
    assert all(c.modality is ChunkModality.TEXT for c in result.chunks)


def test_deeply_nested_sections_resolve_correct_section_id() -> None:
    paper_id = PaperId(uuid4())
    level1 = Section(paper_id=paper_id, title="1", level=1, order=0)
    level2 = Section(paper_id=paper_id, parent_section_id=level1.id, title="1.1", level=2, order=0)
    level3 = Section(
        paper_id=paper_id, parent_section_id=level2.id, title="1.1.1", level=3, order=0
    )
    paragraph = Paragraph(
        paper_id=paper_id, section_id=level3.id, order=0, text="Deeply nested text."
    )
    paper = Paper(
        id=paper_id,
        metadata=Metadata(title="Nested", source_filename="p.pdf"),
        sections=[level1, level2, level3],
        paragraphs=[paragraph],
    )

    result = _builder().build(paper)

    chunk = next(c for c in result.chunks if c.text == "Deeply nested text.")
    assert chunk.section_id == level3.id
