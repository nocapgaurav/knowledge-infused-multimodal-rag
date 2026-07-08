"""Round-trip JSON serialization tests for every domain entity."""

from datetime import date
from uuid import uuid4

from backend.domain import (
    Answer,
    Author,
    BoundingBox,
    Caption,
    CaptionSubjectType,
    Chunk,
    ChunkId,
    ChunkModality,
    Evidence,
    Figure,
    Metadata,
    Paper,
    PaperId,
    Paragraph,
    Query,
    Reference,
    Section,
    Table,
    TableCell,
)


def _build_paper() -> Paper:
    paper_id = PaperId(uuid4())
    box = BoundingBox(page_number=1, x0=0, y0=0, x1=10, y1=10)

    metadata = Metadata(
        title="Knowledge-Infused Multimodal Question Answering",
        authors=[Author(name="Ada Researcher", order=0)],
        abstract="An abstract.",
        publication_date=date(2024, 1, 1),
        source_filename="paper.pdf",
        page_count=10,
    )
    section = Section(
        paper_id=paper_id, title="Introduction", level=1, order=0, bounding_boxes=[box]
    )
    paragraph = Paragraph(
        paper_id=paper_id,
        section_id=section.id,
        order=0,
        text="Body text.",
        bounding_boxes=[box],
    )
    figure = Figure(
        paper_id=paper_id,
        section_id=section.id,
        order=0,
        label="Figure 1",
        bounding_boxes=[box],
    )
    table = Table(
        paper_id=paper_id,
        section_id=section.id,
        order=0,
        label="Table 1",
        num_rows=1,
        num_columns=1,
        cells=[TableCell(row=0, column=0, text="value")],
        bounding_boxes=[box],
    )
    caption = Caption(
        paper_id=paper_id,
        subject_type=CaptionSubjectType.FIGURE,
        subject_id=figure.id,
        text="Figure 1: an example.",
        bounding_boxes=[box],
    )
    reference = Reference(paper_id=paper_id, order=0, raw_text="Smith, J. (2020). A paper.")

    return Paper(
        id=paper_id,
        metadata=metadata,
        sections=[section],
        paragraphs=[paragraph],
        figures=[figure],
        tables=[table],
        captions=[caption],
        references=[reference],
    )


def test_paper_round_trips_through_json() -> None:
    paper = _build_paper()

    restored = Paper.model_validate_json(paper.model_dump_json())

    assert restored == paper


def test_chunk_round_trips_through_json() -> None:
    chunk = Chunk(
        paper_id=PaperId(uuid4()),
        order=0,
        modality=ChunkModality.TEXT,
        text="Body text.",
        source_element_ids=[uuid4()],
    )

    restored = Chunk.model_validate_json(chunk.model_dump_json())

    assert restored == chunk


def test_query_answer_evidence_round_trip_through_json() -> None:
    query = Query(text="What method did the authors use?", paper_ids=[PaperId(uuid4())])
    evidence = Evidence(
        paper_id=PaperId(uuid4()),
        chunk_id=ChunkId(uuid4()),
        quoted_text="We used method X.",
        relevance_score=0.87,
    )
    answer = Answer(query_id=query.id, text="They used method X.", evidence=[evidence])

    assert Query.model_validate_json(query.model_dump_json()) == query
    assert Answer.model_validate_json(answer.model_dump_json()) == answer
