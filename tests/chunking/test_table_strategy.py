"""Tests for building a knowledge unit from a table."""

from uuid import uuid4

from backend.chunking.interfaces.context import BuildContext
from backend.chunking.strategies.table_strategy import TableStrategy
from backend.domain import (
    Caption,
    CaptionSubjectType,
    ChunkModality,
    Metadata,
    Paper,
    PaperId,
    Table,
)


def _context(paper: Paper) -> BuildContext:
    return BuildContext(
        paper=paper,
        max_words_per_chunk=250,
        min_words_per_chunk=4,
        figure_number_lookup={},
        table_number_lookup={},
        reference_number_lookup={},
    )


def test_table_with_caption_and_markdown_combines_both() -> None:
    paper_id = PaperId(uuid4())
    table = Table(paper_id=paper_id, order=0, num_rows=1, num_columns=1, markdown="| a |\n|---|")
    caption = Caption(
        paper_id=paper_id,
        subject_type=CaptionSubjectType.TABLE,
        subject_id=table.id,
        text="Table 1: Results.",
    )
    paper = Paper(
        id=paper_id,
        metadata=Metadata(title="A Paper", source_filename="p.pdf"),
        tables=[table],
        captions=[caption],
    )

    result = TableStrategy().build(table, _context(paper))

    chunk = result.chunks[0]
    assert chunk.modality is ChunkModality.TABLE
    assert "Table 1: Results." in chunk.text
    assert "| a |" in chunk.text
    assert table.id in chunk.source_element_ids
    assert caption.id in chunk.source_element_ids


def test_table_without_caption_falls_back_to_markdown() -> None:
    paper_id = PaperId(uuid4())
    table = Table(paper_id=paper_id, order=0, num_rows=1, num_columns=1, markdown="| a |\n|---|")
    paper = Paper(
        id=paper_id, metadata=Metadata(title="A Paper", source_filename="p.pdf"), tables=[table]
    )

    result = TableStrategy().build(table, _context(paper))

    assert result.chunks[0].text == "| a |\n|---|"


def test_table_without_caption_or_markdown_gets_placeholder_text() -> None:
    paper_id = PaperId(uuid4())
    table = Table(paper_id=paper_id, order=0, num_rows=0, num_columns=0)
    paper = Paper(
        id=paper_id, metadata=Metadata(title="A Paper", source_filename="p.pdf"), tables=[table]
    )

    result = TableStrategy().build(table, _context(paper))

    assert result.chunks[0].text == "Untitled table"
