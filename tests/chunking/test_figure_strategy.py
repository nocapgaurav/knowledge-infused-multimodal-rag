"""Tests for building a knowledge unit from a figure."""

from uuid import uuid4

from backend.chunking.interfaces.context import BuildContext
from backend.chunking.strategies.figure_strategy import FigureStrategy
from backend.domain import (
    Caption,
    CaptionSubjectType,
    ChunkModality,
    Figure,
    Metadata,
    Paper,
    PaperId,
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


def test_figure_with_caption_uses_caption_as_text() -> None:
    paper_id = PaperId(uuid4())
    figure = Figure(paper_id=paper_id, order=0, asset_uri="figures/x.png")
    caption = Caption(
        paper_id=paper_id,
        subject_type=CaptionSubjectType.FIGURE,
        subject_id=figure.id,
        text="Figure 1: An example chart.",
    )
    paper = Paper(
        id=paper_id,
        metadata=Metadata(title="A Paper", source_filename="p.pdf"),
        figures=[figure],
        captions=[caption],
    )

    result = FigureStrategy().build(figure, _context(paper))

    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert chunk.modality is ChunkModality.FIGURE
    assert chunk.text == "Figure 1: An example chart."
    assert chunk.asset_uri == "figures/x.png"
    assert figure.id in chunk.source_element_ids
    assert caption.id in chunk.source_element_ids


def test_figure_without_caption_gets_placeholder_text() -> None:
    paper_id = PaperId(uuid4())
    figure = Figure(paper_id=paper_id, order=0)
    paper = Paper(
        id=paper_id, metadata=Metadata(title="A Paper", source_filename="p.pdf"), figures=[figure]
    )

    result = FigureStrategy().build(figure, _context(paper))

    assert result.chunks[0].text == "Untitled figure"


def test_figure_chunk_is_registered_in_context() -> None:
    paper_id = PaperId(uuid4())
    figure = Figure(paper_id=paper_id, order=0)
    paper = Paper(
        id=paper_id, metadata=Metadata(title="A Paper", source_filename="p.pdf"), figures=[figure]
    )
    context = _context(paper)

    result = FigureStrategy().build(figure, context)

    assert context.entity_chunk_ids[figure.id] == result.chunks[0].id
