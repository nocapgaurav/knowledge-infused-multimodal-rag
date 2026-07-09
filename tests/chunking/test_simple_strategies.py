"""Tests for the abstract and reference strategies."""

from uuid import uuid4

from backend.chunking.interfaces.context import BuildContext
from backend.chunking.strategies.simple_strategies import AbstractStrategy, ReferenceStrategy
from backend.domain import Metadata, Paper, PaperId, Reference


def _context(paper: Paper) -> BuildContext:
    return BuildContext(
        paper=paper,
        max_words_per_chunk=250,
        min_words_per_chunk=4,
        figure_number_lookup={},
        table_number_lookup={},
        reference_number_lookup={},
    )


def test_abstract_becomes_a_chunk_when_present() -> None:
    paper_id = PaperId(uuid4())
    paper = Paper(
        id=paper_id,
        metadata=Metadata(
            title="A Paper", source_filename="p.pdf", abstract="This is the abstract."
        ),
    )

    result = AbstractStrategy().build(_context(paper))

    assert len(result.chunks) == 1
    assert result.chunks[0].text == "This is the abstract."
    assert result.chunks[0].section_id is None


def test_missing_abstract_produces_no_chunk() -> None:
    paper_id = PaperId(uuid4())
    paper = Paper(id=paper_id, metadata=Metadata(title="A Paper", source_filename="p.pdf"))

    result = AbstractStrategy().build(_context(paper))

    assert result.chunks == []


def test_reference_becomes_a_chunk_and_registers_in_context() -> None:
    paper_id = PaperId(uuid4())
    reference = Reference(paper_id=paper_id, order=0, raw_text="[1] Smith, J. (2020).")
    paper = Paper(id=paper_id, metadata=Metadata(title="A Paper", source_filename="p.pdf"))
    context = _context(paper)

    result = ReferenceStrategy().build(reference, context)

    assert result.chunks[0].text == "[1] Smith, J. (2020)."
    assert context.entity_chunk_ids[reference.id] == result.chunks[0].id
