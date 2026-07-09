"""Tests for paragraph splitting, merging, and relationship detection."""

from uuid import uuid4

from backend.chunking.interfaces.context import BuildContext
from backend.chunking.strategies.paragraph_strategy import ParagraphStrategy
from backend.domain import (
    ChunkId,
    FigureId,
    Metadata,
    Paper,
    PaperId,
    Paragraph,
    ReferenceId,
    RelationshipType,
)


def _paper(paper_id: PaperId) -> Paper:
    return Paper(id=paper_id, metadata=Metadata(title="A Paper", source_filename="p.pdf"))


def _context(paper: Paper, max_words: int = 250, min_words: int = 4) -> BuildContext:
    return BuildContext(
        paper=paper,
        max_words_per_chunk=max_words,
        min_words_per_chunk=min_words,
        figure_number_lookup={},
        table_number_lookup={},
        reference_number_lookup={},
    )


def test_normal_paragraphs_become_one_chunk_each() -> None:
    paper_id = PaperId(uuid4())
    paper = _paper(paper_id)
    paragraphs = [
        Paragraph(paper_id=paper_id, order=0, text="This is a perfectly normal paragraph of text."),
        Paragraph(paper_id=paper_id, order=1, text="This is another perfectly normal paragraph."),
    ]

    result = ParagraphStrategy().build(paragraphs, _context(paper))

    assert len(result.chunks) == 2
    assert result.chunks[0].order == 0
    assert result.chunks[1].order == 1


def test_oversized_paragraph_is_split_at_sentence_boundaries_with_overlap() -> None:
    paper_id = PaperId(uuid4())
    paper = _paper(paper_id)
    sentences = [f"This is sentence number {i} in a very long paragraph." for i in range(20)]
    paragraph = Paragraph(paper_id=paper_id, order=0, text=" ".join(sentences))

    result = ParagraphStrategy().build([paragraph], _context(paper, max_words=30))

    assert len(result.chunks) > 1
    # never cut mid-sentence: every chunk's text must end with one of our sentences intact
    for chunk in result.chunks:
        assert chunk.text.rstrip().endswith(".")
    # one-sentence overlap: chunk N and chunk N+1 share at least one sentence
    first_chunk_sentences = {s.strip() for s in result.chunks[0].text.split(".") if s.strip()}
    second_chunk_sentences = {s.strip() for s in result.chunks[1].text.split(".") if s.strip()}
    assert first_chunk_sentences & second_chunk_sentences
    # split siblings are linked
    continues = [
        r for r in result.relationships if r.relationship_type is RelationshipType.CONTINUES
    ]
    assert len(continues) == len(result.chunks) - 1


def test_tiny_paragraph_merges_forward_into_next() -> None:
    paper_id = PaperId(uuid4())
    paper = _paper(paper_id)
    tiny = Paragraph(paper_id=paper_id, order=0, text="Too short.")
    normal = Paragraph(
        paper_id=paper_id, order=1, text="This paragraph is long enough to stand on its own."
    )

    result = ParagraphStrategy().build([tiny, normal], _context(paper, min_words=4))

    assert len(result.chunks) == 1
    assert "Too short." in result.chunks[0].text
    assert "long enough" in result.chunks[0].text


def test_tiny_trailing_paragraph_merges_backward() -> None:
    paper_id = PaperId(uuid4())
    paper = _paper(paper_id)
    normal = Paragraph(
        paper_id=paper_id, order=0, text="This paragraph is long enough to stand on its own."
    )
    tiny = Paragraph(paper_id=paper_id, order=1, text="Too short.")

    result = ParagraphStrategy().build([normal, tiny], _context(paper, min_words=4))

    assert len(result.chunks) == 1
    assert "Too short." in result.chunks[0].text


def test_section_of_only_tiny_paragraphs_is_kept_as_one_block() -> None:
    paper_id = PaperId(uuid4())
    paper = _paper(paper_id)
    tiny_one = Paragraph(paper_id=paper_id, order=0, text="Too short.")
    tiny_two = Paragraph(paper_id=paper_id, order=1, text="Also tiny.")

    result = ParagraphStrategy().build([tiny_one, tiny_two], _context(paper, min_words=4))

    assert len(result.chunks) == 1


def test_empty_input_produces_no_chunks() -> None:
    paper_id = PaperId(uuid4())
    paper = _paper(paper_id)

    result = ParagraphStrategy().build([], _context(paper))

    assert result.chunks == []
    assert result.relationships == []


def test_detects_citation_and_resolves_to_registered_chunk() -> None:
    paper_id = PaperId(uuid4())
    paper = _paper(paper_id)
    reference_id = ReferenceId(uuid4())
    context = _context(paper)
    context.reference_number_lookup[1] = reference_id
    target_chunk_id = ChunkId(uuid4())
    context.register_chunk(reference_id, target_chunk_id)

    paragraph = Paragraph(paper_id=paper_id, order=0, text="This was shown previously [1].")
    chunk = ParagraphStrategy().build([paragraph], context).chunks[0]

    relationships = ParagraphStrategy().detect_relationships(chunk, context)

    assert len(relationships) == 1
    assert relationships[0].relationship_type is RelationshipType.CITES
    assert relationships[0].target_chunk_id == target_chunk_id
    assert relationships[0].source_chunk_id == chunk.id


def test_detects_figure_mention_and_resolves_to_registered_chunk() -> None:
    paper_id = PaperId(uuid4())
    paper = _paper(paper_id)
    figure_id = FigureId(uuid4())
    context = _context(paper)
    context.figure_number_lookup[1] = figure_id
    target_chunk_id = ChunkId(uuid4())
    context.register_chunk(figure_id, target_chunk_id)

    paragraph = Paragraph(paper_id=paper_id, order=0, text="As shown in Figure 1, results improve.")
    chunk = ParagraphStrategy().build([paragraph], context).chunks[0]

    relationships = ParagraphStrategy().detect_relationships(chunk, context)

    assert len(relationships) == 1
    assert relationships[0].relationship_type is RelationshipType.REFERENCES
    assert relationships[0].target_chunk_id == target_chunk_id


def test_unresolvable_citation_produces_no_relationship() -> None:
    paper_id = PaperId(uuid4())
    paper = _paper(paper_id)
    context = _context(paper)  # no lookups registered at all

    paragraph = Paragraph(paper_id=paper_id, order=0, text="This was shown previously [99].")
    chunk = ParagraphStrategy().build([paragraph], context).chunks[0]

    relationships = ParagraphStrategy().detect_relationships(chunk, context)

    assert relationships == []


def test_duplicate_mentions_of_the_same_figure_produce_one_relationship() -> None:
    paper_id = PaperId(uuid4())
    paper = _paper(paper_id)
    figure_id = FigureId(uuid4())
    context = _context(paper)
    context.figure_number_lookup[1] = figure_id
    context.register_chunk(figure_id, ChunkId(uuid4()))

    paragraph = Paragraph(
        paper_id=paper_id, order=0, text="Figure 1 shows this. Figure 1 also shows that."
    )
    chunk = ParagraphStrategy().build([paragraph], context).chunks[0]

    relationships = ParagraphStrategy().detect_relationships(chunk, context)

    assert len(relationships) == 1
