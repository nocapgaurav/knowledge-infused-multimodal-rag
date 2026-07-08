"""Tests for domain-level invariants enforced by the domain layer itself."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.domain import Answer, BoundingBox, Metadata, Paper, PaperId, Query, Section


def test_bounding_box_rejects_an_inverted_rectangle() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(page_number=1, x0=10, y0=0, x1=0, y1=10)


def test_paper_rejects_a_child_from_a_different_paper() -> None:
    paper_id = PaperId(uuid4())
    other_paper_id = PaperId(uuid4())
    metadata = Metadata(title="Title", source_filename="paper.pdf")
    misattached_section = Section(paper_id=other_paper_id, title="Introduction", level=1, order=0)

    with pytest.raises(ValidationError):
        Paper(id=paper_id, metadata=metadata, sections=[misattached_section])


def test_answer_requires_at_least_one_piece_of_evidence() -> None:
    query = Query(text="What method did the authors use?")

    with pytest.raises(ValidationError):
        Answer(query_id=query.id, text="They used method X.", evidence=[])
