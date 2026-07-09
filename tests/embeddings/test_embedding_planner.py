"""Tests for embedding target planning."""

from uuid import uuid4

from backend.domain import Chunk as DomainChunk
from backend.domain import ChunkModality, PaperId
from backend.embeddings.models import EmbeddingTarget
from backend.embeddings.planner.embedding_planner import EmbeddingPlanner


def _chunk(paper_id: PaperId, order: int, asset_uri: str | None = None) -> DomainChunk:
    return DomainChunk(
        paper_id=paper_id,
        order=order,
        modality=ChunkModality.TEXT,
        text="some text",
        asset_uri=asset_uri,
    )


def test_text_only_chunk_gets_one_task() -> None:
    paper_id = PaperId(uuid4())
    chunk = _chunk(paper_id, 0)

    tasks = EmbeddingPlanner().plan([chunk])

    assert len(tasks) == 1
    assert tasks[0].target is EmbeddingTarget.TEXT
    assert tasks[0].chunk.id == chunk.id


def test_chunk_with_asset_uri_gets_text_and_image_tasks() -> None:
    paper_id = PaperId(uuid4())
    chunk = _chunk(paper_id, 0, asset_uri="figures/x.png")

    tasks = EmbeddingPlanner().plan([chunk])

    targets = {task.target for task in tasks}
    assert targets == {EmbeddingTarget.TEXT, EmbeddingTarget.IMAGE}
    assert len(tasks) == 2


def test_plan_is_content_driven_not_modality_hardcoded() -> None:
    """A TABLE-modality chunk with an asset_uri should still get an image
    task -- the rule generalizes beyond FIGURE, since it's driven by
    `asset_uri`, not by `modality`."""
    paper_id = PaperId(uuid4())
    chunk = DomainChunk(
        paper_id=paper_id,
        order=0,
        modality=ChunkModality.TABLE,
        text="| a |\n|---|",
        asset_uri="tables/x.png",
    )

    tasks = EmbeddingPlanner().plan([chunk])

    assert {task.target for task in tasks} == {EmbeddingTarget.TEXT, EmbeddingTarget.IMAGE}


def test_empty_input_produces_no_tasks() -> None:
    assert EmbeddingPlanner().plan([]) == []


def test_multiple_chunks_each_produce_their_own_tasks() -> None:
    paper_id = PaperId(uuid4())
    chunks = [_chunk(paper_id, i) for i in range(5)]

    tasks = EmbeddingPlanner().plan(chunks)

    assert len(tasks) == 5
