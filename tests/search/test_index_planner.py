"""Tests for grouping embedding artifacts into per-collection plans."""

from datetime import UTC, datetime
from uuid import uuid4

from backend.domain import ChunkId, PaperId
from backend.embeddings.models import EmbeddingArtifact, EmbeddingId, EmbeddingTarget
from backend.search.models import DistanceMetric
from backend.search.planner.index_planner import IndexPlanner


def _artifact(
    paper_id: PaperId,
    model_name: str = "BAAI/bge-m3",
    model_version: str = "sha-1",
    target: EmbeddingTarget = EmbeddingTarget.TEXT,
    dimension: int = 4,
) -> EmbeddingArtifact:
    return EmbeddingArtifact(
        embedding_id=EmbeddingId(uuid4()),
        knowledge_unit_id=ChunkId(uuid4()),
        paper_id=paper_id,
        target=target,
        vector=[0.1] * dimension,
        model_name=model_name,
        model_version=model_version,
        embedding_dimension=dimension,
        checksum="abc",
        artifact_version="1.0",
        source_representation_version="hash-1",
        created_at=datetime.now(UTC),
    )


def test_artifacts_from_the_same_model_and_target_share_a_plan() -> None:
    paper_id = PaperId(uuid4())
    artifacts = [_artifact(paper_id), _artifact(paper_id)]

    plans = IndexPlanner(collection_prefix="kimrag").plan(artifacts)

    assert len(plans) == 1
    assert len(plans[0].artifacts) == 2
    assert plans[0].distance is DistanceMetric.COSINE


def test_different_targets_produce_separate_plans() -> None:
    paper_id = PaperId(uuid4())
    artifacts = [
        _artifact(paper_id, target=EmbeddingTarget.TEXT),
        _artifact(paper_id, target=EmbeddingTarget.IMAGE),
    ]

    plans = IndexPlanner(collection_prefix="kimrag").plan(artifacts)

    assert len(plans) == 2
    collection_names = {plan.collection_name for plan in plans}
    assert len(collection_names) == 2


def test_different_model_versions_produce_separate_plans() -> None:
    paper_id = PaperId(uuid4())
    artifacts = [
        _artifact(paper_id, model_version="sha-old"),
        _artifact(paper_id, model_version="sha-new"),
    ]

    plans = IndexPlanner(collection_prefix="kimrag").plan(artifacts)

    assert len(plans) == 2


def test_empty_input_produces_no_plans() -> None:
    assert IndexPlanner(collection_prefix="kimrag").plan([]) == []


def test_plan_dimension_matches_artifact_dimension() -> None:
    paper_id = PaperId(uuid4())
    artifacts = [_artifact(paper_id, dimension=1024)]

    plans = IndexPlanner(collection_prefix="kimrag").plan(artifacts)

    assert plans[0].dimension == 1024
