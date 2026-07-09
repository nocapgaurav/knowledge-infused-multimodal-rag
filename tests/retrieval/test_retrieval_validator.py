"""Tests for RetrievalValidator's structural checks.

Most defects here are unreachable through the normal pipeline (each phase
is correct by construction) -- these tests exercise the validator
directly against hand-crafted, deliberately malformed data, proving each
check actually fires if a future change ever introduces the defect.
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from backend.retrieval.exceptions import (
    BundleConsistencyError,
    DuplicateCandidateError,
    DuplicateEvidenceError,
    GraphCycleError,
    MissingKnowledgeUnitError,
    RankingConsistencyError,
    TraceCompletenessError,
)
from backend.retrieval.models import (
    DiscoveryMethod,
    EvidenceBundle,
    EvidenceGroup,
    GraphPath,
    PhaseTrace,
    RankingExplanation,
    RetrievalCandidate,
    RetrievalManifest,
    RetrievalStatistics,
    RetrievalTrace,
    ScoredCandidate,
    SignalScore,
    TraversalHop,
)
from backend.retrieval.validation.retrieval_validator import RetrievalValidator


def _candidate(document_id, knowledge_unit_id=None, depth=0, hops=None) -> RetrievalCandidate:
    return RetrievalCandidate(
        knowledge_unit_id=knowledge_unit_id or uuid4(),
        document_id=document_id,
        section_id=None,
        modality="text",
        text="text",
        asset_uri=None,
        reading_order=0,
        citation_count=0,
        dense_similarity=0.5 if depth == 0 else None,
        discovery_method=(
            DiscoveryMethod.DENSE_RETRIEVAL if depth == 0 else DiscoveryMethod.GRAPH_EXPANSION
        ),
        graph_path=GraphPath(hops=hops or ()),
    )


def _scored(candidate, rank=1, score=1.0) -> ScoredCandidate:
    return ScoredCandidate(
        candidate=candidate,
        ranking=RankingExplanation(
            signals=(SignalScore(name="dense_similarity", raw_value=score, rank=rank),),
            fused_score=score,
            final_rank=rank,
        ),
    )


def test_validate_candidates_raises_on_duplicate() -> None:
    document_id = uuid4()
    shared_id = uuid4()
    candidates = [_candidate(document_id, shared_id), _candidate(document_id, shared_id)]

    with pytest.raises(DuplicateCandidateError):
        RetrievalValidator().validate_candidates(candidates)


def test_validate_candidates_passes_for_unique_pool() -> None:
    document_id = uuid4()
    candidates = [_candidate(document_id), _candidate(document_id)]

    RetrievalValidator().validate_candidates(candidates)  # should not raise


def test_validate_graph_path_raises_on_cycle() -> None:
    node_a, node_b = str(uuid4()), str(uuid4())
    cyclical_path = GraphPath(
        hops=(
            TraversalHop(
                source_id=node_a, target_id=node_b, relationship_type="NEXT", direction="outgoing"
            ),
            TraversalHop(
                source_id=node_b, target_id=node_a, relationship_type="NEXT", direction="outgoing"
            ),
        )
    )

    with pytest.raises(GraphCycleError):
        RetrievalValidator().validate_graph_path(cyclical_path)


def test_validate_graph_path_passes_for_acyclic_path() -> None:
    node_a, node_b, node_c = str(uuid4()), str(uuid4()), str(uuid4())
    path = GraphPath(
        hops=(
            TraversalHop(
                source_id=node_a, target_id=node_b, relationship_type="NEXT", direction="outgoing"
            ),
            TraversalHop(
                source_id=node_b, target_id=node_c, relationship_type="NEXT", direction="outgoing"
            ),
        )
    )

    RetrievalValidator().validate_graph_path(path)  # should not raise


def test_validate_ranking_raises_on_non_contiguous_ranks() -> None:
    document_id = uuid4()
    scored = [_scored(_candidate(document_id), rank=1), _scored(_candidate(document_id), rank=3)]

    with pytest.raises(RankingConsistencyError):
        RetrievalValidator().validate_ranking(scored)


def test_validate_ranking_raises_when_scores_increase_with_rank() -> None:
    document_id = uuid4()
    scored = [
        _scored(_candidate(document_id), rank=1, score=0.1),
        _scored(_candidate(document_id), rank=2, score=0.9),
    ]

    with pytest.raises(RankingConsistencyError):
        RetrievalValidator().validate_ranking(scored)


def test_validate_evidence_groups_raises_on_duplicate_membership() -> None:
    document_id = uuid4()
    shared = _scored(_candidate(document_id), rank=1)
    other_primary = _scored(_candidate(document_id), rank=2)
    groups = [
        EvidenceGroup(group_id="g1", primary=shared, supporting=(), modalities=("text",)),
        EvidenceGroup(
            group_id="g2", primary=other_primary, supporting=(shared,), modalities=("text",)
        ),
    ]

    with pytest.raises(DuplicateEvidenceError):
        RetrievalValidator().validate_evidence_groups(
            groups, [shared.candidate, other_primary.candidate]
        )


def test_validate_evidence_groups_raises_on_missing_knowledge_unit() -> None:
    document_id = uuid4()
    orphan_primary = _scored(_candidate(document_id), rank=1)
    groups = [
        EvidenceGroup(group_id="g1", primary=orphan_primary, supporting=(), modalities=("text",))
    ]

    with pytest.raises(MissingKnowledgeUnitError):
        RetrievalValidator().validate_evidence_groups(groups, [])


def _bundle(document_id, groups, phases) -> EvidenceBundle:
    evidence_items = sum(1 + len(g.supporting) for g in groups)
    return EvidenceBundle(
        document_id=document_id,
        query="what is the result?",
        candidates=(),
        evidence_groups=tuple(groups),
        trace=RetrievalTrace(phases=tuple(phases), dropped=()),
        manifest=RetrievalManifest(
            document_id=document_id,
            query="what is the result?",
            retrieval_version="1.0",
            retrieval_strategy_version="1.0",
            representation_version="repr",
            embedding_version="emb",
            graph_version="graph",
            statistics=RetrievalStatistics(
                candidates_generated=0,
                candidates_expanded=0,
                candidates_scored=0,
                evidence_groups=len(groups),
                evidence_items=evidence_items,
                duration_ms=1.0,
            ),
            created_at=datetime.now(UTC),
        ),
    )


def _all_phases() -> list[PhaseTrace]:
    return [
        PhaseTrace(phase=name, input_count=0, output_count=0, duration_ms=1.0)
        for name in ("candidate_generation", "expansion", "evaluation", "assembly")
    ]


def test_validate_bundle_raises_on_group_count_mismatch() -> None:
    document_id = uuid4()
    primary = _scored(_candidate(document_id), rank=1)
    group = EvidenceGroup(group_id="g1", primary=primary, supporting=(), modalities=("text",))
    bundle = _bundle(document_id, [group], _all_phases())
    tampered = bundle.model_copy(
        update={
            "manifest": bundle.manifest.model_copy(
                update={
                    "statistics": bundle.manifest.statistics.model_copy(
                        update={"evidence_groups": 99}
                    )
                }
            )
        }
    )

    with pytest.raises(BundleConsistencyError):
        RetrievalValidator().validate_bundle(tampered)


def test_validate_bundle_raises_on_missing_trace_phase() -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, [], _all_phases()[:-1])  # missing "assembly"

    with pytest.raises(TraceCompletenessError):
        RetrievalValidator().validate_bundle(bundle)


def test_validate_bundle_passes_for_consistent_bundle() -> None:
    document_id = uuid4()
    primary = _scored(_candidate(document_id), rank=1)
    group = EvidenceGroup(group_id="g1", primary=primary, supporting=(), modalities=("text",))
    bundle = _bundle(document_id, [group], _all_phases())

    RetrievalValidator().validate_bundle(bundle)  # should not raise
