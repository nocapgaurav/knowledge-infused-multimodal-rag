"""Tests for Phase 3: evidence evaluation (RRF ranking)."""

from uuid import uuid4

from backend.retrieval.evaluation.evidence_evaluator import EvidenceEvaluator
from backend.retrieval.models import DiscoveryMethod, GraphPath, RetrievalCandidate, TraversalHop


def _candidate(
    document_id,
    section_id=None,
    dense_similarity=None,
    citation_count=0,
    depth=0,
    relationship_type="NEXT",
) -> RetrievalCandidate:
    hops = (
        ()
        if depth == 0
        else tuple(
            TraversalHop(
                source_id=str(uuid4()),
                target_id=str(uuid4()),
                relationship_type=relationship_type,
                direction="outgoing",
            )
            for _ in range(depth)
        )
    )
    return RetrievalCandidate(
        knowledge_unit_id=uuid4(),
        document_id=document_id,
        section_id=section_id,
        modality="text",
        text="text",
        asset_uri=None,
        reading_order=0,
        citation_count=citation_count,
        dense_similarity=dense_similarity,
        discovery_method=(
            DiscoveryMethod.DENSE_RETRIEVAL if depth == 0 else DiscoveryMethod.GRAPH_EXPANSION
        ),
        graph_path=GraphPath(hops=hops),
    )


def test_evaluate_empty_pool_returns_empty() -> None:
    assert EvidenceEvaluator().evaluate([]) == []


def test_higher_dense_similarity_ranks_better_all_else_equal() -> None:
    document_id = uuid4()
    strong = _candidate(document_id, dense_similarity=0.9)
    weak = _candidate(document_id, dense_similarity=0.1)

    scored = EvidenceEvaluator().evaluate([weak, strong])

    assert scored[0].candidate.knowledge_unit_id == strong.knowledge_unit_id
    assert scored[0].ranking.final_rank == 1
    assert scored[1].ranking.final_rank == 2


def test_dense_match_outranks_graph_discovered_candidate_on_relationship_confidence() -> None:
    document_id = uuid4()
    dense = _candidate(document_id, dense_similarity=0.5, depth=0)
    graph_found = _candidate(
        document_id, dense_similarity=None, depth=1, relationship_type="BELONGS_TO"
    )

    scored = EvidenceEvaluator().evaluate([graph_found, dense])

    confidence_signal = {
        s.name: s
        for s in next(
            sc.ranking.signals
            for sc in scored
            if sc.candidate.knowledge_unit_id == dense.knowledge_unit_id
        )
    }
    assert confidence_signal["relationship_confidence"].raw_value == 4.0


def test_citation_importance_orders_by_citation_count() -> None:
    document_id = uuid4()
    highly_cited = _candidate(document_id, citation_count=10)
    uncited = _candidate(document_id, citation_count=0)

    scored = EvidenceEvaluator().evaluate([uncited, highly_cited])

    ranks = {sc.candidate.knowledge_unit_id: sc.ranking.final_rank for sc in scored}
    assert ranks[highly_cited.knowledge_unit_id] < ranks[uncited.knowledge_unit_id]


def test_section_relevance_rewards_co_occurring_candidates() -> None:
    document_id = uuid4()
    shared_section = uuid4()
    a = _candidate(document_id, section_id=shared_section)
    b = _candidate(document_id, section_id=shared_section)
    alone = _candidate(document_id, section_id=uuid4())

    scored = EvidenceEvaluator().evaluate([a, b, alone])

    section_signal = {
        sc.candidate.knowledge_unit_id: next(
            s.raw_value for s in sc.ranking.signals if s.name == "section_relevance"
        )
        for sc in scored
    }
    assert section_signal[a.knowledge_unit_id] == 1.0
    assert section_signal[b.knowledge_unit_id] == 1.0
    assert section_signal[alone.knowledge_unit_id] == 0.0


def test_ranks_form_a_contiguous_sequence() -> None:
    document_id = uuid4()
    candidates = [_candidate(document_id, dense_similarity=i / 10) for i in range(5)]

    scored = EvidenceEvaluator().evaluate(candidates)

    assert [sc.ranking.final_rank for sc in scored] == [1, 2, 3, 4, 5]
    scores = [sc.ranking.fused_score for sc in scored]
    assert scores == sorted(scores, reverse=True)


def test_evaluation_is_deterministic_across_repeated_calls() -> None:
    document_id = uuid4()
    candidates = [
        _candidate(document_id, dense_similarity=0.7, citation_count=2),
        _candidate(document_id, dense_similarity=0.3, citation_count=5),
    ]

    first = EvidenceEvaluator().evaluate(candidates)
    second = EvidenceEvaluator().evaluate(candidates)

    assert [sc.ranking.fused_score for sc in first] == [sc.ranking.fused_score for sc in second]
    assert [sc.candidate.knowledge_unit_id for sc in first] == [
        sc.candidate.knowledge_unit_id for sc in second
    ]
