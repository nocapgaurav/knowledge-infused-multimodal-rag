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
    text="text",
    retrieval_context=None,
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
        text=text,
        retrieval_context=retrieval_context,
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


def test_lexical_overlap_promotes_exact_structural_match() -> None:
    document_id = uuid4()
    # Same dense similarity; only the lexical signal separates them.
    table = _candidate(
        document_id,
        dense_similarity=0.5,
        text="COMPARISON OF EXISTING APPROACHES",
        retrieval_context="Table 1",
    )
    prose = _candidate(document_id, dense_similarity=0.5, text="LLMs improved substantially.")

    ranked = EvidenceEvaluator().evaluate([prose, table], query="What is Table 1?")

    assert ranked[0].candidate.knowledge_unit_id == table.knowledge_unit_id


def test_lexical_overlap_counts_retrieval_context_terms() -> None:
    document_id = uuid4()
    title = _candidate(
        document_id,
        dense_similarity=0.4,
        text="Knowledge-Infused Multimodal Retrieval",
        retrieval_context="Title of this paper",
    )
    prose = _candidate(document_id, dense_similarity=0.6, text="Scientific publication grows.")

    ranked = EvidenceEvaluator().evaluate([prose, title], query="What is the title?")

    lexical = {
        s.name: s.rank
        for s in next(
            r for r in ranked if r.candidate.knowledge_unit_id == title.knowledge_unit_id
        ).ranking.signals
    }
    assert lexical["lexical_overlap"] == 1


def test_empty_query_keeps_lexical_signal_inert() -> None:
    document_id = uuid4()
    a = _candidate(document_id, dense_similarity=0.9, text="alpha")
    b = _candidate(document_id, dense_similarity=0.1, text="beta")

    ranked = EvidenceEvaluator().evaluate([a, b])

    assert ranked[0].candidate.knowledge_unit_id == a.knowledge_unit_id


def test_query_independent_priors_do_not_bury_the_best_dense_match() -> None:
    """Regression for the Sprint 1 failure: a section-less #1 dense match
    (e.g. the keywords block) must not be outranked by topically vague
    chunks from popular, frequently-cited sections."""
    document_id = uuid4()
    section = uuid4()
    target = _candidate(
        document_id,
        section_id=None,
        dense_similarity=0.55,
        citation_count=0,
        text="Index Terms -Multimodal Learning, Question Answering",
        retrieval_context="Keywords (index terms)",
    )
    hubs = [
        _candidate(
            document_id,
            section_id=section,
            dense_similarity=0.45 - i * 0.001,
            citation_count=3,
            text="A generally related hub paragraph.",
        )
        for i in range(6)
    ]

    ranked = EvidenceEvaluator().evaluate([*hubs, target], query="What are the keywords?")

    assert ranked[0].candidate.knowledge_unit_id == target.knowledge_unit_id
