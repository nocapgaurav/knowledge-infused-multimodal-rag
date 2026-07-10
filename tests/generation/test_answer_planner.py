"""Tests for Phase 2: answer planning."""

from datetime import UTC, datetime
from uuid import uuid4

from backend.generation.models.answer_plan import ExpectedAnswerType, QuestionType
from backend.generation.planner.answer_planner import AnswerPlanner
from backend.retrieval.models import (
    DiscoveryMethod,
    EvidenceGroup,
    RankingExplanation,
    RetrievalCandidate,
    RetrievalManifest,
    RetrievalStatistics,
    RetrievalTrace,
    ScoredCandidate,
    SignalScore,
)
from backend.retrieval.models.evidence_bundle import EvidenceBundle


def _candidate(document_id, modality="text", discovery_method=DiscoveryMethod.DENSE_RETRIEVAL):
    return RetrievalCandidate(
        knowledge_unit_id=uuid4(),
        document_id=document_id,
        section_id=uuid4(),
        modality=modality,
        text="some evidence text",
        asset_uri=None,
        reading_order=0,
        citation_count=0,
        dense_similarity=0.8 if discovery_method is DiscoveryMethod.DENSE_RETRIEVAL else None,
        discovery_method=discovery_method,
    )


def _group(candidate, rank=1) -> EvidenceGroup:
    scored = ScoredCandidate(
        candidate=candidate,
        ranking=RankingExplanation(
            signals=(SignalScore(name="dense_similarity", raw_value=0.8, rank=rank),),
            fused_score=1.0 / rank,
            final_rank=rank,
        ),
    )
    return EvidenceGroup(
        group_id=str(candidate.knowledge_unit_id),
        primary=scored,
        supporting=(),
        modalities=(candidate.modality,),
    )


def _bundle(document_id, query, groups) -> EvidenceBundle:
    candidates = tuple(group.primary.candidate for group in groups)
    return EvidenceBundle(
        document_id=document_id,
        query=query,
        candidates=candidates,
        evidence_groups=tuple(groups),
        trace=RetrievalTrace(phases=(), dropped=()),
        manifest=RetrievalManifest(
            document_id=document_id,
            query=query,
            retrieval_version="1.0",
            retrieval_strategy_version="1.0",
            representation_version="repr",
            embedding_version="embed",
            graph_version="graph",
            statistics=RetrievalStatistics(
                candidates_generated=len(candidates),
                candidates_expanded=0,
                candidates_scored=len(candidates),
                evidence_groups=len(groups),
                evidence_items=len(groups),
                duration_ms=1.0,
            ),
            created_at=datetime.now(UTC),
        ),
    )


def test_comparative_marker_classified_as_comparative() -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, "compare X and Y", [_group(_candidate(document_id))])

    plan = AnswerPlanner().plan("Compare X and Y in terms of accuracy.", bundle)

    assert plan.question_type is QuestionType.COMPARATIVE
    assert plan.expected_answer_type is ExpectedAnswerType.STRUCTURED_COMPARISON
    assert plan.required_evidence_groups == 2


def test_procedural_marker_classified_as_procedural() -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, "q", [_group(_candidate(document_id))])

    plan = AnswerPlanner().plan("How to reproduce this experiment?", bundle)

    assert plan.question_type is QuestionType.PROCEDURAL


def test_citation_marker_classified_as_citation_centric() -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, "q", [_group(_candidate(document_id))])

    plan = AnswerPlanner().plan("Which paper first proposed this method?", bundle)

    assert plan.question_type is QuestionType.CITATION_CENTRIC


def test_figure_marker_classified_as_figure_centric() -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, "q", [_group(_candidate(document_id))])

    plan = AnswerPlanner().plan("What does Figure 2 show?", bundle)

    assert plan.question_type is QuestionType.FIGURE_CENTRIC
    assert plan.required_modalities == ("figure",)


def test_top_evidence_modality_drives_table_centric_without_lexical_marker() -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, "q", [_group(_candidate(document_id, modality="table"))])

    plan = AnswerPlanner().plan("What were the results?", bundle)

    assert plan.question_type is QuestionType.TABLE_CENTRIC


def test_multi_hop_requires_several_groups_and_graph_expansion() -> None:
    document_id = uuid4()
    groups = [
        _group(_candidate(document_id), rank=1),
        _group(_candidate(document_id), rank=2),
        _group(_candidate(document_id, discovery_method=DiscoveryMethod.GRAPH_EXPANSION), rank=3),
    ]
    bundle = _bundle(document_id, "q", groups)

    plan = AnswerPlanner().plan("What happened overall?", bundle)

    assert plan.question_type is QuestionType.MULTI_HOP


def test_few_groups_does_not_trigger_multi_hop_even_with_graph_expansion() -> None:
    document_id = uuid4()
    groups = [_group(_candidate(document_id, discovery_method=DiscoveryMethod.GRAPH_EXPANSION))]
    bundle = _bundle(document_id, "q", groups)

    plan = AnswerPlanner().plan("What happened?", bundle)

    assert plan.question_type is not QuestionType.MULTI_HOP


def test_descriptive_marker_classified_as_descriptive() -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, "q", [_group(_candidate(document_id))])

    plan = AnswerPlanner().plan("Describe the methodology used.", bundle)

    assert plan.question_type is QuestionType.DESCRIPTIVE


def test_default_fallback_is_factual() -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, "q", [_group(_candidate(document_id))])

    plan = AnswerPlanner().plan("What was the sample size?", bundle)

    assert plan.question_type is QuestionType.FACTUAL


def test_plan_always_requires_citations_and_all_six_sections() -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, "q", [_group(_candidate(document_id))])

    plan = AnswerPlanner().plan("What was the sample size?", bundle)

    assert plan.requires_citations is True
    assert len(plan.expected_sections) == 6
