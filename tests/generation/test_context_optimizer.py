"""Tests for Phase 3: context optimization."""

from datetime import UTC, datetime
from uuid import uuid4

from backend.generation.context.context_optimizer import ContextOptimizer
from backend.generation.models.answer_plan import (
    AnswerPlan,
    AnswerSection,
    ExpectedAnswerType,
    QuestionType,
)
from backend.generation.models.generation_config import GenerationConfig
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


def _candidate(document_id, text="evidence text", section_id=None):
    return RetrievalCandidate(
        knowledge_unit_id=uuid4(),
        document_id=document_id,
        section_id=section_id,
        modality="text",
        text=text,
        asset_uri=None,
        reading_order=0,
        citation_count=0,
        dense_similarity=0.8,
        discovery_method=DiscoveryMethod.DENSE_RETRIEVAL,
    )


def _scored(candidate, rank=1) -> ScoredCandidate:
    return ScoredCandidate(
        candidate=candidate,
        ranking=RankingExplanation(
            signals=(SignalScore(name="dense_similarity", raw_value=0.8, rank=rank),),
            fused_score=1.0 / rank,
            final_rank=rank,
        ),
    )


def _bundle(document_id, groups) -> EvidenceBundle:
    candidates = tuple(
        member.candidate for group in groups for member in (group.primary, *group.supporting)
    )
    return EvidenceBundle(
        document_id=document_id,
        query="a question",
        candidates=candidates,
        evidence_groups=tuple(groups),
        trace=RetrievalTrace(phases=(), dropped=()),
        manifest=RetrievalManifest(
            document_id=document_id,
            query="a question",
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
                evidence_items=len(candidates),
                duration_ms=1.0,
            ),
            created_at=datetime.now(UTC),
        ),
    )


def _plan() -> AnswerPlan:
    return AnswerPlan(
        question_type=QuestionType.FACTUAL,
        expected_answer_type=ExpectedAnswerType.SHORT_FACTUAL,
        required_evidence_groups=1,
        expected_sections=tuple(AnswerSection),
    )


def _config(context_window=4096, max_tokens=800) -> GenerationConfig:
    return GenerationConfig(
        provider="ollama",
        model="qwen2.5:7b-instruct",
        temperature=0.1,
        top_p=0.9,
        max_tokens=max_tokens,
        context_window=context_window,
    )


def test_optimize_preserves_rank_order_and_assigns_sequential_labels() -> None:
    document_id = uuid4()
    a = _candidate(document_id, text="first evidence")
    b = _candidate(document_id, text="second evidence")
    groups = [
        EvidenceGroup(
            group_id=str(a.knowledge_unit_id),
            primary=_scored(a, rank=1),
            supporting=(),
            modalities=("text",),
        ),
        EvidenceGroup(
            group_id=str(b.knowledge_unit_id),
            primary=_scored(b, rank=2),
            supporting=(),
            modalities=("text",),
        ),
    ]
    bundle = _bundle(document_id, groups)

    result = ContextOptimizer().optimize(bundle, _plan(), _config())

    assert [s.citation_label for s in result.context_sections] == ["KU1", "KU2"]
    assert [s.text for s in result.context_sections] == ["first evidence", "second evidence"]


def test_primary_and_supporting_both_included_primary_first() -> None:
    document_id = uuid4()
    primary = _candidate(document_id, text="primary text")
    supporting = _candidate(document_id, text="supporting text")
    group = EvidenceGroup(
        group_id=str(primary.knowledge_unit_id),
        primary=_scored(primary),
        supporting=(_scored(supporting, rank=2),),
        modalities=("text",),
    )
    bundle = _bundle(document_id, [group])

    result = ContextOptimizer().optimize(bundle, _plan(), _config())

    assert [s.text for s in result.context_sections] == ["primary text", "supporting text"]


def test_redundant_substring_evidence_is_removed() -> None:
    document_id = uuid4()
    long_text = "The experiment measured accuracy across five different datasets."
    short_text = "The experiment measured accuracy"  # fully contained in long_text
    a = _candidate(document_id, text=long_text)
    b = _candidate(document_id, text=short_text)
    group_a = EvidenceGroup(
        group_id=str(a.knowledge_unit_id),
        primary=_scored(a, rank=1),
        supporting=(),
        modalities=("text",),
    )
    group_b = EvidenceGroup(
        group_id=str(b.knowledge_unit_id),
        primary=_scored(b, rank=2),
        supporting=(),
        modalities=("text",),
    )
    bundle = _bundle(document_id, [group_a, group_b])

    result = ContextOptimizer().optimize(bundle, _plan(), _config())

    assert len(result.context_sections) == 1
    assert result.context_sections[0].text == long_text
    assert any("redundant" in note for note in result.notes)


def test_token_budget_drops_evidence_that_does_not_fit() -> None:
    document_id = uuid4()
    a = _candidate(document_id, text="x" * 100)
    b = _candidate(document_id, text="y" * 100)
    group_a = EvidenceGroup(
        group_id=str(a.knowledge_unit_id),
        primary=_scored(a, rank=1),
        supporting=(),
        modalities=("text",),
    )
    group_b = EvidenceGroup(
        group_id=str(b.knowledge_unit_id),
        primary=_scored(b, rank=2),
        supporting=(),
        modalities=("text",),
    )
    bundle = _bundle(document_id, [group_a, group_b])
    # context_window - max_tokens - overhead leaves very little room
    tight_config = _config(context_window=600, max_tokens=50)

    result = ContextOptimizer().optimize(bundle, _plan(), tight_config)

    assert len(result.context_sections) < 2
    assert any("budget" in note for note in result.notes)


def test_generous_budget_keeps_everything() -> None:
    document_id = uuid4()
    a = _candidate(document_id, text="short evidence one")
    b = _candidate(document_id, text="short evidence two")
    group_a = EvidenceGroup(
        group_id=str(a.knowledge_unit_id),
        primary=_scored(a, rank=1),
        supporting=(),
        modalities=("text",),
    )
    group_b = EvidenceGroup(
        group_id=str(b.knowledge_unit_id),
        primary=_scored(b, rank=2),
        supporting=(),
        modalities=("text",),
    )
    bundle = _bundle(document_id, [group_a, group_b])

    result = ContextOptimizer().optimize(bundle, _plan(), _config(context_window=32000))

    assert len(result.context_sections) == 2


def test_empty_bundle_produces_no_context_sections() -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, [])

    result = ContextOptimizer().optimize(bundle, _plan(), _config())

    assert result.context_sections == []
    assert result.total_candidates_considered == 0
