"""Tests for Phase 9: answer quality assessment."""

from datetime import UTC, datetime
from uuid import uuid4

from backend.generation.models.answer_plan import (
    AnswerPlan,
    AnswerSection,
    ExpectedAnswerType,
    QuestionType,
)
from backend.generation.models.answer_status import AnswerStatus
from backend.generation.models.citation import (
    CitationResolutionReport,
    ResolvedCitation,
    UnresolvedCitation,
)
from backend.generation.models.grounding_report import (
    ClaimGroundingStatus,
    ClaimVerdict,
    GroundingReport,
)
from backend.generation.models.prompt_context import ContextSection
from backend.generation.quality.answer_quality_assessor import AnswerQualityAssessor
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


def _plan(required_groups=1, required_modalities=()) -> AnswerPlan:
    return AnswerPlan(
        question_type=QuestionType.FACTUAL,
        expected_answer_type=ExpectedAnswerType.SHORT_FACTUAL,
        required_evidence_groups=required_groups,
        required_modalities=required_modalities,
        expected_sections=tuple(AnswerSection),
    )


def _candidate(document_id, dense_similarity=0.8):
    return RetrievalCandidate(
        knowledge_unit_id=uuid4(),
        document_id=document_id,
        section_id=None,
        modality="text",
        text="evidence",
        asset_uri=None,
        reading_order=0,
        citation_count=0,
        dense_similarity=dense_similarity,
        discovery_method=DiscoveryMethod.DENSE_RETRIEVAL,
    )


def _bundle(document_id, dense_similarities) -> EvidenceBundle:
    groups = []
    for i, similarity in enumerate(dense_similarities, start=1):
        candidate = _candidate(document_id, dense_similarity=similarity)
        scored = ScoredCandidate(
            candidate=candidate,
            ranking=RankingExplanation(
                signals=(SignalScore(name="dense_similarity", raw_value=similarity, rank=i),),
                fused_score=1.0 / i,
                final_rank=i,
            ),
        )
        groups.append(
            EvidenceGroup(
                group_id=str(candidate.knowledge_unit_id),
                primary=scored,
                supporting=(),
                modalities=("text",),
            )
        )
    candidates = tuple(g.primary.candidate for g in groups)
    return EvidenceBundle(
        document_id=document_id,
        query="q",
        candidates=candidates,
        evidence_groups=tuple(groups),
        trace=RetrievalTrace(phases=(), dropped=()),
        manifest=RetrievalManifest(
            document_id=document_id,
            query="q",
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


def _grounding(*statuses: ClaimGroundingStatus) -> GroundingReport:
    return GroundingReport(
        claims=tuple(
            ClaimVerdict(claim_text=f"claim {i}", cited_labels=("KU1",), status=status, reason="r")
            for i, status in enumerate(statuses)
        )
    )


def _citations(resolved_count=1, unresolved_count=0) -> CitationResolutionReport:
    return CitationResolutionReport(
        resolved=tuple(
            ResolvedCitation(label=f"KU{i}", knowledge_unit_id=f"id-{i}", text_excerpt="e")
            for i in range(resolved_count)
        ),
        unresolved=tuple(
            UnresolvedCitation(label=f"KUX{i}", reason="r") for i in range(unresolved_count)
        ),
    )


def _section() -> ContextSection:
    return ContextSection(citation_label="KU1", knowledge_unit_id="id-0", text="e", modality="text")


def test_fully_grounded_and_complete_yields_sufficient_evidence() -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, [0.9])

    result = AnswerQualityAssessor().assess(
        bundle,
        _plan(required_groups=1),
        [_section()],
        _grounding(ClaimGroundingStatus.GROUNDED),
        _citations(resolved_count=1),
    )

    assert result.answer_status is AnswerStatus.SUFFICIENT_EVIDENCE
    assert result.confidence > 0.8


def test_partially_grounded_yields_partially_sufficient() -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, [0.9])

    result = AnswerQualityAssessor().assess(
        bundle,
        _plan(required_groups=1),
        [_section()],
        _grounding(ClaimGroundingStatus.GROUNDED, ClaimGroundingStatus.UNSUPPORTED),
        _citations(resolved_count=1),
    )

    assert result.answer_status is AnswerStatus.PARTIALLY_SUFFICIENT_EVIDENCE


def test_no_context_sections_yields_insufficient_evidence() -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, [])

    result = AnswerQualityAssessor().assess(
        bundle,
        _plan(required_groups=1),
        [],
        _grounding(ClaimGroundingStatus.MISSING_CITATION),
        _citations(resolved_count=0),
    )

    assert result.answer_status is AnswerStatus.INSUFFICIENT_EVIDENCE


def test_all_claims_ungrounded_with_evidence_yields_unverified_answer() -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, [0.9])

    result = AnswerQualityAssessor().assess(
        bundle,
        _plan(required_groups=1),
        [_section()],
        _grounding(ClaimGroundingStatus.UNSUPPORTED),
        _citations(resolved_count=0, unresolved_count=1),
    )

    # Evidence existed; only the answer's verification failed -- reporting
    # that as missing evidence was Sprint 2's Problem 5.
    assert result.answer_status is AnswerStatus.UNVERIFIED_ANSWER


def test_missing_required_evidence_groups_lowers_completeness() -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, [0.9])  # only 1 group, but 2 required

    result = AnswerQualityAssessor().assess(
        bundle,
        _plan(required_groups=2),
        [_section()],
        _grounding(ClaimGroundingStatus.GROUNDED),
        _citations(resolved_count=1),
    )

    assert result.evidence_completeness < 1.0
    assert result.answer_status is AnswerStatus.PARTIALLY_SUFFICIENT_EVIDENCE


def test_missing_required_modality_lowers_completeness() -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, [0.9])

    result = AnswerQualityAssessor().assess(
        bundle,
        _plan(required_groups=1, required_modalities=("figure",)),
        [_section()],  # only text, no figure
        _grounding(ClaimGroundingStatus.GROUNDED),
        _citations(resolved_count=1),
    )

    assert result.evidence_completeness < 1.0


def test_unresolved_citations_lower_citation_coverage() -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, [0.9])

    result = AnswerQualityAssessor().assess(
        bundle,
        _plan(required_groups=1),
        [_section()],
        _grounding(ClaimGroundingStatus.GROUNDED),
        _citations(resolved_count=1, unresolved_count=1),
    )

    assert result.citation_coverage == 0.5


def test_confidence_is_deterministic_given_the_same_inputs() -> None:
    document_id = uuid4()
    bundle = _bundle(document_id, [0.9])
    grounding = _grounding(ClaimGroundingStatus.GROUNDED)
    citations = _citations(resolved_count=1)

    first = AnswerQualityAssessor().assess(bundle, _plan(), [_section()], grounding, citations)
    second = AnswerQualityAssessor().assess(bundle, _plan(), [_section()], grounding, citations)

    assert first.confidence == second.confidence
    assert first.answer_status == second.answer_status
