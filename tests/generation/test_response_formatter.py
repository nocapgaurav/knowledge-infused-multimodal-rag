"""Tests for Phase 10: response formatting."""

from datetime import UTC, datetime
from uuid import uuid4

from backend.generation.formatting.response_formatter import FormattingInput, ResponseFormatter
from backend.generation.models.answer_plan import (
    AnswerPlan,
    AnswerSection,
    ExpectedAnswerType,
    QuestionType,
)
from backend.generation.models.answer_provenance import AnswerProvenance
from backend.generation.models.answer_status import AnswerStatus
from backend.generation.models.citation import CitationResolutionReport, ResolvedCitation
from backend.generation.models.generation_manifest import GenerationStatistics
from backend.generation.models.generation_trace import GenerationTrace
from backend.generation.models.grounding_report import (
    ClaimGroundingStatus,
    ClaimVerdict,
    GroundingReport,
)
from backend.generation.models.prompt_context import ContextSection
from backend.generation.quality.answer_quality_assessor import QualityAssessment
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


def _bundle(document_id, group_count=1) -> EvidenceBundle:
    groups = []
    for _ in range(group_count):
        candidate = RetrievalCandidate(
            knowledge_unit_id=uuid4(),
            document_id=document_id,
            section_id=None,
            modality="text",
            text="evidence",
            asset_uri=None,
            reading_order=0,
            citation_count=0,
            dense_similarity=0.8,
            discovery_method=DiscoveryMethod.DENSE_RETRIEVAL,
        )
        scored = ScoredCandidate(
            candidate=candidate,
            ranking=RankingExplanation(
                signals=(SignalScore(name="dense_similarity", raw_value=0.8, rank=1),),
                fused_score=1.0,
                final_rank=1,
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


def _formatting_input(
    document_id,
    answer_text="The answer text [KU1].",
    context_sections=None,
    plan=None,
    grounding_status=ClaimGroundingStatus.GROUNDED,
    resolved=None,
    unresolved_notes=(),
    confidence=0.9,
    answer_status=AnswerStatus.SUFFICIENT_EVIDENCE,
) -> FormattingInput:
    section = ContextSection(
        citation_label="KU1", knowledge_unit_id="id-1", text="the cited evidence", modality="text"
    )
    sections = context_sections if context_sections is not None else [section]
    citation_report = CitationResolutionReport(
        resolved=(
            resolved
            if resolved is not None
            else (
                ResolvedCitation(
                    label="KU1", knowledge_unit_id="id-1", text_excerpt="the cited evidence"
                ),
            )
        ),
        unresolved=(),
    )
    return FormattingInput(
        document_id=document_id,
        query="What happened?",
        answer_text=answer_text,
        plan=plan or _plan(),
        context_sections=sections,
        context_optimization_notes=list(unresolved_notes),
        grounding_report=GroundingReport(
            claims=(
                ClaimVerdict(
                    claim_text=answer_text,
                    cited_labels=("KU1",),
                    status=grounding_status,
                    reason="r",
                ),
            )
        ),
        citation_report=citation_report,
        quality=QualityAssessment(
            confidence=confidence,
            answer_status=answer_status,
            retrieval_quality=0.8,
            grounded_ratio=1.0 if grounding_status is ClaimGroundingStatus.GROUNDED else 0.0,
            citation_coverage=1.0,
            evidence_completeness=1.0,
        ),
        prompt_version="1.0",
        model_name="qwen2.5:7b-instruct",
        model_version="digest-abc",
        generation_trace=GenerationTrace(phases=()),
        generation_statistics=GenerationStatistics(
            context_sections_used=len(sections),
            context_sections_dropped=0,
            claims_total=1,
            claims_grounded=1 if grounding_status is ClaimGroundingStatus.GROUNDED else 0,
            citations_resolved=len(citation_report.resolved),
            citations_unresolved=len(citation_report.unresolved),
            prompt_tokens=10,
            completion_tokens=5,
            duration_ms=1.0,
        ),
        answer_provenance=AnswerProvenance(
            document_id=document_id,
            retrieval_version="1.0",
            retrieval_strategy_version="1.0",
            representation_version="repr",
            embedding_version="embed",
            graph_version="graph",
            knowledge_unit_ids=("id-1",),
            evidence_bundle_checksum="checksum",
        ),
    )


def test_format_produces_all_required_fields() -> None:
    document_id = uuid4()
    formatting_input = _formatting_input(document_id)
    bundle = _bundle(document_id)

    response = ResponseFormatter().format(formatting_input, bundle)

    assert response.document_id == document_id
    assert response.answer == formatting_input.answer_text
    assert response.executive_summary
    assert len(response.supporting_evidence) == 1
    assert response.supporting_evidence[0].knowledge_unit_id == "id-1"
    assert len(response.resolved_citations) == 1
    assert response.references == ("[KU1] the cited evidence",)


def test_executive_summary_is_first_sentence() -> None:
    document_id = uuid4()
    formatting_input = _formatting_input(
        document_id, answer_text="First sentence here [KU1]. Second sentence follows."
    )
    bundle = _bundle(document_id)

    response = ResponseFormatter().format(formatting_input, bundle)

    assert response.executive_summary == "First sentence here [KU1]."


def test_limitations_flag_ungrounded_claims() -> None:
    document_id = uuid4()
    formatting_input = _formatting_input(
        document_id, grounding_status=ClaimGroundingStatus.UNSUPPORTED
    )
    bundle = _bundle(document_id)

    response = ResponseFormatter().format(formatting_input, bundle)

    assert any("could not be fully verified" in limitation for limitation in response.limitations)


def test_limitations_flag_insufficient_evidence_group_count() -> None:
    document_id = uuid4()
    formatting_input = _formatting_input(document_id, plan=_plan(required_groups=3))
    bundle = _bundle(document_id, group_count=1)

    response = ResponseFormatter().format(formatting_input, bundle)

    assert any("evidence group" in limitation for limitation in response.limitations)


def test_limitations_flag_missing_required_modality() -> None:
    document_id = uuid4()
    formatting_input = _formatting_input(document_id, plan=_plan(required_modalities=("figure",)))
    bundle = _bundle(document_id)

    response = ResponseFormatter().format(formatting_input, bundle)

    assert any("figure" in limitation for limitation in response.limitations)


def test_generation_metadata_carries_question_type() -> None:
    document_id = uuid4()
    formatting_input = _formatting_input(document_id)
    bundle = _bundle(document_id)

    response = ResponseFormatter().format(formatting_input, bundle)

    assert response.generation_metadata["question_type"] == "factual"


def test_uncited_answer_falls_back_to_context_as_labeled_evidence() -> None:
    document_id = uuid4()
    result = ResponseFormatter().format(
        _formatting_input(document_id, answer_text="An answer with no citations.", resolved=()),
        _bundle(document_id),
    )

    assert len(result.supporting_evidence) == 1
    item = result.supporting_evidence[0]
    assert item.label == "KU1"
    assert item.discovery == "Shown to the model as context (not cited in the answer)"
