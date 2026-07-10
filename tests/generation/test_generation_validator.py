"""Tests for GenerationValidator's whole-response structural checks.

Most defects here are unreachable through the normal pipeline (each phase
is correct by construction) -- these tests exercise the validator
directly against hand-crafted, deliberately malformed responses, proving
each check actually fires if a future change ever introduces the defect.
"""

from uuid import uuid4

import pytest

from backend.generation.exceptions import (
    CitationConsistencyError,
    GenerationTraceIncompleteError,
    StatisticsMismatchError,
)
from backend.generation.models.answer_provenance import AnswerProvenance
from backend.generation.models.answer_status import AnswerStatus
from backend.generation.models.citation import ResolvedCitation
from backend.generation.models.generation_manifest import GenerationStatistics
from backend.generation.models.generation_trace import GenerationTrace, PhaseTrace
from backend.generation.models.grounded_response import GroundedResponse, SupportingEvidenceItem
from backend.generation.models.prompt_context import ContextSection
from backend.generation.validation.generation_validator import (
    EXPECTED_PHASES,
    GenerationValidator,
)


def _statistics(**overrides) -> GenerationStatistics:
    defaults = dict(
        context_sections_used=1,
        context_sections_dropped=0,
        claims_total=1,
        claims_grounded=1,
        citations_resolved=1,
        citations_unresolved=0,
        prompt_tokens=10,
        completion_tokens=5,
        duration_ms=1.0,
    )
    defaults.update(overrides)
    return GenerationStatistics(**defaults)


def _response(
    document_id, statistics=None, resolved_citations=None, supporting_evidence=None, phases=None
) -> GroundedResponse:
    return GroundedResponse(
        document_id=document_id,
        query="q",
        answer="The answer [KU1].",
        executive_summary="The answer.",
        supporting_evidence=(
            supporting_evidence
            if supporting_evidence is not None
            else (
                SupportingEvidenceItem(
                    label="KU1", knowledge_unit_id="id-1", text="evidence", modality="text"
                ),
            )
        ),
        resolved_citations=(
            resolved_citations
            if resolved_citations is not None
            else (ResolvedCitation(label="KU1", knowledge_unit_id="id-1", text_excerpt="evidence"),)
        ),
        limitations=(),
        references=("[KU1] evidence",),
        warnings=(),
        confidence=0.9,
        answer_status=AnswerStatus.SUFFICIENT_EVIDENCE,
        generation_metadata={},
        prompt_version="1.0",
        model_name="qwen2.5:7b-instruct",
        model_version="digest",
        generation_trace=GenerationTrace(
            phases=(
                phases
                if phases is not None
                else tuple(
                    PhaseTrace(phase=name, input_count=1, output_count=1, duration_ms=1.0)
                    for name in EXPECTED_PHASES
                )
            )
        ),
        generation_statistics=statistics or _statistics(),
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


def _section() -> ContextSection:
    return ContextSection(
        citation_label="KU1", knowledge_unit_id="id-1", text="evidence", modality="text"
    )


def test_valid_response_passes() -> None:
    document_id = uuid4()
    GenerationValidator().validate(_response(document_id), [_section()])  # should not raise


def test_context_sections_used_mismatch_raises() -> None:
    document_id = uuid4()
    response = _response(document_id, statistics=_statistics(context_sections_used=5))

    with pytest.raises(StatisticsMismatchError):
        GenerationValidator().validate(response, [_section()])


def test_citations_resolved_count_mismatch_raises() -> None:
    document_id = uuid4()
    response = _response(document_id, statistics=_statistics(citations_resolved=5))

    with pytest.raises(StatisticsMismatchError):
        GenerationValidator().validate(response, [_section()])


def test_duplicate_resolved_citation_label_raises() -> None:
    document_id = uuid4()
    duplicate = (
        ResolvedCitation(label="KU1", knowledge_unit_id="id-1", text_excerpt="e"),
        ResolvedCitation(label="KU1", knowledge_unit_id="id-2", text_excerpt="e"),
    )
    response = _response(
        document_id,
        resolved_citations=duplicate,
        statistics=_statistics(citations_resolved=2),
    )

    with pytest.raises(CitationConsistencyError):
        GenerationValidator().validate(response, [_section()])


def test_supporting_evidence_referencing_unresolved_id_raises() -> None:
    document_id = uuid4()
    orphan_supporting = (
        SupportingEvidenceItem(
            label="KU2", knowledge_unit_id="id-not-resolved", text="e", modality="text"
        ),
    )
    response = _response(document_id, supporting_evidence=orphan_supporting)

    with pytest.raises(CitationConsistencyError):
        GenerationValidator().validate(response, [_section()])


def test_missing_trace_phase_raises() -> None:
    document_id = uuid4()
    incomplete_phases = tuple(
        PhaseTrace(phase=name, input_count=1, output_count=1, duration_ms=1.0)
        for name in EXPECTED_PHASES[:-1]  # missing "response_formatting"
    )
    response = _response(document_id, phases=incomplete_phases)

    with pytest.raises(GenerationTraceIncompleteError):
        GenerationValidator().validate(response, [_section()])
