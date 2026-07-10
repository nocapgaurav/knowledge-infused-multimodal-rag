"""Phase 10: Response Formatting.

Assembles every prior phase's output into the final `GroundedResponse`.
Purely deterministic assembly -- no new judgment calls are made here; by
this point Grounding Validation, Citation Resolution, and Answer Quality
Assessment have already decided everything that matters. This phase only
renders those decisions into the structured, six-section scientific
response the architecture requires.

`answer` is the model's raw text, verbatim, citation markers ([KU1], ...)
included -- kept intact rather than rewritten, both because it is the
simplest, most deterministic choice (no further text transformation to
introduce bugs into) and because the inline markers are exactly the
sentence-level provenance Phase 8 exists to make possible.
"""

from dataclasses import dataclass

from backend.domain import PaperId
from backend.generation.models.answer_plan import AnswerPlan
from backend.generation.models.answer_provenance import AnswerProvenance
from backend.generation.models.citation import CitationResolutionReport
from backend.generation.models.generation_manifest import GenerationStatistics
from backend.generation.models.generation_trace import GenerationTrace
from backend.generation.models.grounded_response import GroundedResponse, SupportingEvidenceItem
from backend.generation.models.grounding_report import ClaimGroundingStatus, GroundingReport
from backend.generation.models.prompt_context import ContextSection
from backend.generation.quality.answer_quality_assessor import QualityAssessment
from backend.retrieval.models import EvidenceBundle


@dataclass(frozen=True)
class FormattingInput:
    """Every piece of prior-phase output Response Formatting needs.

    Bundled into one object because Response Formatting is, by design,
    the single place all ten phases' results converge -- passing each as
    a separate parameter would not make the dependency any smaller, only
    harder to read.
    """

    document_id: PaperId
    query: str
    answer_text: str
    plan: AnswerPlan
    context_sections: list[ContextSection]
    context_optimization_notes: list[str]
    grounding_report: GroundingReport
    citation_report: CitationResolutionReport
    quality: QualityAssessment
    prompt_version: str
    model_name: str
    model_version: str
    generation_trace: GenerationTrace
    generation_statistics: GenerationStatistics
    answer_provenance: AnswerProvenance


class ResponseFormatter:
    """Assembles the final GroundedResponse from every prior phase's output."""

    def format(self, formatting_input: FormattingInput, bundle: EvidenceBundle) -> GroundedResponse:
        """Format the final grounded response.

        Args:
            formatting_input: Every piece of prior-phase output needed.
            bundle: The evidence retrieved for this question (Module 9's
                output), for evidence-completeness reporting.

        Returns:
            The complete, structured `GroundedResponse`.
        """
        section_by_label = {
            section.citation_label: section for section in formatting_input.context_sections
        }

        supporting_evidence = tuple(
            SupportingEvidenceItem(
                label=citation.label,
                knowledge_unit_id=citation.knowledge_unit_id,
                text=citation.text_excerpt,
                modality=section_by_label[citation.label].modality,
            )
            for citation in formatting_input.citation_report.resolved
            if citation.label in section_by_label
        )

        references = tuple(
            f"[{citation.label}] {citation.text_excerpt}"
            for citation in formatting_input.citation_report.resolved
        )

        limitations = _build_limitations(
            formatting_input.plan,
            bundle,
            formatting_input.context_sections,
            formatting_input.grounding_report,
        )
        warnings = _build_warnings(
            formatting_input.context_optimization_notes, formatting_input.citation_report
        )
        executive_summary = _build_executive_summary(formatting_input.answer_text)

        return GroundedResponse(
            document_id=formatting_input.document_id,
            query=formatting_input.query,
            answer=formatting_input.answer_text,
            executive_summary=executive_summary,
            supporting_evidence=supporting_evidence,
            resolved_citations=formatting_input.citation_report.resolved,
            limitations=limitations,
            references=references,
            warnings=warnings,
            confidence=formatting_input.quality.confidence,
            answer_status=formatting_input.quality.answer_status,
            generation_metadata={"question_type": formatting_input.plan.question_type.value},
            prompt_version=formatting_input.prompt_version,
            model_name=formatting_input.model_name,
            model_version=formatting_input.model_version,
            generation_trace=formatting_input.generation_trace,
            generation_statistics=formatting_input.generation_statistics,
            answer_provenance=formatting_input.answer_provenance,
        )


def _build_executive_summary(answer_text: str) -> str:
    first_sentence_end = min(
        (index for index in (answer_text.find(mark) for mark in (". ", "! ", "? ")) if index != -1),
        default=-1,
    )
    if first_sentence_end == -1:
        return answer_text
    return answer_text[: first_sentence_end + 1]


def _build_limitations(
    plan: AnswerPlan,
    bundle: EvidenceBundle,
    context_sections: list[ContextSection],
    grounding_report: GroundingReport,
) -> tuple[str, ...]:
    limitations: list[str] = []

    if len(bundle.evidence_groups) < plan.required_evidence_groups:
        limitations.append(
            f"Only {len(bundle.evidence_groups)} of the {plan.required_evidence_groups} "
            f"evidence group(s) typically needed for this kind of question were available."
        )

    available_modalities = {section.modality for section in context_sections}
    for modality in plan.required_modalities:
        if modality not in available_modalities:
            limitations.append(
                f"No {modality.value} evidence was found, though this question specifically "
                f"calls for it."
            )

    ungrounded_count = sum(
        1 for claim in grounding_report.claims if claim.status is not ClaimGroundingStatus.GROUNDED
    )
    if ungrounded_count:
        limitations.append(
            f"{ungrounded_count} statement(s) in the answer could not be fully verified "
            f"against the provided evidence."
        )

    return tuple(limitations)


def _build_warnings(
    context_optimization_notes: list[str], citation_report: CitationResolutionReport
) -> tuple[str, ...]:
    warnings = list(context_optimization_notes)
    if citation_report.unresolved:
        warnings.append(
            f"{len(citation_report.unresolved)} citation(s) referenced evidence that was not "
            f"shown to the model and were disregarded."
        )
    return tuple(warnings)
