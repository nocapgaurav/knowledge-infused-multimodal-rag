"""Phase 9: Answer Quality Assessment.

Computes confidence entirely from measurable facts already available by
this point in the pipeline -- never from the LLM's own self-reported
certainty, which is not a real signal (a hallucinating model is not
statistically more likely to sound unsure).

The four signals named by the architecture (retrieval quality, grounding
validation, citation coverage, evidence completeness) are each already a
ratio in `[0.0, 1.0]` -- unlike Module 9's ranking signals (an RRF fusion
of incomparable raw magnitudes), these four are already on the same
"how complete/correct is this" scale, so an unweighted average is not the
same arbitrary-weighting problem RRF was chosen to avoid there; it is a
symmetric combination of already-comparable quantities.

`AnswerStatus` is derived from explicit rules, not a confidence cutoff --
"sufficient evidence" should mean every claim was actually grounded and
every plan requirement was met, not merely "the average score was high."
"""

from dataclasses import dataclass

from backend.generation.models.answer_plan import AnswerPlan
from backend.generation.models.answer_status import AnswerStatus
from backend.generation.models.citation import CitationResolutionReport
from backend.generation.models.grounding_report import GroundingReport
from backend.generation.models.prompt_context import ContextSection
from backend.retrieval.models import EvidenceBundle

_DEFAULT_DENSE_SIMILARITY_FOR_GRAPH_ONLY_EVIDENCE = 0.5
"""Used only when an evidence group's primary candidate was discovered
solely through graph expansion (no direct dense match) -- a neutral
midpoint, since such a candidate has no dense similarity score to draw on
but was still selected as a group's anchor by Module 9's own ranking."""


@dataclass(frozen=True)
class QualityAssessment:
    """The complete, deterministic quality assessment for one generated answer.

    Attributes:
        confidence: The final confidence score, `[0.0, 1.0]`.
        answer_status: The deterministic evidence-sufficiency outcome.
        retrieval_quality: Average dense similarity of evidence group
            primaries -- how strong the underlying retrieval was.
        grounded_ratio: Fraction of claims that passed Grounding Validation.
        citation_coverage: Fraction of citation labels used that resolved
            to real evidence.
        evidence_completeness: How well the available evidence met the
            answer plan's stated requirements.
    """

    confidence: float
    answer_status: AnswerStatus
    retrieval_quality: float
    grounded_ratio: float
    citation_coverage: float
    evidence_completeness: float


class AnswerQualityAssessor:
    """Deterministically assesses a generated answer's quality and confidence."""

    def assess(
        self,
        bundle: EvidenceBundle,
        plan: AnswerPlan,
        context_sections: list[ContextSection],
        grounding_report: GroundingReport,
        citation_report: CitationResolutionReport,
    ) -> QualityAssessment:
        """Assess a generated answer.

        Args:
            bundle: The evidence retrieved for this question (Module 9's output).
            plan: The answer plan this answer was built to satisfy.
            context_sections: The optimized evidence actually shown to the model.
            grounding_report: Phase 7's per-claim verdicts.
            citation_report: Phase 8's citation resolution outcome.

        Returns:
            The complete quality assessment.
        """
        retrieval_quality = _average_primary_dense_similarity(bundle)
        grounded_ratio = grounding_report.grounded_ratio
        citation_coverage = _citation_coverage(citation_report)
        evidence_completeness = _evidence_completeness(bundle, plan, context_sections)

        confidence = (
            retrieval_quality + grounded_ratio + citation_coverage + evidence_completeness
        ) / 4

        answer_status = _determine_status(context_sections, grounded_ratio, evidence_completeness)

        return QualityAssessment(
            confidence=confidence,
            answer_status=answer_status,
            retrieval_quality=retrieval_quality,
            grounded_ratio=grounded_ratio,
            citation_coverage=citation_coverage,
            evidence_completeness=evidence_completeness,
        )


def _average_primary_dense_similarity(bundle: EvidenceBundle) -> float:
    if not bundle.evidence_groups:
        return 0.0
    similarities = [
        (
            group.primary.candidate.dense_similarity
            if group.primary.candidate.dense_similarity is not None
            else _DEFAULT_DENSE_SIMILARITY_FOR_GRAPH_ONLY_EVIDENCE
        )
        for group in bundle.evidence_groups
    ]
    return sum(similarities) / len(similarities)


def _citation_coverage(citation_report: CitationResolutionReport) -> float:
    total = len(citation_report.resolved) + len(citation_report.unresolved)
    if total == 0:
        # No citation was ever attempted; the absence of any citation at
        # all is already penalized by grounded_ratio (every uncited claim
        # is MISSING_CITATION there) -- this signal only measures whether
        # attempted citations were valid, so it is vacuously satisfied.
        return 1.0
    return len(citation_report.resolved) / total


def _evidence_completeness(
    bundle: EvidenceBundle, plan: AnswerPlan, context_sections: list[ContextSection]
) -> float:
    groups_completeness = min(len(bundle.evidence_groups) / plan.required_evidence_groups, 1.0)
    if not plan.required_modalities:
        modality_completeness = 1.0
    else:
        available_modalities = {section.modality for section in context_sections}
        satisfied = sum(1 for m in plan.required_modalities if m in available_modalities)
        modality_completeness = satisfied / len(plan.required_modalities)
    return (groups_completeness + modality_completeness) / 2


def _determine_status(
    context_sections: list[ContextSection], grounded_ratio: float, evidence_completeness: float
) -> AnswerStatus:
    if not context_sections:
        return AnswerStatus.INSUFFICIENT_EVIDENCE
    if grounded_ratio >= 1.0 and evidence_completeness >= 1.0:
        return AnswerStatus.SUFFICIENT_EVIDENCE
    if grounded_ratio > 0.0:
        return AnswerStatus.PARTIALLY_SUFFICIENT_EVIDENCE
    return AnswerStatus.INSUFFICIENT_EVIDENCE
