"""Fail-loud whole-response structural verification, run once at the very
end of the pipeline.

Distinct from Grounding Validation (Phase 7, which judges *content* --
whether claims are supported) and Prompt Validation (Phase 5, which
checks the *prompt* before generation): this checks that the fully
assembled `GroundedResponse` is internally consistent with itself and
with the counts it claims about its own construction. Every check here
should be unreachable through correct phase implementations.
"""

import logging

from backend.generation.exceptions import (
    CitationConsistencyError,
    GenerationTraceIncompleteError,
    StatisticsMismatchError,
)
from backend.generation.models.grounded_response import GroundedResponse
from backend.generation.models.prompt_context import ContextSection

logger = logging.getLogger(__name__)

EXPECTED_PHASES = (
    "answer_planning",
    "context_optimization",
    "prompt_composition",
    "prompt_validation",
    "generation",
    "grounding_validation",
    "citation_resolution",
    "answer_quality_assessment",
    "response_formatting",
)


class GenerationValidator:
    """Verifies a fully assembled GroundedResponse is internally consistent."""

    def validate(self, response: GroundedResponse, context_sections: list[ContextSection]) -> None:
        """Validate a fully assembled response.

        Args:
            response: The complete, formatted response.
            context_sections: The optimized evidence actually shown to the
                model for this response.

        Raises:
            StatisticsMismatchError: The manifest's recorded statistics
                don't match the response's actual contents.
            CitationConsistencyError: A resolved citation or supporting
                evidence item is internally inconsistent.
            GenerationTraceIncompleteError: The trace does not cover every
                expected phase.
        """
        self._validate_statistics(response, context_sections)
        self._validate_citation_consistency(response)
        self._validate_trace_completeness(response)

    def _validate_statistics(
        self, response: GroundedResponse, context_sections: list[ContextSection]
    ) -> None:
        stats = response.generation_statistics
        if stats.context_sections_used != len(context_sections):
            raise StatisticsMismatchError(
                reason=(
                    f"manifest reports {stats.context_sections_used} context sections used, "
                    f"but {len(context_sections)} were actually shown to the model"
                )
            )
        if stats.citations_resolved != len(response.resolved_citations):
            raise StatisticsMismatchError(
                reason=(
                    f"manifest reports {stats.citations_resolved} resolved citations, "
                    f"response has {len(response.resolved_citations)}"
                )
            )

    def _validate_citation_consistency(self, response: GroundedResponse) -> None:
        seen_labels: set[str] = set()
        for citation in response.resolved_citations:
            if citation.label in seen_labels:
                raise CitationConsistencyError(
                    reason=f"citation label '{citation.label}' is resolved more than once"
                )
            seen_labels.add(citation.label)

        if not response.resolved_citations:
            # The uncited-answer fallback: when nothing resolved, the
            # formatter surfaces the context shown to the model, honestly
            # labeled as uncited -- those items intentionally have no
            # matching resolved citation to be consistent with.
            return

        resolved_ids = {citation.knowledge_unit_id for citation in response.resolved_citations}
        for item in response.supporting_evidence:
            if item.knowledge_unit_id not in resolved_ids:
                raise CitationConsistencyError(
                    reason=(
                        f"supporting evidence references {item.knowledge_unit_id}, "
                        f"which is not among the resolved citations"
                    )
                )

    def _validate_trace_completeness(self, response: GroundedResponse) -> None:
        traced_phases = {phase.phase for phase in response.generation_trace.phases}
        missing_phases = sorted(set(EXPECTED_PHASES) - traced_phases)
        if missing_phases:
            raise GenerationTraceIncompleteError(missing_phases=missing_phases)
