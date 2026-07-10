"""AnswerStatus: the deterministic evidence-sufficiency outcome for a generated answer.

Kept in its own module (not nested in `grounded_response.py`) because both
`GroundedResponse` and `GenerationManifest` need it, and neither should
depend on the other.
"""

from enum import StrEnum


class AnswerStatus(StrEnum):
    """The deterministic evidence-sufficiency outcome for a generated answer.

    Never inferred from the LLM's own tone or self-assessment -- computed
    from measurable grounding and citation coverage (see
    `quality/answer_quality_assessor.py`).
    """

    SUFFICIENT_EVIDENCE = "sufficient_evidence"
    PARTIALLY_SUFFICIENT_EVIDENCE = "partially_sufficient_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
