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

    `INSUFFICIENT_EVIDENCE` means retrieval found nothing to answer from
    -- a statement about the paper. `UNVERIFIED_ANSWER` means evidence
    was found but none of the answer's claims could be verified against
    it (typically the model failed to cite) -- a statement about this
    answer. Conflating the two erodes trust: the information may well be
    in the paper (observed live in Sprint 2's investigation).
    """

    SUFFICIENT_EVIDENCE = "sufficient_evidence"
    PARTIALLY_SUFFICIENT_EVIDENCE = "partially_sufficient_evidence"
    UNVERIFIED_ANSWER = "unverified_answer"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
