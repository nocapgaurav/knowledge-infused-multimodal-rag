"""GroundingReport: the deterministic verdict on every claim in a generated answer.

A claim is never trusted because it sounds plausible or because the model
attached a citation to it -- every claim's citation is checked against the
actual evidence it was given, and every claim's presence in the report is
mandatory: an answer with zero claims found is as much a defect as one
with unsupported claims.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ClaimGroundingStatus(StrEnum):
    """The deterministic outcome of checking one claim against the evidence it cited."""

    GROUNDED = "grounded"
    """The claim cites at least one label present in the prompt's
    evidence, and the cited evidence's text supports the claim."""

    MISSING_CITATION = "missing_citation"
    """The claim carries no citation at all, though the answer plan
    requires citations."""

    UNRESOLVED_CITATION = "unresolved_citation"
    """The claim cites a label that does not correspond to any evidence
    actually shown to the model -- an invented or mistyped citation."""

    UNSUPPORTED = "unsupported"
    """The claim cites a real label, but the cited evidence's text does
    not actually support the claim -- a citation mismatch."""


class ClaimVerdict(BaseModel):
    """The grounding verdict for one sentence-level claim.

    Attributes:
        claim_text: The claim, as extracted from the generated answer.
        cited_labels: Citation labels (e.g. "KU1") the claim carries, in
            the order they appear. Empty if the claim carries none.
        status: The deterministic grounding outcome.
        reason: Human-readable explanation of the verdict.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_text: str = Field(min_length=1)
    cited_labels: tuple[str, ...] = Field(default_factory=tuple)
    status: ClaimGroundingStatus
    reason: str = Field(min_length=1)


class GroundingReport(BaseModel):
    """The complete grounding verdict for a generated answer.

    Attributes:
        claims: Every claim found in the answer, with its verdict. Never
            empty for a non-empty answer -- an answer that produces no
            extractable claims is itself a grounding defect, not a
            vacuous pass.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    claims: tuple[ClaimVerdict, ...] = Field(min_length=1)

    @property
    def is_fully_grounded(self) -> bool:
        """Whether every claim in the answer is `GROUNDED`."""
        return all(claim.status is ClaimGroundingStatus.GROUNDED for claim in self.claims)

    @property
    def grounded_ratio(self) -> float:
        """Fraction of claims that are `GROUNDED`, in `[0.0, 1.0]`."""
        return sum(
            1 for claim in self.claims if claim.status is ClaimGroundingStatus.GROUNDED
        ) / len(self.claims)
