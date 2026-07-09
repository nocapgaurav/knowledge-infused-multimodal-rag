"""Ranking: the per-signal, per-candidate explanation behind a final score."""

from pydantic import BaseModel, ConfigDict, Field

from backend.retrieval.models.retrieval_candidate import RetrievalCandidate


class SignalScore(BaseModel):
    """One deterministic signal's contribution to a candidate's ranking.

    Attributes:
        name: Name of the signal (e.g. "dense_similarity", "graph_proximity").
        raw_value: The signal's own raw value, in its own units -- not
            comparable across signals, which is exactly why fusion (see
            `RankingExplanation`) happens over ranks, not raw values.
        rank: This candidate's 1-based rank among all candidates under this
            signal alone (1 is best). Ties share the same rank.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    raw_value: float
    rank: int = Field(ge=1)


class RankingExplanation(BaseModel):
    """The complete, reproducible justification for one candidate's final rank.

    Attributes:
        signals: Every signal considered, with this candidate's raw value
            and rank under each.
        fused_score: The Reciprocal Rank Fusion score combining every
            signal's rank -- see `evaluation/evidence_evaluator.py` for why
            RRF, not a hand-picked weighted sum, is used.
        final_rank: This candidate's 1-based rank after fusion (1 is best).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    signals: tuple[SignalScore, ...] = Field(min_length=1)
    fused_score: float
    final_rank: int = Field(ge=1)


class ScoredCandidate(BaseModel):
    """A candidate paired with its ranking explanation.

    Attributes:
        candidate: The underlying candidate.
        ranking: Why it was ranked where it was.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: RetrievalCandidate
    ranking: RankingExplanation
