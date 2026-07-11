"""Phase 3: Evidence Evaluation.

Ranks candidates using multiple independent, deterministic signals, fused
by Reciprocal Rank Fusion (RRF) rather than a hand-weighted linear
combination.

Why not a weighted sum: the five signals below live on incomparable
scales -- a cosine similarity in roughly [0, 1], an inverse hop count, an
integer citation count, an integer co-occurrence count. Combining raw
values with per-signal weights would require calibrating exactly the kind
of arbitrary constants this design is required to avoid, and the "right"
weights would silently depend on corpus size and citation density in ways
that never generalize. RRF instead fuses each signal's *rank* -- score(c)
= sum(1 / (k + rank_signal(c))) -- which needs no cross-signal calibration
at all: a rank is already a comparable, scale-free ordinal. `k = 60` is
not a tuned weight; it is the constant from Cormack, Clarke & Buettcher
(2009), "Reciprocal Rank Fusion outperforms Condorcet and individual Rank
Learning Methods," found to work well across many independent ranking
tasks and widely reused since (e.g. Elasticsearch's and OpenSearch's own
RRF implementations use the same default).

Signals deliberately NOT included, with justification:
    - Evidence diversity: a property of a *selected set*, not of one
      candidate in isolation -- belongs to Phase 4's assembly strategy.
    - Reading order, document hierarchy: neither has a principled,
      direction-justified effect on evidence quality given this module's
      inputs -- an earlier paragraph is not inherently better evidence
      than a later one, and Module 8's Section nodes carry no title/level
      to derive a real hierarchy depth from.
    - Modality: a relevant figure is not inherently better or worse
      evidence than a relevant paragraph; modality diversity is an
      assembly concern (Phase 4), not a ranking-priority concern.
    - Citation importance and section co-occurrence (both fused in an
      earlier revision): query-INDEPENDENT priors, not relevance
      rankings, so fusing them violates the premise of the RRF result
      cited above (which fuses independent *relevance* estimates).
      Measured live in Sprint 1, they systematically buried the correct
      evidence for factual questions: the chunk answering "What are the
      keywords?" was the #1 dense match yet fell outside the top five
      because it had no section (worst co-occurrence rank) and no
      citations, while topically vague "hub" paragraphs from popular,
      frequently-cited sections outranked it on both priors.
"""

import logging
import re
from collections.abc import Callable, Sequence

from backend.retrieval.expansion.relationship_policy import confidence_tier
from backend.retrieval.models import (
    RankingExplanation,
    RetrievalCandidate,
    ScoredCandidate,
    SignalScore,
)

logger = logging.getLogger(__name__)

_RRF_K = 60

_DENSE_MATCH_CONFIDENCE_TIER = 4
"""A candidate found directly by dense retrieval needed no graph
corroboration -- the strongest available signal already vouches for it,
so it outranks every graph-discovered relationship tier (max tier 3, see
`relationship_policy.py`) under this signal."""


class EvidenceEvaluator:
    """Ranks a candidate pool using multiple deterministic signals, fused by RRF."""

    def evaluate(
        self, candidates: Sequence[RetrievalCandidate], query: str = ""
    ) -> list[ScoredCandidate]:
        """Score and rank a candidate pool.

        Args:
            candidates: Every candidate under consideration (Phase 1's
                dense matches plus Phase 2's graph-expanded discoveries).
            query: The user's question, for the lexical overlap signal.
                Empty disables that signal (it ranks all candidates tied).

        Returns:
            Scored candidates ordered by ascending final rank (best first).
        """
        if not candidates:
            return []

        query_terms = _content_terms(query)
        signal_definitions: dict[str, Callable[[RetrievalCandidate], float | None]] = {
            "dense_similarity": lambda c: c.dense_similarity,
            "lexical_overlap": lambda c: _lexical_overlap(c, query_terms),
            "graph_proximity": lambda c: 1.0 / (1 + c.graph_path.depth),
            "relationship_confidence": _relationship_confidence,
        }
        signal_ranks = {
            name: _rank_descending(candidates, key_fn)
            for name, key_fn in signal_definitions.items()
        }

        fused: list[tuple[RetrievalCandidate, tuple[SignalScore, ...], float]] = []
        for candidate in candidates:
            key = str(candidate.knowledge_unit_id)
            signals = tuple(
                SignalScore(name=name, raw_value=ranks[key][0], rank=ranks[key][1])
                for name, ranks in signal_ranks.items()
            )
            fused_score = sum(1.0 / (_RRF_K + signal.rank) for signal in signals)
            fused.append((candidate, signals, fused_score))

        fused.sort(key=lambda item: item[2], reverse=True)
        return [
            ScoredCandidate(
                candidate=candidate,
                ranking=RankingExplanation(
                    signals=signals, fused_score=fused_score, final_rank=index + 1
                ),
            )
            for index, (candidate, signals, fused_score) in enumerate(fused)
        ]


def _relationship_confidence(candidate: RetrievalCandidate) -> float:
    if candidate.graph_path.depth == 0:
        return float(_DENSE_MATCH_CONFIDENCE_TIER)
    return float(confidence_tier(candidate.graph_path.hops[-1].relationship_type))


_TERM_PATTERN = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    {"a", "an", "and", "are", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or"}
    | {"that", "the", "this", "to", "was", "were", "what", "when", "where", "which"}
    | {"who", "why", "how"}
)


def _content_terms(text: str) -> frozenset[str]:
    return frozenset(_TERM_PATTERN.findall(text.lower())) - _STOPWORDS


def _lexical_overlap(candidate: RetrievalCandidate, query_terms: frozenset[str]) -> float:
    """Fraction of the query's content terms appearing in the candidate.

    The one query-dependent signal besides dense similarity. Deterministic
    and scale-free, it directly rewards structural-term matches dense
    embeddings blur ("Table", "Figure", "Abstract", a printed number) --
    the candidate's `retrieval_context` participates precisely so that a
    chunk's structural identity counts as matchable text. With no query
    terms every candidate ties, and RRF's rank fusion makes a fully tied
    signal a no-op rather than noise.
    """
    if not query_terms:
        return 0.0
    haystack = candidate.text
    if candidate.retrieval_context:
        haystack = f"{candidate.retrieval_context} {haystack}"
    candidate_terms = _content_terms(haystack)
    return len(query_terms & candidate_terms) / len(query_terms)


def _rank_descending(
    candidates: Sequence[RetrievalCandidate],
    key_fn: Callable[[RetrievalCandidate], float | None],
) -> dict[str, tuple[float, int]]:
    """Rank candidates by a signal, higher raw value ranked better (rank 1).

    Candidates for which `key_fn` returns `None` (the signal does not
    apply, e.g. no section) all tie for the worst rank, with a reported
    raw value of `0.0` -- distinct from a candidate that has the signal
    and legitimately scores `0.0` on it, which keeps its true rank among
    the candidates that do have a value.
    """
    with_value: list[tuple[RetrievalCandidate, float]] = []
    without_value: list[RetrievalCandidate] = []
    for candidate in candidates:
        value = key_fn(candidate)
        if value is None:
            without_value.append(candidate)
        else:
            with_value.append((candidate, value))
    with_value.sort(key=lambda item: item[1], reverse=True)

    result: dict[str, tuple[float, int]] = {}
    current_rank = 0
    previous_value: float | None = None
    for position, (candidate, value) in enumerate(with_value, start=1):
        if value != previous_value:
            current_rank = position
        result[str(candidate.knowledge_unit_id)] = (value, current_rank)
        previous_value = value

    worst_rank = len(with_value) + 1
    for candidate in without_value:
        result[str(candidate.knowledge_unit_id)] = (0.0, worst_rank)
    return result
