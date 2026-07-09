"""Fail-loud structural verification, run at every phase boundary.

Provider response shapes are validated implicitly, not by a separate
method here: every value a provider returns is immediately constructed
into a Pydantic model (`GraphNeighbor`, `SearchResult`, `VectorPoint`)
whose own field constraints (non-empty labels, required fields) already
reject a malformed response at construction time, before this module's
own logic ever sees it. A second, redundant check here would just
re-verify what the type system already enforced.
"""

import logging
from collections.abc import Sequence

from backend.retrieval.exceptions import (
    BundleConsistencyError,
    DuplicateCandidateError,
    DuplicateEvidenceError,
    GraphCycleError,
    MissingKnowledgeUnitError,
    RankingConsistencyError,
    TraceCompletenessError,
)
from backend.retrieval.models import (
    EvidenceBundle,
    EvidenceGroup,
    GraphPath,
    RetrievalCandidate,
    ScoredCandidate,
)

logger = logging.getLogger(__name__)

_EXPECTED_TRACE_PHASES = ("candidate_generation", "expansion", "evaluation", "assembly")


class RetrievalValidator:
    """Verifies structural invariants at every retrieval phase boundary."""

    def validate_candidates(self, candidates: Sequence[RetrievalCandidate]) -> None:
        """Verify no knowledge unit appears more than once in a candidate pool.

        Args:
            candidates: The candidate pool to check.

        Raises:
            DuplicateCandidateError: The same knowledge unit appears twice.
        """
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate.knowledge_unit_id)
            if key in seen:
                raise DuplicateCandidateError(knowledge_unit_id=key)
            seen.add(key)

    def validate_graph_path(self, path: GraphPath) -> None:
        """Verify a traversal path never revisits a node -- i.e., contains no cycle.

        Unreachable via `GraphExpander` today (its global visited-set makes
        a cycle structurally impossible), but fails loudly rather than
        silently accepting corrupted data if a future change to the
        expander ever violates that invariant.

        Args:
            path: The traversal path to check.

        Raises:
            GraphCycleError: A node id appears more than once along the path.
        """
        if not path.hops:
            return
        visited = {path.hops[0].source_id}
        for hop in path.hops:
            if hop.target_id in visited:
                raise GraphCycleError(node_id=hop.target_id)
            visited.add(hop.target_id)

    def validate_ranking(self, scored_candidates: Sequence[ScoredCandidate]) -> None:
        """Verify final ranks form a contiguous 1..N sequence in descending score order.

        Args:
            scored_candidates: Every scored candidate, in the order they
                will be presented (best first).

        Raises:
            RankingConsistencyError: Ranks are not a contiguous 1..N
                sequence, or scores are not non-increasing in rank order.
        """
        expected_ranks = list(range(1, len(scored_candidates) + 1))
        actual_ranks = [scored.ranking.final_rank for scored in scored_candidates]
        if actual_ranks != expected_ranks:
            raise RankingConsistencyError(
                reason=f"expected ranks {expected_ranks}, found {actual_ranks}"
            )
        scores = [scored.ranking.fused_score for scored in scored_candidates]
        if any(scores[i] < scores[i + 1] for i in range(len(scores) - 1)):
            raise RankingConsistencyError(reason="fused scores are not non-increasing by rank")

    def validate_evidence_groups(
        self, groups: Sequence[EvidenceGroup], candidates: Sequence[RetrievalCandidate]
    ) -> None:
        """Verify evidence groups are internally consistent and faithful to the candidate pool.

        Args:
            groups: The assembled evidence groups.
            candidates: The full candidate pool groups were assembled from.

        Raises:
            DuplicateEvidenceError: A candidate appears in more than one group.
            MissingKnowledgeUnitError: A group references a candidate
                absent from the candidate pool.
        """
        known_ids = {str(candidate.knowledge_unit_id) for candidate in candidates}
        seen: set[str] = set()
        for group in groups:
            members = (group.primary, *group.supporting)
            for member in members:
                key = str(member.candidate.knowledge_unit_id)
                if key not in known_ids:
                    raise MissingKnowledgeUnitError(knowledge_unit_id=key)
                if key in seen:
                    raise DuplicateEvidenceError(knowledge_unit_id=key)
                seen.add(key)

    def validate_bundle(self, bundle: EvidenceBundle) -> None:
        """Verify the fully assembled bundle is internally consistent.

        Args:
            bundle: The complete evidence bundle.

        Raises:
            BundleConsistencyError: The manifest's recorded statistics
                don't match the bundle's actual contents.
            TraceCompletenessError: The trace does not cover every
                expected phase.
        """
        actual_groups = len(bundle.evidence_groups)
        if bundle.manifest.statistics.evidence_groups != actual_groups:
            raise BundleConsistencyError(
                reason=(
                    f"manifest reports {bundle.manifest.statistics.evidence_groups} evidence "
                    f"groups, bundle has {actual_groups}"
                )
            )
        actual_items = sum(1 + len(group.supporting) for group in bundle.evidence_groups)
        if bundle.manifest.statistics.evidence_items != actual_items:
            raise BundleConsistencyError(
                reason=(
                    f"manifest reports {bundle.manifest.statistics.evidence_items} evidence "
                    f"items, bundle has {actual_items}"
                )
            )
        traced_phases = {phase.phase for phase in bundle.trace.phases}
        missing_phases = sorted(set(_EXPECTED_TRACE_PHASES) - traced_phases)
        if missing_phases:
            raise TraceCompletenessError(missing_phases=missing_phases)
