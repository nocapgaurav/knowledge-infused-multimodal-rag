"""Phase 4: Evidence Assembly.

Groups ranked candidates into coherent evidence: a primary finding plus
whatever it is directly, deterministically connected to (its cited
references, the figure/table it mentions, its adjacent context) --
exactly the "Paragraph + Figure + Table + Supporting References" grouping
the module is required to produce when appropriate.

Diversity is enforced structurally rather than via embedding similarity:
each `ScoredCandidate`'s graph path already records, for a graph-
discovered candidate, whether its immediate predecessor (`hops[-1]
.source_id`) is a given primary -- so "is B directly connected to primary
P" is a lookup, not a new graph query or a similarity computation. And
rather than comparing candidate text for near-duplicates (which would
need embeddings this phase doesn't have, and would reintroduce exactly
the "what threshold?" arbitrary-constant problem RRF was chosen to avoid
in Phase 3), this module caps how many *primary* candidates may come from
the same section. In this corpus, near-duplicate evidence is
overwhelmingly adjacent or co-located text -- capping primaries per
section is a deterministic, already-available proxy for that, not a
weaker version of semantic deduplication.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass

from backend.retrieval.models import DroppedCandidate, EvidenceGroup, ScoredCandidate

logger = logging.getLogger(__name__)

_ASSEMBLY_PHASE_NAME = "assembly"


@dataclass(frozen=True)
class AssemblyBudget:
    """Limits on how many evidence groups assembly may produce.

    Attributes:
        max_evidence_groups: Maximum number of evidence groups in the final bundle.
        max_primaries_per_section: Maximum number of groups whose primary
            candidate comes from the same section -- the mechanism that
            keeps evidence from collapsing onto one over-represented passage.
    """

    max_evidence_groups: int = 5
    max_primaries_per_section: int = 2


@dataclass(frozen=True)
class AssemblyResult:
    """The outcome of one assembly call.

    Attributes:
        groups: Assembled evidence groups, ranked best-first.
        dropped: Candidates considered but excluded, with why.
    """

    groups: list[EvidenceGroup]
    dropped: list[DroppedCandidate]


class EvidenceAssembler:
    """Assembles ranked candidates into coherent, diverse evidence groups."""

    def assemble(
        self, scored_candidates: list[ScoredCandidate], budget: AssemblyBudget
    ) -> AssemblyResult:
        """Assemble a ranked candidate pool into evidence groups.

        Args:
            scored_candidates: Every scored candidate, ordered best-first
                (Phase 3's output).
            budget: Limits on group count and per-section primary count.

        Returns:
            The assembled groups and every excluded candidate, with why.
        """
        claimed: set[str] = set()
        section_primary_counts: dict[str, int] = defaultdict(int)
        groups: list[EvidenceGroup] = []
        dropped: list[DroppedCandidate] = []

        for scored in scored_candidates:
            candidate_id = str(scored.candidate.knowledge_unit_id)
            if candidate_id in claimed:
                continue

            if len(groups) >= budget.max_evidence_groups:
                dropped.append(_dropped(candidate_id, "evidence group budget reached"))
                continue

            section_key = _diversity_bucket(scored)
            if section_primary_counts[section_key] >= budget.max_primaries_per_section:
                dropped.append(
                    _dropped(
                        candidate_id,
                        f"section {section_key} already has "
                        f"{budget.max_primaries_per_section} primary evidence group(s)",
                    )
                )
                continue

            supporting = [
                other
                for other in scored_candidates
                if str(other.candidate.knowledge_unit_id) not in claimed
                and str(other.candidate.knowledge_unit_id) != candidate_id
                and _is_direct_neighbor_of(other, candidate_id)
            ]

            claimed.add(candidate_id)
            for member in supporting:
                claimed.add(str(member.candidate.knowledge_unit_id))
            section_primary_counts[section_key] += 1

            modalities = tuple(
                dict.fromkeys(
                    [
                        scored.candidate.modality,
                        *(member.candidate.modality for member in supporting),
                    ]
                )
            )
            groups.append(
                EvidenceGroup(
                    group_id=candidate_id,
                    primary=scored,
                    supporting=tuple(supporting),
                    modalities=modalities,
                )
            )

        for scored in scored_candidates:
            candidate_id = str(scored.candidate.knowledge_unit_id)
            if candidate_id not in claimed and not any(
                d.knowledge_unit_id == candidate_id for d in dropped
            ):
                dropped.append(
                    _dropped(candidate_id, "not selected: outranked by claimed evidence")
                )

        logger.info(
            "evidence assembled",
            extra={"groups": len(groups), "dropped": len(dropped)},
        )
        return AssemblyResult(groups=groups, dropped=dropped)


def _diversity_bucket(scored: ScoredCandidate) -> str:
    """The diversity bucket a candidate's primary slot counts against.

    Sectioned chunks bucket by section, as before. Section-less chunks
    bucket by their structural-identity *family* ("Bibliography
    reference [7]" -> "Bibliography reference"), so twenty near-identical
    bibliography entries share one capped bucket -- observed live: they
    flooded four of five evidence groups for an author-style question --
    while the title, abstract, keywords, and author block each remain
    individually eligible rather than competing for the same two slots.
    """
    if scored.candidate.section_id is not None:
        return str(scored.candidate.section_id)
    context = scored.candidate.retrieval_context
    if context:
        return "ctx:" + context.split("[")[0].strip()
    return "__no_section__"


def _is_direct_neighbor_of(scored: ScoredCandidate, primary_id: str) -> bool:
    hops = scored.candidate.graph_path.hops
    return bool(hops) and hops[-1].source_id == primary_id


def _dropped(knowledge_unit_id: str, reason: str) -> DroppedCandidate:
    return DroppedCandidate(
        knowledge_unit_id=knowledge_unit_id, phase=_ASSEMBLY_PHASE_NAME, reason=reason
    )
