"""Tests for Phase 4: evidence assembly."""

from uuid import uuid4

from backend.retrieval.assembly.evidence_assembler import AssemblyBudget, EvidenceAssembler
from backend.retrieval.models import (
    DiscoveryMethod,
    GraphPath,
    RankingExplanation,
    RetrievalCandidate,
    ScoredCandidate,
    SignalScore,
    TraversalHop,
)


def _scored(
    document_id,
    section_id=None,
    rank=1,
    score=1.0,
    depth=0,
    source_id=None,
) -> ScoredCandidate:
    candidate_id = uuid4()
    hops = ()
    if depth > 0:
        assert source_id is not None
        hops = (
            TraversalHop(
                source_id=source_id,
                target_id=str(candidate_id),
                relationship_type="CITES",
                direction="outgoing",
            ),
        )
    candidate = RetrievalCandidate(
        knowledge_unit_id=candidate_id,
        document_id=document_id,
        section_id=section_id,
        modality="text",
        text="text",
        asset_uri=None,
        reading_order=0,
        citation_count=0,
        dense_similarity=0.5 if depth == 0 else None,
        discovery_method=(
            DiscoveryMethod.DENSE_RETRIEVAL if depth == 0 else DiscoveryMethod.GRAPH_EXPANSION
        ),
        graph_path=GraphPath(hops=hops),
    )
    return ScoredCandidate(
        candidate=candidate,
        ranking=RankingExplanation(
            signals=(SignalScore(name="dense_similarity", raw_value=score, rank=rank),),
            fused_score=score,
            final_rank=rank,
        ),
    )


def test_assemble_creates_one_group_per_unconnected_candidate() -> None:
    document_id = uuid4()
    a = _scored(document_id, rank=1)
    b = _scored(document_id, rank=2)

    result = EvidenceAssembler().assemble([a, b], AssemblyBudget())

    assert len(result.groups) == 2
    assert result.groups[0].primary.candidate.knowledge_unit_id == a.candidate.knowledge_unit_id
    assert result.groups[0].supporting == ()


def test_direct_graph_neighbor_becomes_supporting_not_its_own_group() -> None:
    document_id = uuid4()
    primary = _scored(document_id, rank=1)
    supporting = _scored(
        document_id, rank=2, depth=1, source_id=str(primary.candidate.knowledge_unit_id)
    )

    result = EvidenceAssembler().assemble([primary, supporting], AssemblyBudget())

    assert len(result.groups) == 1
    assert (
        result.groups[0].primary.candidate.knowledge_unit_id == primary.candidate.knowledge_unit_id
    )
    assert [m.candidate.knowledge_unit_id for m in result.groups[0].supporting] == [
        supporting.candidate.knowledge_unit_id
    ]
    assert result.dropped == []


def test_max_evidence_groups_budget_caps_group_count() -> None:
    document_id = uuid4()
    candidates = [_scored(document_id, rank=i + 1) for i in range(5)]

    result = EvidenceAssembler().assemble(candidates, AssemblyBudget(max_evidence_groups=2))

    assert len(result.groups) == 2
    assert len(result.dropped) == 3
    assert all(d.reason == "evidence group budget reached" for d in result.dropped[:3])


def test_max_primaries_per_section_diversifies_across_sections() -> None:
    document_id = uuid4()
    section = uuid4()
    candidates = [_scored(document_id, section_id=section, rank=i + 1) for i in range(4)]

    result = EvidenceAssembler().assemble(
        candidates, AssemblyBudget(max_evidence_groups=10, max_primaries_per_section=2)
    )

    assert len(result.groups) == 2
    assert len(result.dropped) == 2


def test_no_candidate_appears_in_more_than_one_group() -> None:
    document_id = uuid4()
    primary_a = _scored(document_id, rank=1)
    shared_candidate = _scored(
        document_id, rank=2, depth=1, source_id=str(primary_a.candidate.knowledge_unit_id)
    )
    primary_b = _scored(document_id, rank=3)

    result = EvidenceAssembler().assemble(
        [primary_a, shared_candidate, primary_b], AssemblyBudget()
    )

    all_member_ids = [
        str(member.candidate.knowledge_unit_id)
        for group in result.groups
        for member in (group.primary, *group.supporting)
    ]
    assert len(all_member_ids) == len(set(all_member_ids))


def test_modalities_include_primary_and_supporting() -> None:
    document_id = uuid4()
    primary = _scored(document_id, rank=1)

    result = EvidenceAssembler().assemble([primary], AssemblyBudget())

    assert result.groups[0].modalities == ("text",)
