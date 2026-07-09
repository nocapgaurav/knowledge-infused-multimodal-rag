"""Tests for Phase 2: evidence expansion, using fakes -- no real services needed."""

from collections.abc import Sequence
from uuid import UUID, uuid4

from backend.retrieval.expansion.graph_expander import ExpansionBudget, GraphExpander
from backend.retrieval.interfaces.graph_retriever import GraphRetriever
from backend.retrieval.interfaces.vector_retriever import VectorRetriever
from backend.retrieval.models import DiscoveryMethod, GraphNeighbor, TraversalDirection
from backend.retrieval.models.retrieval_candidate import RetrievalCandidate
from backend.search.models import VectorPoint


class _FakeGraphRetriever(GraphRetriever):
    """A small, hand-built directed graph: edges are (source, type, target)."""

    def __init__(
        self, edges: list[tuple[str, str, str]], labels: dict[str, tuple[str, ...]]
    ) -> None:
        self._edges = edges
        self._labels = labels
        self.call_count = 0

    def neighbors(
        self,
        node_ids: Sequence[str],
        relationship_types: Sequence[str],
        direction: TraversalDirection,
    ) -> list[GraphNeighbor]:
        self.call_count += 1
        node_id_set = set(node_ids)
        rel_set = set(relationship_types)
        found: list[GraphNeighbor] = []
        for source, rel, target in self._edges:
            if rel not in rel_set:
                continue
            if direction is TraversalDirection.OUTGOING and source in node_id_set:
                found.append(
                    GraphNeighbor(
                        source_id=source,
                        neighbor_id=target,
                        neighbor_labels=self._labels[target],
                        relationship_type=rel,
                        direction=direction,
                    )
                )
            elif direction is TraversalDirection.INCOMING and target in node_id_set:
                found.append(
                    GraphNeighbor(
                        source_id=target,
                        neighbor_id=source,
                        neighbor_labels=self._labels[source],
                        relationship_type=rel,
                        direction=direction,
                    )
                )
        return found


class _FakeVectorRetriever(VectorRetriever):
    def __init__(self, document_id: UUID) -> None:
        self._document_id = document_id
        self.retrieved_ids: list[UUID] = []

    def search(self, *args: object, **kwargs: object) -> list:  # pragma: no cover - unused
        raise NotImplementedError

    def retrieve_by_ids(self, collection: str, ids: Sequence[UUID]) -> list[VectorPoint]:
        self.retrieved_ids = list(ids)
        return [
            VectorPoint(
                id=node_id,
                vector=[0.1, 0.2, 0.3, 0.4],
                payload={
                    "document_id": str(self._document_id),
                    "section_id": None,
                    "modality": "text",
                    "text": f"text-{node_id}",
                    "asset_uri": None,
                    "reading_order": 0,
                    "citation_count": 0,
                },
            )
            for node_id in ids
        ]


def _seed(document_id: UUID, knowledge_unit_id: UUID) -> RetrievalCandidate:
    return RetrievalCandidate(
        knowledge_unit_id=knowledge_unit_id,
        document_id=document_id,
        section_id=None,
        modality="text",
        text="seed text",
        asset_uri=None,
        reading_order=0,
        citation_count=0,
        dense_similarity=0.9,
        discovery_method=DiscoveryMethod.DENSE_RETRIEVAL,
    )


def test_expand_discovers_a_direct_next_neighbor() -> None:
    document_id = uuid4()
    a, b = str(uuid4()), str(uuid4())
    graph = _FakeGraphRetriever(
        edges=[(a, "NEXT", b)],
        labels={a: ("KnowledgeUnit",), b: ("KnowledgeUnit",)},
    )
    vectors = _FakeVectorRetriever(document_id)
    expander = GraphExpander(graph, vectors)

    result = expander.expand([_seed(document_id, UUID(a))], "collection", ExpansionBudget())

    assert [str(c.knowledge_unit_id) for c in result.candidates] == [b]
    assert result.candidates[0].discovery_method is DiscoveryMethod.GRAPH_EXPANSION
    assert result.candidates[0].dense_similarity is None
    assert result.candidates[0].graph_path.depth == 1
    assert result.candidates[0].graph_path.hops[0].relationship_type == "NEXT"


def test_expand_passes_through_a_section_to_reach_a_sibling() -> None:
    document_id = uuid4()
    a, s, c = str(uuid4()), str(uuid4()), str(uuid4())
    graph = _FakeGraphRetriever(
        edges=[(a, "BELONGS_TO", s), (c, "BELONGS_TO", s)],
        labels={a: ("KnowledgeUnit",), s: ("Section",), c: ("KnowledgeUnit",)},
    )
    vectors = _FakeVectorRetriever(document_id)
    expander = GraphExpander(graph, vectors)

    result = expander.expand(
        [_seed(document_id, UUID(a))],
        "collection",
        ExpansionBudget(max_depth=2),
    )

    discovered_ids = {str(candidate.knowledge_unit_id) for candidate in result.candidates}
    assert discovered_ids == {c}  # the Section itself is never returned as evidence
    assert result.candidates[0].graph_path.depth == 2


def test_expand_never_revisits_a_node_cycle_safe() -> None:
    document_id = uuid4()
    a, b = str(uuid4()), str(uuid4())
    # a cycle: a -NEXT-> b, b -NEXT-> a
    graph = _FakeGraphRetriever(
        edges=[(a, "NEXT", b), (b, "NEXT", a)],
        labels={a: ("KnowledgeUnit",), b: ("KnowledgeUnit",)},
    )
    vectors = _FakeVectorRetriever(document_id)
    expander = GraphExpander(graph, vectors)

    result = expander.expand(
        [_seed(document_id, UUID(a))], "collection", ExpansionBudget(max_depth=5)
    )

    # only b is ever discovered; a (the seed) is never rediscovered despite the cycle
    assert [str(c.knowledge_unit_id) for c in result.candidates] == [b]


def test_max_depth_budget_stops_traversal() -> None:
    document_id = uuid4()
    a, b, c = str(uuid4()), str(uuid4()), str(uuid4())
    graph = _FakeGraphRetriever(
        edges=[(a, "NEXT", b), (b, "NEXT", c)],
        labels={a: ("KnowledgeUnit",), b: ("KnowledgeUnit",), c: ("KnowledgeUnit",)},
    )
    vectors = _FakeVectorRetriever(document_id)
    expander = GraphExpander(graph, vectors)

    result = expander.expand(
        [_seed(document_id, UUID(a))], "collection", ExpansionBudget(max_depth=1)
    )

    assert [str(candidate.knowledge_unit_id) for candidate in result.candidates] == [b]


def test_max_neighbors_per_node_caps_fan_out() -> None:
    document_id = uuid4()
    a = str(uuid4())
    neighbors = [str(uuid4()) for _ in range(5)]
    graph = _FakeGraphRetriever(
        edges=[(a, "NEXT", n) for n in neighbors],
        labels={a: ("KnowledgeUnit",), **{n: ("KnowledgeUnit",) for n in neighbors}},
    )
    vectors = _FakeVectorRetriever(document_id)
    expander = GraphExpander(graph, vectors)

    result = expander.expand(
        [_seed(document_id, UUID(a))],
        "collection",
        ExpansionBudget(max_neighbors_per_node=2),
    )

    assert len(result.candidates) == 2


def test_max_total_evidence_budget_stops_discovery_across_the_whole_traversal() -> None:
    document_id = uuid4()
    a = str(uuid4())
    neighbors = [str(uuid4()) for _ in range(5)]
    graph = _FakeGraphRetriever(
        edges=[(a, "NEXT", n) for n in neighbors],
        labels={a: ("KnowledgeUnit",), **{n: ("KnowledgeUnit",) for n in neighbors}},
    )
    vectors = _FakeVectorRetriever(document_id)
    expander = GraphExpander(graph, vectors)

    result = expander.expand(
        [_seed(document_id, UUID(a))],
        "collection",
        ExpansionBudget(max_neighbors_per_node=10, max_total_evidence=3),
    )

    assert len(result.candidates) == 3
    assert result.budget_exhausted == "max_total_evidence"


def test_expand_returns_no_budget_exhausted_when_traversal_completes_naturally() -> None:
    document_id = uuid4()
    a, b = str(uuid4()), str(uuid4())
    graph = _FakeGraphRetriever(
        edges=[(a, "NEXT", b)],
        labels={a: ("KnowledgeUnit",), b: ("KnowledgeUnit",)},
    )
    vectors = _FakeVectorRetriever(document_id)
    expander = GraphExpander(graph, vectors)

    result = expander.expand(
        [_seed(document_id, UUID(a))], "collection", ExpansionBudget(max_depth=5)
    )

    assert result.budget_exhausted is None


def test_expand_with_no_seeds_returns_no_candidates() -> None:
    document_id = uuid4()
    graph = _FakeGraphRetriever(edges=[], labels={})
    vectors = _FakeVectorRetriever(document_id)
    expander = GraphExpander(graph, vectors)

    result = expander.expand([], "collection", ExpansionBudget())

    assert result.candidates == []
    assert graph.call_count == 0
