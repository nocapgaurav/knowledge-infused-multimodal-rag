"""Hybrid retrieval's own data models -- not part of `backend.domain` for
the same reason `EmbeddingArtifact`/`IndexManifest`/`GraphManifest` aren't:
this is versioned, strategy-dependent infrastructure output, not a
permanent fact about a paper. Deliberately expose no Qdrant or Neo4j type.
"""

from backend.retrieval.models.evidence_bundle import EvidenceBundle
from backend.retrieval.models.evidence_group import EvidenceGroup
from backend.retrieval.models.graph_path import (
    GraphNeighbor,
    GraphPath,
    TraversalDirection,
    TraversalHop,
)
from backend.retrieval.models.ranking import RankingExplanation, ScoredCandidate, SignalScore
from backend.retrieval.models.retrieval_candidate import DiscoveryMethod, RetrievalCandidate
from backend.retrieval.models.retrieval_manifest import RetrievalManifest, RetrievalStatistics
from backend.retrieval.models.retrieval_trace import DroppedCandidate, PhaseTrace, RetrievalTrace

__all__ = [
    "DiscoveryMethod",
    "DroppedCandidate",
    "EvidenceBundle",
    "EvidenceGroup",
    "GraphNeighbor",
    "GraphPath",
    "PhaseTrace",
    "RankingExplanation",
    "RetrievalCandidate",
    "RetrievalManifest",
    "RetrievalStatistics",
    "RetrievalTrace",
    "ScoredCandidate",
    "SignalScore",
    "TraversalDirection",
    "TraversalHop",
]
