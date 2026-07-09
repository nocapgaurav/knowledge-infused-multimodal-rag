"""GraphManifest: describes one complete graph construction run for a document."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.domain import PaperId


class GraphManifest(BaseModel):
    """Describes one complete graph construction run for a document.

    Persisted at `data/graph/<document_id>/graph_manifest.json` -- the
    canonical record future modules (and this module, on its next run)
    read to know whether an existing graph is still fresh, without
    recomputing anything.

    Attributes:
        document_id: Identifier of the document this graph represents.
        artifact_version: Schema version of this persisted manifest shape.
        graph_version: Version of this module's own construction rules
            (which chunks become which nodes, which relationships become
            which edges). Bumped when those rules change, independently of
            whether the upstream knowledge representation has -- this is
            what lets a graph be detected as stale purely because the
            *logic* that built it is outdated, with no change to
            `knowledge_units.json`/`relationships.json` at all.
        node_count: Number of nodes in the graph this manifest describes.
        relationship_count: Number of edges in the graph this manifest describes.
        checksum: Hash of the graph's own content (nodes and edges) --
            distinct from `source_representation_version`, which hashes the
            *input* this graph was built from.
        source_representation_version: Hash of the knowledge representation
            (`knowledge_units.json` + `relationships.json`) this graph was
            built from. Comparing this against a freshly computed hash is
            how upstream staleness is detected.
        created_at: Timestamp this manifest was generated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: PaperId
    artifact_version: str
    graph_version: str
    node_count: int = Field(ge=0)
    relationship_count: int = Field(ge=0)
    checksum: str
    source_representation_version: str
    created_at: datetime
