"""Builds the payload accompanying each indexed vector."""

from collections import defaultdict
from collections.abc import Sequence

from backend.domain import Chunk, ChunkId, Relationship, RelationshipType
from backend.embeddings.models import EmbeddingArtifact, EmbeddingManifest
from backend.search.models import VectorPoint


def build_citation_counts(relationships: Sequence[Relationship]) -> dict[ChunkId, int]:
    """Count incoming `CITES` relationships per target chunk.

    Args:
        relationships: All relationships for a paper.

    Returns:
        A mapping of chunk id to the number of `CITES` relationships
        targeting it. Chunks with no incoming citations are absent from
        the mapping -- callers should treat a missing key as zero.
    """
    counts: dict[ChunkId, int] = defaultdict(int)
    for relationship in relationships:
        if relationship.relationship_type is RelationshipType.CITES:
            counts[relationship.target_chunk_id] += 1
    return dict(counts)


class PayloadBuilder:
    """Builds a `VectorPoint` (vector plus full payload) for each embedding artifact."""

    def build_point(
        self,
        artifact: EmbeddingArtifact,
        chunk: Chunk,
        citation_count: int,
        manifest: EmbeddingManifest,
    ) -> VectorPoint:
        """Build the vector point for a single embedding artifact.

        Args:
            artifact: The embedding artifact to build a point for.
            chunk: The knowledge unit this artifact was computed from.
            citation_count: Number of `CITES` relationships targeting this chunk.
            manifest: The embedding manifest this artifact belongs to.

        Returns:
            A `VectorPoint` with vector and payload fully populated.
        """
        page_numbers = sorted({box.page_number for box in chunk.bounding_boxes})
        payload = {
            "knowledge_unit_id": str(chunk.id),
            "document_id": str(chunk.paper_id),
            "section_id": str(chunk.section_id) if chunk.section_id else None,
            "modality": chunk.modality.value,
            "embedding_target": artifact.target.value,
            "page_numbers": page_numbers,
            "reading_order": chunk.order,
            "citation_count": citation_count,
            "artifact_version": artifact.artifact_version,
            "embedding_model": artifact.model_name,
            "embedding_version": artifact.model_version,
            "source_document_checksum": manifest.source_representation_version,
            "text": chunk.text,
            "token_count": chunk.token_count,
            "asset_uri": chunk.asset_uri,
        }
        return VectorPoint(id=artifact.knowledge_unit_id, vector=artifact.vector, payload=payload)
