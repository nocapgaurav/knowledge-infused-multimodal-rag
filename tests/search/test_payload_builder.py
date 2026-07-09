"""Tests for building the payload accompanying each indexed vector."""

from datetime import UTC, datetime
from uuid import uuid4

from backend.domain import (
    BoundingBox,
    ChunkModality,
    PaperId,
    Relationship,
    RelationshipType,
    SectionId,
)
from backend.domain import Chunk as DomainChunk
from backend.embeddings.models import (
    EmbeddingArtifact,
    EmbeddingId,
    EmbeddingManifest,
    EmbeddingTarget,
)
from backend.search.payload.payload_builder import PayloadBuilder, build_citation_counts


def _manifest(paper_id: PaperId) -> EmbeddingManifest:
    return EmbeddingManifest(
        document_id=paper_id,
        model_name="BAAI/bge-m3",
        model_version="sha-1",
        embedding_dimension=4,
        artifact_version="1.0",
        source_representation_version="repr-hash",
        embedding_count=1,
        failed_count=0,
        skipped_image_count=0,
        created_at=datetime.now(UTC),
    )


def _artifact(paper_id: PaperId, chunk: DomainChunk) -> EmbeddingArtifact:
    return EmbeddingArtifact(
        embedding_id=EmbeddingId(uuid4()),
        knowledge_unit_id=chunk.id,
        paper_id=paper_id,
        target=EmbeddingTarget.TEXT,
        vector=[0.1, 0.2, 0.3, 0.4],
        model_name="BAAI/bge-m3",
        model_version="sha-1",
        embedding_dimension=4,
        checksum="abc",
        artifact_version="1.0",
        source_representation_version="repr-hash",
        created_at=datetime.now(UTC),
    )


def test_build_point_populates_every_required_field() -> None:
    paper_id = PaperId(uuid4())
    section_id = SectionId(uuid4())
    chunk = DomainChunk(
        paper_id=paper_id,
        section_id=section_id,
        order=3,
        modality=ChunkModality.TEXT,
        text="This chunk discusses an important finding.",
        token_count=7,
        bounding_boxes=[
            BoundingBox(page_number=2, x0=0, y0=0, x1=10, y1=10),
            BoundingBox(page_number=1, x0=0, y0=0, x1=10, y1=10),
        ],
    )
    artifact = _artifact(paper_id, chunk)
    manifest = _manifest(paper_id)

    point = PayloadBuilder().build_point(artifact, chunk, citation_count=5, manifest=manifest)

    assert point.id == chunk.id
    assert point.vector == artifact.vector
    assert point.payload["knowledge_unit_id"] == str(chunk.id)
    assert point.payload["document_id"] == str(paper_id)
    assert point.payload["section_id"] == str(section_id)
    assert point.payload["modality"] == "text"
    assert point.payload["embedding_target"] == "text"
    assert point.payload["page_numbers"] == [1, 2]  # deduplicated and sorted
    assert point.payload["reading_order"] == 3
    assert point.payload["citation_count"] == 5
    assert point.payload["artifact_version"] == "1.0"
    assert point.payload["embedding_model"] == "BAAI/bge-m3"
    assert point.payload["embedding_version"] == "sha-1"
    assert point.payload["source_document_checksum"] == "repr-hash"
    assert point.payload["text"] == chunk.text
    assert point.payload["token_count"] == 7
    assert point.payload["asset_uri"] is None


def test_build_point_handles_missing_section_and_asset() -> None:
    paper_id = PaperId(uuid4())
    chunk = DomainChunk(
        paper_id=paper_id, order=0, modality=ChunkModality.TEXT, text="unsectioned text"
    )
    artifact = _artifact(paper_id, chunk)

    point = PayloadBuilder().build_point(
        artifact, chunk, citation_count=0, manifest=_manifest(paper_id)
    )

    assert point.payload["section_id"] is None
    assert point.payload["page_numbers"] == []


def test_citation_counts_only_count_cites_relationships() -> None:
    paper_id = PaperId(uuid4())
    target_a = uuid4()
    target_b = uuid4()
    relationships = [
        Relationship(
            paper_id=paper_id,
            source_chunk_id=uuid4(),
            target_chunk_id=target_a,
            relationship_type=RelationshipType.CITES,
        ),
        Relationship(
            paper_id=paper_id,
            source_chunk_id=uuid4(),
            target_chunk_id=target_a,
            relationship_type=RelationshipType.CITES,
        ),
        Relationship(
            paper_id=paper_id,
            source_chunk_id=uuid4(),
            target_chunk_id=target_b,
            relationship_type=RelationshipType.REFERENCES,
        ),
    ]

    counts = build_citation_counts(relationships)

    assert counts[target_a] == 2
    assert target_b not in counts  # REFERENCES, not CITES -- not counted


def test_citation_counts_for_uncited_chunk_defaults_to_zero_via_get() -> None:
    counts = build_citation_counts([])

    assert counts.get(uuid4(), 0) == 0
