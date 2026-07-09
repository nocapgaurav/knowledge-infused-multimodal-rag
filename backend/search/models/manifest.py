"""IndexManifest: describes one complete indexing run for a document."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.domain import PaperId
from backend.search.models.vector_point import DistanceMetric


class IndexManifest(BaseModel):
    """Describes one complete indexing run for a document.

    Persisted at `data/index/<document_id>/index_manifest.json` -- the
    canonical input future retrieval modules read to know which collection
    holds a document's vectors, without recomputing anything.

    Attributes:
        document_id: Identifier of the indexed document.
        collection_name: Name of the collection these vectors were written to.
        vector_dimension: Dimension of the indexed vectors.
        distance_metric: Similarity metric the collection is configured with.
        embedding_model: Name of the embedding model that produced these vectors.
        embedding_version: Resolved revision of the embedding model.
        artifact_version: Schema version of this persisted manifest shape.
        source_embedding_manifest: Hash of the embedding manifest
            (`manifest.json`) these vectors were indexed from. Comparing
            this against a freshly computed hash is how staleness is detected.
        checksum: Hash of the embedding artifacts (`embeddings.json`)
            actually indexed -- distinct from `source_embedding_manifest`,
            which hashes the summary metadata file; this hashes the vector
            data itself.
        indexed_vectors: Number of vectors successfully indexed.
        failed_vectors: Number of vectors that failed to index after retries.
        created_at: Timestamp this manifest was generated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: PaperId
    collection_name: str
    vector_dimension: int
    distance_metric: DistanceMetric
    embedding_model: str
    embedding_version: str
    artifact_version: str
    source_embedding_manifest: str
    checksum: str
    indexed_vectors: int = Field(ge=0)
    failed_vectors: int = Field(ge=0)
    created_at: datetime
