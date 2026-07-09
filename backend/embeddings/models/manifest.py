"""EmbeddingManifest: describes one complete embedding run for a document."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from backend.domain import PaperId


class EmbeddingManifest(BaseModel):
    """Describes one complete embedding generation run for a document.

    Persisted alongside the embeddings themselves -- this is the file
    future modules read to determine staleness using only metadata, with
    no need to load or recompute any embedding just to check whether it's
    still valid.

    Attributes:
        document_id: Identifier of the represented document.
        model_name: Name of the text embedding model used.
        model_version: Resolved, concrete revision of the text embedding model.
        embedding_dimension: Dimension of text embeddings produced.
        artifact_version: Schema version of the persisted embedding artifacts.
        source_representation_version: Hash of the knowledge representation
            snapshot these embeddings were computed from.
        embedding_count: Number of embedding artifacts successfully produced.
        failed_count: Number of knowledge units that failed to embed.
        skipped_image_count: Number of image embeddings skipped because no
            image embedding provider was configured.
        created_at: Timestamp this manifest was generated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: PaperId
    model_name: str
    model_version: str
    embedding_dimension: int
    artifact_version: str
    source_representation_version: str
    embedding_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    skipped_image_count: int = Field(ge=0)
    created_at: datetime
