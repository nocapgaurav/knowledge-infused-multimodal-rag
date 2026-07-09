"""EmbeddingArtifact: a single, fully self-describing embedding vector.

Deliberately not part of `backend.domain`. A `Relationship` (Module 5) is
permanent domain knowledge -- a citation is a citation regardless of which
embedding model ever runs. An `EmbeddingArtifact` is the opposite: versioned,
model-dependent, disposable infrastructure output that gets superseded as
models evolve. Keeping it out of the shared domain layer is what "every
external AI component must be replaceable" and "business logic must never
depend on specific AI models" require in practice -- putting it in
`backend.domain` would mean every domain-layer consumer, including modules
with nothing to do with embeddings, transitively carries ML-model-shaped
fields.
"""

from datetime import datetime
from enum import StrEnum
from typing import NewType
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from backend.domain import ChunkId, PaperId

EmbeddingId = NewType("EmbeddingId", UUID)


class EmbeddingTarget(StrEnum):
    """What kind of content an embedding vector represents.

    A single knowledge unit can produce more than one embedding: a figure's
    caption text (`TEXT`) and its rendered image (`IMAGE`) live in
    different vector spaces and are never combined into one artifact.
    """

    TEXT = "text"
    IMAGE = "image"


class EmbeddingArtifact(BaseModel):
    """A single embedding vector, fully self-describing for reproducibility.

    Attributes:
        embedding_id: Unique identifier for this specific artifact.
        knowledge_unit_id: Identifier of the chunk this embedding represents.
        paper_id: Identifier of the paper the chunk belongs to.
        target: Which kind of content this embedding represents.
        vector: The embedding vector.
        model_name: Name of the model that produced this vector (e.g. "BAAI/bge-m3").
        model_version: Resolved, concrete model revision (e.g. a HuggingFace
            commit SHA) -- never a floating "latest" label, so that "same
            model_version" is a guarantee of identical weights.
        embedding_dimension: Length of `vector`.
        checksum: SHA-256 hex digest of the exact source content (text or
            image bytes) that was embedded. Comparing this against a
            freshly computed hash of the chunk's current content is how
            staleness is detected.
        artifact_version: Schema version of this persisted shape,
            independent of the model.
        source_representation_version: Hash of the knowledge representation
            snapshot this embedding was computed from. Shared by every
            artifact produced in the same embedding run.
        created_at: Timestamp this artifact was generated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    embedding_id: EmbeddingId
    knowledge_unit_id: ChunkId
    paper_id: PaperId
    target: EmbeddingTarget
    vector: list[float]
    model_name: str
    model_version: str
    embedding_dimension: int
    checksum: str
    artifact_version: str
    source_representation_version: str
    created_at: datetime
