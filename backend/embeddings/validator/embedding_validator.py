"""Structural validation of a freshly generated set of embedding artifacts."""

from backend.domain import Chunk, PaperId
from backend.embeddings.exceptions import (
    DimensionMismatchError,
    DuplicateEmbeddingError,
    NoEmbeddingsProducedError,
    UnknownKnowledgeUnitError,
)
from backend.embeddings.models import EmbeddingArtifact


def validate_embeddings(
    document_id: PaperId, chunks: list[Chunk], artifacts: list[EmbeddingArtifact]
) -> None:
    """Validate a freshly generated set of embedding artifacts before persistence.

    Args:
        document_id: Identifier of the document the artifacts belong to.
        chunks: Knowledge units the artifacts were generated from.
        artifacts: Embedding artifacts to validate.

    Raises:
        NoEmbeddingsProducedError: No artifacts were produced at all.
        UnknownKnowledgeUnitError: An artifact references a chunk that
            doesn't exist among `chunks`.
        DuplicateEmbeddingError: The same (knowledge unit, target) pair
            appears more than once.
        DimensionMismatchError: Artifacts for the same target and model
            report inconsistent vector dimensions.
    """
    _validate_not_empty(document_id, artifacts)
    _validate_known_knowledge_units(document_id, chunks, artifacts)
    _validate_no_duplicates(document_id, artifacts)
    _validate_consistent_dimensions(document_id, artifacts)


def _validate_not_empty(document_id: PaperId, artifacts: list[EmbeddingArtifact]) -> None:
    if not artifacts:
        raise NoEmbeddingsProducedError(document_id=document_id)


def _validate_known_knowledge_units(
    document_id: PaperId, chunks: list[Chunk], artifacts: list[EmbeddingArtifact]
) -> None:
    chunk_ids = {chunk.id for chunk in chunks}
    for artifact in artifacts:
        if artifact.knowledge_unit_id not in chunk_ids:
            raise UnknownKnowledgeUnitError(
                document_id=document_id,
                reason=(
                    f"artifact {artifact.embedding_id} references unknown chunk "
                    f"{artifact.knowledge_unit_id}"
                ),
            )


def _validate_no_duplicates(document_id: PaperId, artifacts: list[EmbeddingArtifact]) -> None:
    seen: set[tuple[object, object]] = set()
    for artifact in artifacts:
        key = (artifact.knowledge_unit_id, artifact.target)
        if key in seen:
            raise DuplicateEmbeddingError(
                document_id=document_id,
                reason=(
                    f"chunk {artifact.knowledge_unit_id} has more than one "
                    f"{artifact.target} embedding"
                ),
            )
        seen.add(key)


def _validate_consistent_dimensions(
    document_id: PaperId, artifacts: list[EmbeddingArtifact]
) -> None:
    dimension_by_target_model: dict[tuple[object, str, str], int] = {}
    for artifact in artifacts:
        key = (artifact.target, artifact.model_name, artifact.model_version)
        expected = dimension_by_target_model.setdefault(key, artifact.embedding_dimension)
        if (
            artifact.embedding_dimension != expected
            or len(artifact.vector) != artifact.embedding_dimension
        ):
            raise DimensionMismatchError(
                document_id=document_id,
                reason=(
                    f"artifact {artifact.embedding_id} has dimension "
                    f"{artifact.embedding_dimension} (vector length {len(artifact.vector)}), "
                    f"expected {expected}"
                ),
            )
