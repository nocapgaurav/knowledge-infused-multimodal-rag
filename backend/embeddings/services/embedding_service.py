"""Orchestrates embedding generation.

Read the knowledge representation -> check staleness (skip if unchanged,
unless forced) -> plan targets -> embed in batches, with retry and
batch-level partial success -> validate -> persist. Each step (repository,
planner, provider, validator) is independently testable; this class's only
job is calling them in the right order.
"""

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from backend.domain import Chunk, PaperId
from backend.embeddings.exceptions import EmbeddingProviderError, NoEmbeddingsProducedError
from backend.embeddings.interfaces.embedding_provider import (
    EmbeddingProvider,
    ImageEmbeddingProvider,
)
from backend.embeddings.models import (
    EmbeddingArtifact,
    EmbeddingId,
    EmbeddingManifest,
    EmbeddingTarget,
)
from backend.embeddings.planner.embedding_planner import EmbeddingPlanner, EmbeddingTask
from backend.embeddings.repository.embedding_repository import EmbeddingRepository
from backend.embeddings.validator.embedding_validator import validate_embeddings

logger = logging.getLogger(__name__)

ARTIFACT_SCHEMA_VERSION = "1.0"
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 0.1


@dataclass(frozen=True)
class EmbeddingResult:
    """The outcome of one `embed_document` call.

    Attributes:
        manifest: The manifest describing this run (freshly generated, or
            the existing one if generation was skipped as not stale).
        artifacts: Newly generated artifacts. Empty if generation was
            skipped because existing embeddings were already fresh.
    """

    manifest: EmbeddingManifest
    artifacts: list[EmbeddingArtifact]


class EmbeddingService:
    """Generates and persists embeddings for a document's knowledge representation."""

    def __init__(
        self,
        repository: EmbeddingRepository,
        text_provider: EmbeddingProvider,
        planner: EmbeddingPlanner | None = None,
        image_provider: ImageEmbeddingProvider | None = None,
        batch_size: int = 32,
    ) -> None:
        """Initialize the service.

        Args:
            repository: Reads representations and persists artifacts.
            text_provider: Produces `TEXT` embeddings. Required.
            planner: Decides which targets each chunk needs. Defaults to a
                new `EmbeddingPlanner`.
            image_provider: Produces `IMAGE` embeddings. Optional -- when
                `None`, chunks needing an image embedding are skipped and
                counted, not attempted or faked.
            batch_size: Number of chunks embedded per provider call. A
                failing batch is retried, then marked failed as a whole if
                retries are exhausted -- this bounds failure blast radius
                without per-item call overhead in the common case.
        """
        self._repository = repository
        self._text_provider = text_provider
        self._image_provider = image_provider
        self._planner = planner or EmbeddingPlanner()
        self._batch_size = batch_size

    def embed_document(self, document_id: PaperId, force: bool = False) -> EmbeddingResult:
        """Generate and persist embeddings for a document's knowledge representation.

        Idempotent by default: if existing embeddings are already fresh
        (same model, same representation content), no work is done and the
        existing manifest is returned unchanged.

        Args:
            document_id: Identifier of a document with an existing
                knowledge representation (Module 5's output).
            force: Regenerate even if existing embeddings are already fresh.

        Returns:
            The manifest and any newly generated artifacts.

        Raises:
            RepresentationNotFoundError: No knowledge representation exists
                for this document.
            NoEmbeddingsProducedError: Every knowledge unit failed to embed.
            EmbeddingValidationError: The generated artifacts are structurally invalid.
            EmbeddingStorageError: A storage failure prevented persistence.
        """
        if not force and not self._repository.is_stale(
            document_id, self._text_provider.model_name, self._text_provider.model_version
        ):
            manifest = self._repository.load_manifest(document_id)
            if manifest is not None:
                logger.info(
                    "embeddings already up to date, skipping regeneration",
                    extra={"document_id": str(document_id)},
                )
                return EmbeddingResult(manifest=manifest, artifacts=[])

        payload = self._repository.read_knowledge_units_payload(document_id)
        chunks = [Chunk.model_validate(item) for item in payload["chunks"]]
        representation_version = self._repository.compute_representation_version(document_id)

        tasks = self._planner.plan(chunks)
        artifacts, failed_count, skipped_image_count = self._embed_tasks(
            tasks, representation_version
        )

        if not artifacts:
            raise NoEmbeddingsProducedError(document_id=document_id)

        validate_embeddings(document_id, chunks, artifacts)

        manifest = EmbeddingManifest(
            document_id=document_id,
            model_name=self._text_provider.model_name,
            model_version=self._text_provider.model_version,
            embedding_dimension=self._text_provider.embedding_dimension,
            artifact_version=ARTIFACT_SCHEMA_VERSION,
            source_representation_version=representation_version,
            embedding_count=len(artifacts),
            failed_count=failed_count,
            skipped_image_count=skipped_image_count,
            created_at=datetime.now(UTC),
        )
        self._repository.save(document_id, artifacts, manifest)

        logger.info(
            "document embedded",
            extra={
                "document_id": str(document_id),
                "embeddings": len(artifacts),
                "failed": failed_count,
                "skipped_image": skipped_image_count,
            },
        )
        return EmbeddingResult(manifest=manifest, artifacts=artifacts)

    def _embed_tasks(
        self, tasks: list[EmbeddingTask], representation_version: str
    ) -> tuple[list[EmbeddingArtifact], int, int]:
        text_tasks = [task for task in tasks if task.target is EmbeddingTarget.TEXT]
        image_tasks = [task for task in tasks if task.target is EmbeddingTarget.IMAGE]

        if image_tasks and self._image_provider is None:
            logger.info(
                "skipping image embeddings: no image embedding provider configured",
                extra={"skipped_count": len(image_tasks)},
            )

        artifacts: list[EmbeddingArtifact] = []
        failed_count = 0

        for batch_start in range(0, len(text_tasks), self._batch_size):
            batch = text_tasks[batch_start : batch_start + self._batch_size]
            batch_artifacts, batch_failed = self._embed_text_batch(batch, representation_version)
            artifacts.extend(batch_artifacts)
            failed_count += batch_failed

        return artifacts, failed_count, len(image_tasks)

    def _embed_text_batch(
        self, batch: list[EmbeddingTask], representation_version: str
    ) -> tuple[list[EmbeddingArtifact], int]:
        texts = [task.chunk.text for task in batch]
        try:
            vectors = _call_with_retry(lambda: self._text_provider.embed_texts(texts))
        except EmbeddingProviderError:
            logger.error(
                "batch embedding failed after retries; marking batch as failed",
                exc_info=True,
                extra={"batch_size": len(batch)},
            )
            return [], len(batch)

        now = datetime.now(UTC)
        artifacts = [
            EmbeddingArtifact(
                embedding_id=EmbeddingId(uuid4()),
                knowledge_unit_id=task.chunk.id,
                paper_id=task.chunk.paper_id,
                target=task.target,
                vector=vector,
                model_name=self._text_provider.model_name,
                model_version=self._text_provider.model_version,
                embedding_dimension=self._text_provider.embedding_dimension,
                checksum=hashlib.sha256(task.chunk.text.encode("utf-8")).hexdigest(),
                artifact_version=ARTIFACT_SCHEMA_VERSION,
                source_representation_version=representation_version,
                created_at=now,
            )
            for task, vector in zip(batch, vectors, strict=True)
        ]
        return artifacts, 0


def _call_with_retry(func: Callable[[], list[list[float]]]) -> list[list[float]]:
    """Call a provider function with bounded retry and linear backoff.

    Provider-agnostic by design: today's concrete provider (in-process
    SentenceTransformers) has no real transient-failure mode to retry, but
    a future served backend (Infinity, HuggingFace TEI) genuinely does --
    this wraps every provider call regardless of which concrete
    implementation is wired in.
    """
    last_error: EmbeddingProviderError | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return func()
        except EmbeddingProviderError as exc:
            last_error = exc
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
    assert last_error is not None
    raise last_error
