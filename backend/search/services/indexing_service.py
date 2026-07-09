"""Orchestrates indexing.

Check staleness (skip if unchanged, unless forced) -> plan collections ->
build payloads -> ensure the collection exists -> upsert in batches, with
retry and batch-level partial success -> verify -> persist the index
manifest. Each step (repository, planner, payload builder, provider,
validator) is independently testable; this class's only job is calling
them in the right order.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial

from backend.domain import PaperId
from backend.search.exceptions import (
    MultiCollectionIndexingNotSupportedError,
    NoVectorsIndexedError,
    VectorStoreError,
)
from backend.search.interfaces.vector_store import VectorStore
from backend.search.models import IndexedField, IndexManifest, PayloadFieldType, VectorPoint
from backend.search.payload.payload_builder import PayloadBuilder, build_citation_counts
from backend.search.planner.index_planner import IndexPlanner
from backend.search.repository.index_repository import IndexRepository
from backend.search.validator.index_validator import IndexValidator

logger = logging.getLogger(__name__)

ARTIFACT_SCHEMA_VERSION = "1.0"
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 0.1

_STANDARD_INDEXED_FIELDS = (
    IndexedField(name="document_id", field_type=PayloadFieldType.KEYWORD),
    IndexedField(name="section_id", field_type=PayloadFieldType.KEYWORD),
    IndexedField(name="modality", field_type=PayloadFieldType.KEYWORD),
    IndexedField(name="embedding_target", field_type=PayloadFieldType.KEYWORD),
    IndexedField(name="embedding_version", field_type=PayloadFieldType.KEYWORD),
)


@dataclass(frozen=True)
class IndexingResult:
    """The outcome of one `index_document` call.

    Attributes:
        manifest: The manifest describing this run (freshly generated, or
            the existing one if indexing was skipped as not stale).
        newly_indexed: Number of vectors newly indexed by this call. `0` if
            indexing was skipped because the existing index was already fresh.
    """

    manifest: IndexManifest
    newly_indexed: int


class IndexingService:
    """Indexes a document's embedding artifacts into a vector store."""

    def __init__(
        self,
        repository: IndexRepository,
        vector_store: VectorStore,
        collection_prefix: str,
        batch_size: int = 64,
    ) -> None:
        """Initialize the service.

        Args:
            repository: Reads embedding/knowledge artifacts and persists
                index manifests.
            vector_store: The vector store to index into.
            collection_prefix: Namespace prefix for collection names.
            batch_size: Number of points upserted per vector store call. A
                failing batch is retried, then marked failed as a whole if
                retries are exhausted.
        """
        self._repository = repository
        self._vector_store = vector_store
        self._planner = IndexPlanner(collection_prefix=collection_prefix)
        self._payload_builder = PayloadBuilder()
        self._validator = IndexValidator(vector_store)
        self._batch_size = batch_size

    def index_document(self, document_id: PaperId, force: bool = False) -> IndexingResult:
        """Index a document's embedding artifacts, verifying the result.

        Idempotent by default: if the existing index is already fresh
        (embedding manifest unchanged), no work is done and the existing
        manifest is returned unchanged.

        Args:
            document_id: Identifier of a document with existing embedding
                artifacts (Module 6's output).
            force: Reindex even if the existing index is already fresh.

        Returns:
            The manifest and the number of vectors newly indexed.

        Raises:
            EmbeddingArtifactsNotFoundError: No embedding artifacts exist
                for this document.
            MultiCollectionIndexingNotSupportedError: The document's
                embeddings span more than one collection.
            NoVectorsIndexedError: Every vector failed to index.
            IndexValidationError: Post-indexing verification failed.
            IndexStorageError: A storage failure prevented persistence.
        """
        if not force and not self._repository.is_stale(document_id):
            manifest = self._repository.load_index_manifest(document_id)
            if manifest is not None:
                logger.info(
                    "index already up to date, skipping reindexing",
                    extra={"document_id": str(document_id)},
                )
                return IndexingResult(manifest=manifest, newly_indexed=0)

        artifacts = self._repository.read_embedding_artifacts(document_id)
        embedding_manifest = self._repository.read_embedding_manifest(document_id)
        chunks = self._repository.read_chunks(document_id)
        citation_counts = build_citation_counts(self._repository.read_relationships(document_id))

        plans = self._planner.plan(artifacts)
        if len(plans) > 1:
            raise MultiCollectionIndexingNotSupportedError(
                document_id=document_id,
                collection_names=[plan.collection_name for plan in plans],
            )
        if not plans:
            raise NoVectorsIndexedError(document_id=document_id)
        plan = plans[0]

        _call_with_retry(
            lambda: self._vector_store.ensure_collection(
                plan.collection_name, plan.dimension, plan.distance, _STANDARD_INDEXED_FIELDS
            )
        )

        points = [
            self._payload_builder.build_point(
                artifact,
                chunks[artifact.knowledge_unit_id],
                citation_counts.get(artifact.knowledge_unit_id, 0),
                embedding_manifest,
            )
            for artifact in plan.artifacts
        ]

        indexed_points, failed_count = self._upsert_in_batches(plan.collection_name, points)
        if not indexed_points:
            raise NoVectorsIndexedError(document_id=document_id)

        self._validator.verify(
            document_id=document_id,
            collection=plan.collection_name,
            expected_dimension=plan.dimension,
            expected_count=len(indexed_points),
            indexed_points=indexed_points,
        )

        manifest = IndexManifest(
            document_id=document_id,
            collection_name=plan.collection_name,
            vector_dimension=plan.dimension,
            distance_metric=plan.distance,
            embedding_model=embedding_manifest.model_name,
            embedding_version=embedding_manifest.model_version,
            artifact_version=ARTIFACT_SCHEMA_VERSION,
            source_embedding_manifest=self._repository.compute_embedding_manifest_hash(document_id),
            checksum=self._repository.compute_embeddings_checksum(document_id),
            indexed_vectors=len(indexed_points),
            failed_vectors=failed_count,
            created_at=datetime.now(UTC),
        )
        self._repository.save_index_manifest(document_id, manifest)

        logger.info(
            "document indexed",
            extra={
                "document_id": str(document_id),
                "collection": plan.collection_name,
                "indexed": len(indexed_points),
                "failed": failed_count,
            },
        )
        return IndexingResult(manifest=manifest, newly_indexed=len(indexed_points))

    def _upsert_in_batches(
        self, collection: str, points: list[VectorPoint]
    ) -> tuple[list[VectorPoint], int]:
        indexed: list[VectorPoint] = []
        failed_count = 0
        for batch_start in range(0, len(points), self._batch_size):
            batch = points[batch_start : batch_start + self._batch_size]
            try:
                _call_with_retry(partial(self._vector_store.upsert, collection, batch))
                indexed.extend(batch)
            except VectorStoreError:
                logger.error(
                    "batch upsert failed after retries; marking batch as failed",
                    exc_info=True,
                    extra={"batch_size": len(batch)},
                )
                failed_count += len(batch)
        return indexed, failed_count


def _call_with_retry(func: Callable[[], None]) -> None:
    """Call a vector store operation with bounded retry and linear backoff.

    Provider-agnostic by design: today's concrete provider (Qdrant over a
    local Docker network) rarely hits a real transient failure, but a
    remote or heavily-loaded deployment genuinely can -- this wraps every
    vector store call regardless of which concrete implementation is wired in.
    """
    last_error: VectorStoreError | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            func()
            return
        except VectorStoreError as exc:
            last_error = exc
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
    assert last_error is not None
    raise last_error
