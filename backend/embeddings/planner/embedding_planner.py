"""Decides which embedding targets each knowledge unit needs."""

from collections.abc import Sequence
from dataclasses import dataclass

from backend.domain import Chunk
from backend.embeddings.models import EmbeddingTarget


@dataclass(frozen=True)
class EmbeddingTask:
    """One unit of embedding work: a chunk plus the target it needs.

    Attributes:
        chunk: The knowledge unit to embed.
        target: Which kind of embedding this task produces.
    """

    chunk: Chunk
    target: EmbeddingTarget


class EmbeddingPlanner:
    """Determines which embedding targets each knowledge unit needs.

    A content-driven rule, not a hardcoded per-modality one: every chunk
    needs a `TEXT` embedding (`Chunk.text` is guaranteed non-empty by the
    domain model); any chunk with a non-null `asset_uri` additionally needs
    an `IMAGE` embedding -- regardless of its own `modality`, so this
    generalizes cleanly if `Table.asset_uri` is ever populated.
    """

    def plan(self, chunks: Sequence[Chunk]) -> list[EmbeddingTask]:
        """Build the list of embedding tasks for a set of knowledge units.

        Args:
            chunks: Knowledge units to plan embeddings for.

        Returns:
            One `TEXT` task per chunk, plus one `IMAGE` task for every
            chunk that has an `asset_uri`.
        """
        tasks: list[EmbeddingTask] = []
        for chunk in chunks:
            tasks.append(EmbeddingTask(chunk=chunk, target=EmbeddingTarget.TEXT))
            if chunk.asset_uri is not None:
                tasks.append(EmbeddingTask(chunk=chunk, target=EmbeddingTarget.IMAGE))
        return tasks
