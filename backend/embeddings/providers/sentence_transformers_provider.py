"""SentenceTransformers-based implementation of `EmbeddingProvider`.

This is the only file in the application permitted to import
`sentence_transformers` or `huggingface_hub`.
"""

import logging
from collections.abc import Sequence

from huggingface_hub import model_info
from sentence_transformers import SentenceTransformer

from backend.embeddings.exceptions import EmbeddingProviderError
from backend.embeddings.interfaces.embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformersProvider(EmbeddingProvider):
    """Text embedding provider backed by a local `sentence-transformers` model.

    Resolves and pins an explicit model revision at load time, so
    `model_version` is a concrete, reproducible identifier (a HuggingFace
    commit SHA) rather than a floating "latest" label that could silently
    change out from under previously generated embeddings.
    """

    def __init__(self, model_name: str, revision: str | None = None, batch_size: int = 32) -> None:
        """Load the model, pinned to an explicit revision.

        Args:
            model_name: HuggingFace model identifier (e.g. "BAAI/bge-m3").
            revision: Explicit commit revision to pin to. If `None`, the
                current revision is resolved once via the HuggingFace Hub
                API and pinned for the lifetime of this instance.
            batch_size: Number of texts encoded per internal batch.

        Raises:
            EmbeddingProviderError: The model could not be loaded.
        """
        try:
            resolved_revision = revision or model_info(model_name).sha
            if resolved_revision is None:
                raise EmbeddingProviderError(
                    reason=f"could not resolve a revision for model '{model_name}'"
                )
            self._model = SentenceTransformer(model_name, revision=resolved_revision)
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            raise EmbeddingProviderError(
                reason=f"failed to load model '{model_name}': {exc}"
            ) from exc

        self._model_name = model_name
        self._model_version = resolved_revision
        self._batch_size = batch_size
        # get_embedding_dimension() is the current name; fall back for older
        # sentence-transformers releases still within our pinned version range.
        if hasattr(self._model, "get_embedding_dimension"):
            self._embedding_dimension = self._model.get_embedding_dimension()
        else:
            self._embedding_dimension = self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        max_seq_length = self._model.max_seq_length
        for text in texts:
            approx_tokens = len(text.split())
            if approx_tokens > max_seq_length:
                logger.warning(
                    "text exceeds the model's max sequence length and will be "
                    "silently truncated by the model",
                    extra={"approx_word_count": approx_tokens, "max_seq_length": max_seq_length},
                )

        try:
            vectors = self._model.encode(
                list(texts),
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingProviderError(reason=f"embedding call failed: {exc}") from exc

        return [vector.tolist() for vector in vectors]
