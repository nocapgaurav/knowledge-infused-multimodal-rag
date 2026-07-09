"""Provider ports: what any embedding backend must provide.

Business logic (the planner, the service) depends only on these two
interfaces, never on a specific model or library. Swapping the concrete
provider -- SentenceTransformers today, Infinity, HuggingFace TEI, or vLLM
tomorrow -- changes zero lines outside `providers/`.

Two interfaces, not one: a text embedding model and an image-capable model
have different inputs, different vector spaces, and are typically served by
different backends. Forcing them into one interface would be exactly the
kind of hollow abstraction that comes from unifying genuinely incompatible
shapes for the sake of having a single interface.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingProvider(ABC):
    """Turns text into dense embedding vectors."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the underlying model (e.g. "BAAI/bge-m3")."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Resolved, concrete revision of the model -- never a floating "latest" label."""

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Length of every vector this provider produces."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: Texts to embed, in order.

        Returns:
            One vector per input text, in the same order.

        Raises:
            EmbeddingProviderError: The provider failed to produce embeddings.
        """


class ImageEmbeddingProvider(ABC):
    """Turns image bytes into dense embedding vectors.

    Not implemented by any concrete provider in this module (see
    `providers/` docstring) -- defined now because the planner and artifact
    schema need it to model figures correctly, even before a concrete
    implementation exists.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the underlying model (e.g. "clip-ViT-B-32")."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Resolved, concrete revision of the model."""

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Length of every vector this provider produces."""

    @abstractmethod
    def embed_images(self, images: Sequence[bytes]) -> list[list[float]]:
        """Embed a batch of images.

        Args:
            images: Raw image bytes to embed, in order.

        Returns:
            One vector per input image, in the same order.

        Raises:
            EmbeddingProviderError: The provider failed to produce embeddings.
        """
