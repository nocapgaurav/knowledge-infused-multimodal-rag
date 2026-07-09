"""Integration tests for the real SentenceTransformers/BGE-M3 provider.

Unlike the other tests in this suite (which use a fake provider), this
exercises the real model -- the only test that verifies the actual
dimension, revision-pinning, and encoding behavior hold up against the real
library, not an assumption about its shape. Slower than the rest of the
suite (loads real model weights); the provider fixture is module-scoped so
that cost is paid once.
"""

import pytest

from backend.embeddings.exceptions import EmbeddingProviderError
from backend.embeddings.providers.sentence_transformers_provider import (
    SentenceTransformersProvider,
)


@pytest.fixture(scope="module")
def provider() -> SentenceTransformersProvider:
    return SentenceTransformersProvider(model_name="BAAI/bge-m3")


def test_reports_expected_dimension(provider: SentenceTransformersProvider) -> None:
    assert provider.embedding_dimension == 1024


def test_model_version_is_a_resolved_concrete_revision(
    provider: SentenceTransformersProvider,
) -> None:
    # A real HuggingFace commit SHA, not a floating label like "latest".
    assert provider.model_version
    assert provider.model_version != "latest"
    assert len(provider.model_version) >= 7


def test_embed_texts_produces_correctly_shaped_vectors(
    provider: SentenceTransformersProvider,
) -> None:
    vectors = provider.embed_texts(
        ["A paper about knowledge-infused retrieval.", "A short caption."]
    )

    assert len(vectors) == 2
    assert all(len(vector) == provider.embedding_dimension for vector in vectors)


def test_embed_texts_is_deterministic(provider: SentenceTransformersProvider) -> None:
    text = "This finding was established in prior work."

    first = provider.embed_texts([text])
    second = provider.embed_texts([text])

    assert first == second


def test_empty_input_returns_empty_output(provider: SentenceTransformersProvider) -> None:
    assert provider.embed_texts([]) == []


def test_pinning_an_explicit_revision_matches_the_resolved_one(
    provider: SentenceTransformersProvider,
) -> None:
    pinned = SentenceTransformersProvider(model_name="BAAI/bge-m3", revision=provider.model_version)

    assert pinned.model_version == provider.model_version
    assert pinned.embedding_dimension == provider.embedding_dimension


def test_loading_an_unknown_model_raises_embedding_provider_error() -> None:
    with pytest.raises(EmbeddingProviderError):
        SentenceTransformersProvider(model_name="this-model-does-not-exist/at-all")
