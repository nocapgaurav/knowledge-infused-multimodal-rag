"""Tests for deterministic collection naming."""

from backend.search.collection_naming import build_collection_name


def test_same_inputs_produce_the_same_name() -> None:
    first = build_collection_name(
        "kimrag", "BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", "text"
    )
    second = build_collection_name(
        "kimrag", "BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", "text"
    )

    assert first == second


def test_different_model_versions_produce_different_names() -> None:
    first = build_collection_name("kimrag", "BAAI/bge-m3", "aaaaaaaaaaaa", "text")
    second = build_collection_name("kimrag", "BAAI/bge-m3", "bbbbbbbbbbbb", "text")

    assert first != second


def test_different_targets_produce_different_names() -> None:
    text_name = build_collection_name("kimrag", "BAAI/bge-m3", "5617a9f6", "text")
    image_name = build_collection_name("kimrag", "BAAI/bge-m3", "5617a9f6", "image")

    assert text_name != image_name


def test_name_is_sanitized_and_lowercase() -> None:
    name = build_collection_name("kimrag", "BAAI/bge-m3", "5617A9F6", "text")

    assert name == "kimrag_baai_bge_m3_5617a9f6_text"
    assert "/" not in name
