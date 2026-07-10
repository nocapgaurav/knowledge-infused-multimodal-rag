"""Unit tests for loading an evaluation dataset from JSON."""

from pathlib import Path
from uuid import UUID

import pytest

from backend.evaluation.datasets.evaluation_dataset import load_dataset_from_json
from backend.evaluation.exceptions import DatasetNotFoundError
from backend.evaluation.models.evaluation_case import Difficulty

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_dataset_from_json_parses_every_case() -> None:
    dataset = load_dataset_from_json(FIXTURES_DIR / "valid_dataset.json")

    assert len(dataset.cases) == 2
    first = dataset.cases[0]
    assert first.case_id == "case-001"
    assert first.document_id == UUID("12f9e408-f3de-414d-80da-5bbbedd1e2a6")
    assert first.expected_knowledge_units == ("ku-1", "ku-2")
    assert first.difficulty == Difficulty.EASY


def test_load_dataset_from_json_computes_a_stable_version_hash() -> None:
    first = load_dataset_from_json(FIXTURES_DIR / "valid_dataset.json")
    second = load_dataset_from_json(FIXTURES_DIR / "valid_dataset.json")

    assert first.dataset_version == second.dataset_version
    assert len(first.dataset_version) == 64


def test_load_dataset_from_json_version_differs_for_different_content() -> None:
    valid = load_dataset_from_json(FIXTURES_DIR / "valid_dataset.json")
    duplicate = load_dataset_from_json(FIXTURES_DIR / "duplicate_case_id_dataset.json")

    assert valid.dataset_version != duplicate.dataset_version


def test_load_dataset_from_json_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError):
        load_dataset_from_json(tmp_path / "does-not-exist.json")
