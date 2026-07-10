"""Unit tests for evaluation dataset validation."""

from pathlib import Path

import pytest

from backend.evaluation.datasets.evaluation_dataset import EvaluationDataset, load_dataset_from_json
from backend.evaluation.exceptions import DuplicateCaseIdError, EmptyDatasetError
from backend.evaluation.validation.evaluation_validator import EvaluationValidator

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_validate_dataset_accepts_a_valid_dataset() -> None:
    dataset = load_dataset_from_json(FIXTURES_DIR / "valid_dataset.json")

    EvaluationValidator().validate_dataset(dataset, path="valid_dataset.json")


def test_validate_dataset_rejects_an_empty_dataset() -> None:
    empty = EvaluationDataset(cases=(), dataset_version="empty-hash")

    with pytest.raises(EmptyDatasetError):
        EvaluationValidator().validate_dataset(empty, path="empty.json")


def test_validate_dataset_rejects_duplicate_case_ids() -> None:
    dataset = load_dataset_from_json(FIXTURES_DIR / "duplicate_case_id_dataset.json")

    with pytest.raises(DuplicateCaseIdError):
        EvaluationValidator().validate_dataset(dataset, path="duplicate_case_id_dataset.json")
