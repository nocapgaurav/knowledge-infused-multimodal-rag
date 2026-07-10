"""Loads evaluation datasets from JSON, computing a deterministic content
version for every load.

Structural validation (empty dataset, duplicate case ids) deliberately
does not happen here -- that is `validation/evaluation_validator.py`'s
job, matching how every other module keeps "parse this artifact" and
"verify it's structurally sound" as separate, independently testable steps.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.evaluation.exceptions import DatasetNotFoundError
from backend.evaluation.models.evaluation_case import EvaluationCase


@dataclass(frozen=True)
class EvaluationDataset:
    """An in-memory evaluation dataset, loaded from JSON.

    Attributes:
        cases: Every case in the dataset, in file order.
        dataset_version: SHA-256 hex digest of the dataset's canonical
            JSON content -- two benchmark runs are only comparable if
            this matches.
    """

    cases: tuple[EvaluationCase, ...]
    dataset_version: str


def load_dataset_from_json(path: Path) -> EvaluationDataset:
    """Load an evaluation dataset from a JSON file.

    Args:
        path: Path to a JSON file containing a list of case objects.

    Returns:
        The loaded dataset, with a deterministic version hash.

    Raises:
        DatasetNotFoundError: The file does not exist.
    """
    if not path.is_file():
        raise DatasetNotFoundError(path=str(path))

    payload: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    cases = tuple(EvaluationCase.model_validate(item) for item in payload)
    dataset_version = _hash_payload(payload)
    return EvaluationDataset(cases=cases, dataset_version=dataset_version)


def _hash_payload(payload: list[dict[str, Any]]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
