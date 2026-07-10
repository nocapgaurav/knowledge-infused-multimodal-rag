"""Fail-loud validation of an evaluation dataset.

The dataset is the one place genuinely external, untrusted input enters
this module -- every other artifact (benchmark results, summaries) is
already validated at construction time by its own Pydantic model, so
there is nothing further to check about them here.
"""

import logging

from backend.evaluation.datasets.evaluation_dataset import EvaluationDataset
from backend.evaluation.exceptions import DuplicateCaseIdError, EmptyDatasetError

logger = logging.getLogger(__name__)


class EvaluationValidator:
    """Validates a loaded evaluation dataset before it is run."""

    def validate_dataset(self, dataset: EvaluationDataset, path: str) -> None:
        """Validate a loaded evaluation dataset.

        Args:
            dataset: The loaded dataset to validate.
            path: Path the dataset was loaded from, for error messages.

        Raises:
            EmptyDatasetError: The dataset contains no cases.
            DuplicateCaseIdError: The same case_id appears more than once.
        """
        if not dataset.cases:
            raise EmptyDatasetError(path=path)

        seen_case_ids: set[str] = set()
        for case in dataset.cases:
            if case.case_id in seen_case_ids:
                raise DuplicateCaseIdError(case_id=case.case_id)
            seen_case_ids.add(case.case_id)

        logger.info(
            "evaluation dataset validated",
            extra={"path": path, "case_count": len(dataset.cases)},
        )
