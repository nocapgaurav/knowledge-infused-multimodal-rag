"""Structural validation for a freshly mapped `Paper`.

Some validation is already handled by construction: `Paper` itself enforces
that every child belongs to it, and `Metadata.title` cannot be blank
(pydantic `min_length=1`) -- the mapper raises its own
`MissingRequiredMetadataError` before construction to give that specific
failure a clearer message than a generic validation error would. What
remains here is exactly what field-level constraints cannot express:
whether the section tree is actually a tree, and whether captions point at
real subjects.
"""

import logging

from backend.domain import CaptionSubjectType, Paper, SectionId
from backend.parser.exceptions import (
    EmptyDocumentError,
    InvalidSectionHierarchyError,
    MissingFigureReferenceError,
)

logger = logging.getLogger(__name__)


def validate_document(paper: Paper) -> None:
    """Validate a freshly parsed document before it is persisted.

    Args:
        paper: The mapped `Paper` to validate.

    Raises:
        EmptyDocumentError: The document has no sections or paragraphs.
        InvalidSectionHierarchyError: A section references a parent that
            does not exist, or the hierarchy contains a cycle.
        MissingFigureReferenceError: A caption references a figure or
            table that does not exist.
    """
    _validate_not_empty(paper)
    _validate_section_hierarchy(paper)
    _validate_caption_subjects(paper)
    _warn_on_uncaptioned_visuals(paper)


def _validate_not_empty(paper: Paper) -> None:
    has_content = bool(paper.sections or paper.paragraphs or paper.figures or paper.tables)
    if not has_content:
        raise EmptyDocumentError(document_id=paper.id)


def _validate_section_hierarchy(paper: Paper) -> None:
    section_ids = {section.id for section in paper.sections}
    for section in paper.sections:
        if section.parent_section_id is not None and section.parent_section_id not in section_ids:
            raise InvalidSectionHierarchyError(
                document_id=paper.id,
                reason=(
                    f"section {section.id} references unknown parent "
                    f"{section.parent_section_id}"
                ),
            )

    parent_by_id = {section.id: section.parent_section_id for section in paper.sections}
    for section in paper.sections:
        visited: set[SectionId] = set()
        current: SectionId | None = section.id
        while current is not None:
            if current in visited:
                raise InvalidSectionHierarchyError(
                    document_id=paper.id,
                    reason=f"cycle detected in section hierarchy starting at {section.id}",
                )
            visited.add(current)
            current = parent_by_id.get(current)


def _validate_caption_subjects(paper: Paper) -> None:
    figure_ids = {figure.id for figure in paper.figures}
    table_ids = {table.id for table in paper.tables}
    for caption in paper.captions:
        valid_ids = figure_ids if caption.subject_type is CaptionSubjectType.FIGURE else table_ids
        if caption.subject_id not in valid_ids:
            raise MissingFigureReferenceError(document_id=paper.id, caption_id=caption.id)


def _warn_on_uncaptioned_visuals(paper: Paper) -> None:
    captioned_subject_ids = {caption.subject_id for caption in paper.captions}
    for figure in paper.figures:
        if figure.id not in captioned_subject_ids:
            logger.warning(
                "figure has no caption",
                extra={"document_id": str(paper.id), "figure_id": str(figure.id)},
            )
    for table in paper.tables:
        if table.id not in captioned_subject_ids:
            logger.warning(
                "table has no caption",
                extra={"document_id": str(paper.id), "table_id": str(table.id)},
            )
