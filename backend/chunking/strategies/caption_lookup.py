"""Shared caption lookup used by the figure and table strategies."""

from uuid import UUID

from backend.domain import Caption, CaptionSubjectType, Paper


def find_caption(
    paper: Paper, subject_id: UUID, subject_type: CaptionSubjectType
) -> Caption | None:
    """Find the caption describing a given figure or table, if one exists.

    Args:
        paper: The paper to search.
        subject_id: Identifier of the figure or table.
        subject_type: Whether `subject_id` refers to a figure or a table.

    Returns:
        The matching caption, or `None` if the figure/table has no caption.
    """
    for caption in paper.captions:
        if caption.subject_type is subject_type and caption.subject_id == subject_id:
            return caption
    return None
