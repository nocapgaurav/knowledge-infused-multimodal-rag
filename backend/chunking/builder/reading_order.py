"""Reconstructs true document reading order from a Paper's flat collections.

`Paragraph.order` is scoped to siblings within the same section;
`Figure.order`/`Table.order` are scoped globally across the whole paper.
These are independent counters -- comparing them directly tells you
nothing about how a figure interleaves with the paragraphs around it
inside a shared section. This module recovers the true visual order using
each entity's bounding box position, which the parser already preserved.
"""

from collections import defaultdict
from dataclasses import dataclass

from backend.domain import Figure, Paper, Paragraph, Section, SectionId, Table

type PositionedEntity = Paragraph | Table | Figure


@dataclass(frozen=True)
class OrderedItem:
    """One entity, placed at its resolved position in the reading-order stream.

    Attributes:
        section_id: Section this entity belongs to (`None` for entities
            with no detected section, e.g. text before the first heading).
        item: The paragraph, table, or figure at this position.
    """

    section_id: SectionId | None
    item: PositionedEntity


def compute_reading_order(paper: Paper) -> list[OrderedItem]:
    """Compute the true reading-order sequence of a paper's positioned content.

    Sections are visited depth-first (matching table-of-contents order).
    Within each section, paragraphs/tables/figures belonging to it are
    sorted by page number and vertical position when available, falling
    back to a paragraphs-then-figures-then-tables convention (each ordered
    by its own per-type `order`) when position data is missing.

    Abstract and reference content are not included here -- they are not
    part of the section-based reading stream (the parser already extracted
    the abstract into `Paper.metadata`, and references form their own
    trailing list), so the builder handles them separately.

    Args:
        paper: The paper to compute reading order for.

    Returns:
        Every paragraph, table, and figure in the paper, in reading order.
    """
    section_order = _depth_first_section_order(paper)

    buckets: dict[SectionId | None, list[PositionedEntity]] = defaultdict(list)
    for paragraph in paper.paragraphs:
        buckets[paragraph.section_id].append(paragraph)
    for table in paper.tables:
        buckets[table.section_id].append(table)
    for figure in paper.figures:
        buckets[figure.section_id].append(figure)

    ordered: list[OrderedItem] = []
    for section_id in (None, *section_order):
        for item in sorted(buckets.get(section_id, ()), key=_sort_key):
            ordered.append(OrderedItem(section_id=section_id, item=item))
    return ordered


def _depth_first_section_order(paper: Paper) -> list[SectionId]:
    """Return every section id in depth-first, sibling-ordered traversal order."""
    children_by_parent: dict[SectionId | None, list[Section]] = defaultdict(list)
    for section in paper.sections:
        children_by_parent[section.parent_section_id].append(section)
    for children in children_by_parent.values():
        children.sort(key=lambda section: section.order)

    ordered_ids: list[SectionId] = []

    def visit(parent_id: SectionId | None) -> None:
        for child in children_by_parent.get(parent_id, ()):
            ordered_ids.append(child.id)
            visit(child.id)

    visit(None)
    return ordered_ids


def _sort_key(item: PositionedEntity) -> tuple[int, int, float, int, int]:
    type_rank = _type_rank(item)
    if item.bounding_boxes:
        box = item.bounding_boxes[0]
        return (0, box.page_number, box.y0, type_rank, item.order)
    return (1, 0, 0.0, type_rank, item.order)


def _type_rank(item: PositionedEntity) -> int:
    if isinstance(item, Paragraph):
        return 0
    if isinstance(item, Figure):
        return 1
    return 2
