"""Builds a knowledge unit from a table, fusing its caption and markdown."""

from backend.chunking.interfaces.context import BuildContext, StrategyResult
from backend.chunking.strategies.caption_lookup import find_caption
from backend.domain import CaptionSubjectType, Chunk, ChunkModality, Table

_UNTITLED_TABLE_TEXT = "Untitled table"


class TableStrategy:
    """Builds one knowledge unit per table, with its caption and markdown fused in.

    Unlike a figure, a table has substantive content of its own
    (`Table.markdown`) beyond its caption -- both are combined when
    present, since the caption carries the topic/label and the markdown
    carries the actual queryable data. A minor redundancy is possible if
    the parser's markdown export already embeds the caption text; that's a
    low-cost tradeoff against under-serving retrieval by using the caption
    alone.
    """

    def build(self, table: Table, context: BuildContext) -> StrategyResult:
        """Build a knowledge unit for a single table.

        Args:
            table: The table to build a unit for.
            context: Shared build context.

        Returns:
            A single-chunk result. No relationships are detected from a
            table's own caption/markdown text.
        """
        caption = find_caption(context.paper, table.id, CaptionSubjectType.TABLE)

        text_parts = []
        if caption is not None:
            text_parts.append(caption.text)
        if table.markdown:
            text_parts.append(table.markdown)
        text = "\n\n".join(text_parts) if text_parts else _UNTITLED_TABLE_TEXT

        source_element_ids = [table.id, *([caption.id] if caption is not None else [])]

        chunk = Chunk(
            paper_id=context.paper.id,
            section_id=table.section_id,
            order=context.next_order(),
            modality=ChunkModality.TABLE,
            text=text,
            asset_uri=table.asset_uri,
            source_element_ids=source_element_ids,
            bounding_boxes=table.bounding_boxes,
        )
        context.register_chunk(table.id, chunk.id)
        return StrategyResult(chunks=[chunk], relationships=[])
