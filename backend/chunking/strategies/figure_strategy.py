"""Builds a knowledge unit from a figure, fusing its caption text."""

from backend.chunking.interfaces.context import BuildContext, StrategyResult
from backend.chunking.strategies.caption_lookup import find_caption
from backend.domain import CaptionSubjectType, Chunk, ChunkModality, Figure

_UNCAPTIONED_FIGURE_TEXT = "Untitled figure"


class FigureStrategy:
    """Builds one knowledge unit per figure, with its caption fused in.

    A figure's caption is its textual identity for retrieval purposes --
    without it, an image has no text to embed against a text-first index.
    Fusing them here (rather than treating `Caption` as its own retrievable
    unit) reflects that a caption has no independent meaning without the
    figure it describes.
    """

    def build(self, figure: Figure, context: BuildContext) -> StrategyResult:
        """Build a knowledge unit for a single figure.

        Args:
            figure: The figure to build a unit for.
            context: Shared build context.

        Returns:
            A single-chunk result. No relationships are detected from a
            figure's own caption text.
        """
        caption = find_caption(context.paper, figure.id, CaptionSubjectType.FIGURE)
        text = caption.text if caption is not None else _UNCAPTIONED_FIGURE_TEXT
        source_element_ids = [figure.id, *([caption.id] if caption is not None else [])]

        chunk = Chunk(
            paper_id=context.paper.id,
            section_id=figure.section_id,
            order=context.next_order(),
            modality=ChunkModality.FIGURE,
            text=text,
            asset_uri=figure.asset_uri,
            source_element_ids=source_element_ids,
            bounding_boxes=figure.bounding_boxes,
        )
        context.register_chunk(figure.id, chunk.id)
        return StrategyResult(chunks=[chunk], relationships=[])
