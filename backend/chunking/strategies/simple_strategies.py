"""Trivial, direct-mapping strategies: title, abstract, and bibliography entries.

Grouped in one file since none has splitting, merging, or relationship
-detection logic -- each is a single entity becoming a single chunk.
"""

from backend.chunking.interfaces.context import BuildContext, StrategyResult
from backend.domain import Chunk, ChunkModality, Reference


class TitleStrategy:
    """Builds one knowledge unit from the paper's title.

    Without this, the title exists only in `Paper.metadata` and is
    invisible to retrieval entirely -- observed live: "What is the title?"
    could not retrieve the title because no chunk contained it.
    """

    def build(self, context: BuildContext) -> StrategyResult:
        """Build a knowledge unit for the paper's title.

        Args:
            context: Shared build context.

        Returns:
            A single-chunk result. The parser guarantees a title exists.
        """
        title = context.paper.metadata.title
        chunk = Chunk(
            paper_id=context.paper.id,
            section_id=None,
            order=context.next_order(),
            modality=ChunkModality.TEXT,
            text=title,
            retrieval_context="Title of this paper",
            token_count=len(title.split()),
            source_element_ids=[context.paper.metadata.id],
            bounding_boxes=[],
        )
        return StrategyResult(chunks=[chunk], relationships=[])


class AbstractStrategy:
    """Builds one knowledge unit from the paper's abstract, if it has one.

    The parser (Module 4) already extracts the abstract out of the section
    body into `Paper.metadata.abstract` -- without this strategy, that text
    would be silently invisible to retrieval entirely, since it never
    appears in `paper.sections`/`paper.paragraphs`.
    """

    def build(self, context: BuildContext) -> StrategyResult:
        """Build a knowledge unit for the paper's abstract.

        Args:
            context: Shared build context.

        Returns:
            A single-chunk result, or an empty result if the paper has no abstract.
        """
        metadata = context.paper.metadata
        if not metadata.abstract:
            return StrategyResult(chunks=[], relationships=[])

        chunk = Chunk(
            paper_id=context.paper.id,
            section_id=None,
            order=context.next_order(),
            modality=ChunkModality.TEXT,
            text=metadata.abstract,
            retrieval_context="Abstract",
            token_count=len(metadata.abstract.split()),
            source_element_ids=[metadata.id],
            bounding_boxes=[],
        )
        return StrategyResult(chunks=[chunk], relationships=[])


class ReferenceStrategy:
    """Builds one knowledge unit per bibliography entry.

    Registered with the build context so in-text citations (detected by
    `ParagraphStrategy.detect_relationships`) can resolve to this chunk.
    """

    def build(self, reference: Reference, context: BuildContext) -> StrategyResult:
        """Build a knowledge unit for a single reference.

        Args:
            reference: The bibliography entry to build a unit for.
            context: Shared build context.

        Returns:
            A single-chunk result.
        """
        chunk = Chunk(
            paper_id=context.paper.id,
            section_id=None,
            order=context.next_order(),
            modality=ChunkModality.TEXT,
            text=reference.raw_text,
            retrieval_context=f"Bibliography reference [{reference.order + 1}]",
            token_count=len(reference.raw_text.split()),
            source_element_ids=[reference.id],
            bounding_boxes=[],
        )
        context.register_chunk(reference.id, chunk.id)
        return StrategyResult(chunks=[chunk], relationships=[])
