"""Visual analysis of figure evidence for figure-centric questions.

When a question is about a figure, the figure's stored image -- extracted
by the parser from the paper itself -- is analyzed by a local vision
model, and the observation is appended to that figure's evidence text,
clearly labeled as automated visual analysis. The model answering the
question then reasons from what the figure actually shows, not merely
from its caption and surrounding prose.

Grounding note: the analysis is derived exclusively from the paper's own
figure image (an artifact of the uploaded document), so treating it as
evidence context introduces no outside knowledge. It is always labeled as
automated analysis, never presented as the paper's own words.

Failure policy: visual analysis is an enhancement, never a dependency --
any failure (missing asset, vision model unavailable, timeout) logs and
falls back to the caption-only behavior that existed before this module.
"""

import logging
from collections.abc import Callable

from backend.domain import ChunkModality, PaperId
from backend.generation.models.prompt_context import ContextSection
from backend.retrieval.models import EvidenceBundle
from backend.storage.interfaces import WorkspaceStorage

logger = logging.getLogger(__name__)

VISUAL_ANALYSIS_LABEL = "Visual content (automated analysis of the figure image)"

_VISION_INSTRUCTION = (
    "You are looking at a figure from a scientific paper. Its caption is: "
    "{caption!r}. Describe concretely what the figure visually shows: its "
    "components, layout, axes or flows, arrows and connections, and any "
    "labels or values you can read. Note visible patterns or trends. "
    "Describe ONLY what is visible in the image -- never speculate about "
    "content you cannot see."
)

VisionDescriber = Callable[[bytes, str], str]
"""Given image bytes and an instruction, return a textual description.
Injected as a plain callable so tests never need a live vision model."""


class FigureAnalyst:
    """Turns figure evidence into first-class visual evidence."""

    def __init__(
        self,
        parsed_storage: WorkspaceStorage,
        describe: VisionDescriber,
        enabled: bool = True,
    ) -> None:
        """Initialize the analyst.

        Args:
            parsed_storage: Storage the parser wrote figure images to.
            describe: The vision call -- image bytes plus instruction in,
                description out.
            enabled: Master switch; disabled behaves as a no-op.
        """
        self._parsed_storage = parsed_storage
        self._describe = describe
        self._enabled = enabled

    def augment(
        self,
        document_id: PaperId,
        sections: list[ContextSection],
        bundle: EvidenceBundle,
    ) -> tuple[list[ContextSection], list[str]]:
        """Append visual analysis to every figure section that has an image.

        Args:
            document_id: The document the figures belong to.
            sections: The optimized evidence context (Phase 3's output).
            bundle: The evidence bundle, for asset lookups.

        Returns:
            The (possibly augmented) sections plus human-readable notes on
            what was analyzed or skipped.
        """
        if not self._enabled:
            return sections, []

        asset_by_id = {
            str(candidate.knowledge_unit_id): candidate.asset_uri
            for candidate in bundle.candidates
            if candidate.asset_uri
        }

        augmented: list[ContextSection] = []
        notes: list[str] = []
        for section in sections:
            if section.modality is not ChunkModality.FIGURE:
                augmented.append(section)
                continue
            asset_uri = asset_by_id.get(section.knowledge_unit_id)
            if not asset_uri:
                augmented.append(section)
                notes.append(f"{section.citation_label}: figure has no stored image; caption only")
                continue
            analysis = self._analyze(document_id, asset_uri, section.text)
            if analysis is None:
                augmented.append(section)
                notes.append(f"{section.citation_label}: visual analysis unavailable; caption only")
                continue
            augmented.append(
                section.model_copy(
                    update={"text": f"{section.text}\n\n{VISUAL_ANALYSIS_LABEL}: {analysis}"}
                )
            )
            notes.append(f"{section.citation_label}: figure image analyzed")
        return augmented, notes

    def _analyze(self, document_id: PaperId, asset_uri: str, caption: str) -> str | None:
        try:
            image = self._parsed_storage.read_bytes(document_id, asset_uri)
        except Exception:
            logger.warning(
                "figure image could not be read; skipping visual analysis",
                exc_info=True,
                extra={"document_id": str(document_id), "asset_uri": asset_uri},
            )
            return None
        try:
            description = self._describe(image, _VISION_INSTRUCTION.format(caption=caption))
        except Exception:
            logger.warning(
                "vision model call failed; skipping visual analysis",
                exc_info=True,
                extra={"document_id": str(document_id), "asset_uri": asset_uri},
            )
            return None
        description = description.strip()
        return description or None
