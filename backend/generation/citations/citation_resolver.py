"""Phase 8: Citation Resolution.

The model is only ever allowed to cite by the short label it was shown
(`[KU1]`, `[KU2]`, ...). Nothing it writes is trusted as a real reference
until resolved here against the exact label -> knowledge unit mapping the
prompt was built from -- an invented, mistyped, or unshown label resolves
to nothing, never to a guess.
"""

import re

from backend.generation.models.citation import (
    CitationResolutionReport,
    ResolvedCitation,
    UnresolvedCitation,
)
from backend.generation.models.prompt_context import PromptContext

CITATION_LABEL_IN_TEXT_PATTERN = re.compile(r"\[(KU\d+)\]")
"""Shared with `grounding/grounding_validator.py`: both phases must agree
on exactly what a citation looks like in generated text."""


class CitationResolver:
    """Resolves every citation label used in a generated answer against
    the evidence actually shown to the model."""

    def resolve(self, answer_text: str, prompt_context: PromptContext) -> CitationResolutionReport:
        """Resolve every citation label found in the answer.

        Args:
            answer_text: The generated answer text.
            prompt_context: The prompt the answer was generated from --
                the sole source of truth for which labels are real.

        Returns:
            Every citation label used, partitioned into resolved (backed
            by real evidence) and unresolved (invented, mistyped, or
            otherwise absent from the prompt).
        """
        evidence_by_label = {
            section.citation_label: section for section in prompt_context.context_sections
        }
        labels_used = _distinct_labels_in_order(answer_text)

        resolved: list[ResolvedCitation] = []
        unresolved: list[UnresolvedCitation] = []
        for label in labels_used:
            section = evidence_by_label.get(label)
            if section is None:
                unresolved.append(
                    UnresolvedCitation(
                        label=label, reason="label was never shown to the model in this prompt"
                    )
                )
                continue
            resolved.append(
                ResolvedCitation(
                    label=label,
                    knowledge_unit_id=section.knowledge_unit_id,
                    text_excerpt=section.text,
                )
            )
        return CitationResolutionReport(resolved=tuple(resolved), unresolved=tuple(unresolved))


def _distinct_labels_in_order(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for label in CITATION_LABEL_IN_TEXT_PATTERN.findall(text):
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered
