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

_CITATION_GROUP_PATTERN = re.compile(r"[\[(]\s*(KU\d+(?:\s*(?:,|;|and|&)\s*KU\d+)*)\s*[\])]")
_CITATION_LABEL_PATTERN = re.compile(r"KU\d+")


def citation_labels_in(text: str) -> list[str]:
    """Every citation label used in `text`, in order of appearance.

    Shared with `grounding/grounding_validator.py`: both phases must agree
    on exactly what a citation looks like in generated text.

    Accepts `[KU1]` (the format the prompt asks for) plus the delimiter
    substitutions real models make often enough to matter -- observed
    live with qwen2.5: `(KU1)` and comma-separated lists like
    `(KU4, KU8)` or `[KU4, KU8]`. A delimited group must consist solely
    of KU labels and separators, so ordinary prose parentheses such as
    "(Doe, 2021)" are never treated as citations. Tolerating delimiters
    cannot fabricate evidence: every extracted label is still resolved
    against the exact label -> knowledge unit mapping the prompt was
    built from.

    Args:
        text: Generated answer text (or a single claim from it).

    Returns:
        Each label occurrence, in order, repeats included.
    """
    labels: list[str] = []
    for group in _CITATION_GROUP_PATTERN.finditer(text):
        labels.extend(_CITATION_LABEL_PATTERN.findall(group.group(1)))
    return labels


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
        labels_used.extend(
            label
            for label in _identity_citations(answer_text, prompt_context)
            if label not in labels_used
        )

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
                    display_label=section.retrieval_context,
                    page_numbers=section.page_numbers,
                    bounding_boxes=section.bounding_boxes,
                )
            )
        return CitationResolutionReport(resolved=tuple(resolved), unresolved=tuple(unresolved))


def _identity_citations(answer_text: str, prompt_context: PromptContext) -> list[str]:
    """Labels cited via their structural identity instead of `[KUn]`.

    Each evidence section is shown to the model with its identity in
    parentheses (e.g. "(Figure 2)"), and real models sometimes copy that
    identity as the citation -- observed live: "...affiliations provided
    in (Authors and affiliations (title page))". Resolution is an exact,
    case-sensitive match against the identities actually shown in this
    prompt, so it can only ever point at evidence the model was given --
    the same no-fabrication guarantee as label resolution.
    """
    labels: list[tuple[int, str]] = []
    for section in prompt_context.context_sections:
        identity = section.retrieval_context
        if not identity:
            continue
        index = answer_text.find(f"({identity})")
        if index != -1:
            labels.append((index, section.citation_label))
    return [label for _, label in sorted(labels)]


def _distinct_labels_in_order(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for label in citation_labels_in(text):
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered
