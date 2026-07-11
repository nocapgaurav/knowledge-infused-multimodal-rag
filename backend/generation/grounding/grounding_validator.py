"""Phase 7: Grounding Validation.

Checks every factual claim the model produced against the evidence it
actually cited -- deterministically, with no LLM and no re-embedding
involved. That second constraint is not incidental: Module 10 must never
regenerate embeddings, which rules out the more sophisticated approach of
checking claim-vs-evidence *semantic* similarity (it would require
embedding both the claim and the evidence text fresh). What remains is a
lexical overlap check: how much of a claim's meaningful vocabulary
actually appears in the evidence it cites. This is documented as an
approximation, not a claim of deep semantic verification -- see the final
report's known limitations for what a future version could add.

Never silently accepts an unsupported answer: a `GroundingReport` is
always produced with a verdict for every claim, and `is_fully_grounded`
is `False` the moment even one claim fails.
"""

import re

from backend.generation.citations.citation_resolver import citation_labels_in
from backend.generation.exceptions import NoClaimsExtractedError
from backend.generation.models.grounding_report import (
    ClaimGroundingStatus,
    ClaimVerdict,
    GroundingReport,
)
from backend.generation.models.prompt_context import ContextSection, PromptContext

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_WORD_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "of",
        "in",
        "on",
        "to",
        "and",
        "or",
        "for",
        "with",
        "that",
        "this",
        "it",
        "as",
        "by",
        "from",
        "be",
        "at",
        "which",
        "its",
        "these",
        "those",
        "has",
        "have",
        "had",
        "not",
        "but",
        "can",
        "could",
        "will",
        "would",
        "may",
        "might",
        "also",
        "than",
        "into",
        "such",
        "each",
        "more",
        "most",
        "some",
        "any",
    }
)
_MINIMUM_LEXICAL_OVERLAP_RATIO = 0.5
"""At least half of a claim's meaningful (non-stopword) vocabulary must
appear in its cited evidence to count as supported. A documented,
named threshold, not an unexplained magic number -- chosen to require
substantial, not incidental, lexical overlap while tolerating normal
paraphrasing (word order, function words, minor rewording)."""


class GroundingValidator:
    """Verifies every claim in a generated answer against its cited evidence."""

    def validate(self, answer_text: str, prompt_context: PromptContext) -> GroundingReport:
        """Validate every claim in a generated answer.

        Args:
            answer_text: The generated answer text.
            prompt_context: The prompt the answer was generated from --
                the sole source of truth for what evidence was available.

        Returns:
            A verdict for every claim found in the answer.

        Raises:
            NoClaimsExtractedError: The answer contains no extractable
                claims (a structural defect -- an empty or whitespace-only
                answer should never reach this phase).
        """
        claims = _split_into_claims(answer_text)
        if not claims:
            raise NoClaimsExtractedError()

        evidence_by_label = {
            section.citation_label: section for section in prompt_context.context_sections
        }

        return GroundingReport(
            claims=tuple(_verdict_for(claim, evidence_by_label) for claim in claims)
        )


def _verdict_for(claim: str, evidence_by_label: dict[str, ContextSection]) -> ClaimVerdict:
    labels = citation_labels_in(claim)
    if not labels:
        return ClaimVerdict(
            claim_text=claim,
            cited_labels=(),
            status=ClaimGroundingStatus.MISSING_CITATION,
            reason="claim carries no citation",
        )

    unresolved = [label for label in labels if label not in evidence_by_label]
    if unresolved:
        return ClaimVerdict(
            claim_text=claim,
            cited_labels=tuple(labels),
            status=ClaimGroundingStatus.UNRESOLVED_CITATION,
            reason=f"citation label(s) not shown to the model: {unresolved}",
        )

    combined_evidence_text = " ".join(
        f"{evidence_by_label[label].retrieval_context or ''} {evidence_by_label[label].text}"
        for label in labels
    )
    if _is_supported(claim, combined_evidence_text):
        return ClaimVerdict(
            claim_text=claim,
            cited_labels=tuple(labels),
            status=ClaimGroundingStatus.GROUNDED,
            reason="cited evidence lexically supports the claim",
        )
    return ClaimVerdict(
        claim_text=claim,
        cited_labels=tuple(labels),
        status=ClaimGroundingStatus.UNSUPPORTED,
        reason="cited evidence does not sufficiently overlap with the claim",
    )


_MINIMUM_ALPHABETIC_CHARACTERS_FOR_A_CLAIM = 3
"""A sentence-boundary regex alone splits a numbered-list marker like
"2." into its own fragment whenever the answer uses a markdown-style
numbered list (confirmed against a real model response: "...0.87.\\n\\n2.
**Method Stages**..." splits after both "0.87." and "2."), producing a
spurious near-empty "claim" that can never carry a citation. Filtering out
fragments with fewer than a handful of alphabetic characters discards
those punctuation-only artifacts without discarding any real claim, which
always has substantially more alphabetic content than this."""


def _split_into_claims(answer_text: str) -> list[str]:
    fragments = (claim.strip() for claim in _SENTENCE_SPLIT_PATTERN.split(answer_text.strip()))
    return [claim for claim in fragments if _looks_like_a_claim(claim)]


def _looks_like_a_claim(fragment: str) -> bool:
    alphabetic_count = sum(1 for character in fragment if character.isalpha())
    return alphabetic_count >= _MINIMUM_ALPHABETIC_CHARACTERS_FOR_A_CLAIM


def _is_supported(claim: str, evidence_text: str) -> bool:
    """Whether a claim's vocabulary is sufficiently covered by the cited text.

    Called with the UNION of every cited section's text, each prefixed
    with its structural identity: a synthesis sentence legitimately draws
    vocabulary from all the evidence it cites, and a sentence like "the
    authors are listed on the title page [KU1]" legitimately draws its
    vocabulary from the evidence's identity ("Authors and affiliations
    (title page)") rather than from the names in its body.
    """
    claim_words = _significant_words(claim)
    if not claim_words:
        return True  # a claim with no meaningful vocabulary (e.g. "Yes.") has nothing to check
    evidence_words = _significant_words(evidence_text)
    overlap_ratio = len(claim_words & evidence_words) / len(claim_words)
    return overlap_ratio >= _MINIMUM_LEXICAL_OVERLAP_RATIO


def _significant_words(text: str) -> set[str]:
    return {word for word in _WORD_PATTERN.findall(text.lower()) if word not in _STOPWORDS}
