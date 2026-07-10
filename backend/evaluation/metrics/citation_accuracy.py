"""Citation Accuracy: of the citation labels the model attempted to use,
what fraction resolved to real evidence.

Derived from `GroundedResponse.generation_statistics`, the same values
Module 10's own Citation Resolution (Phase 8) already computed.
"""


def citation_accuracy(citations_resolved: int, citations_unresolved: int) -> float:
    """Compute citation accuracy for one generated answer.

    Args:
        citations_resolved: Citation labels that resolved to real evidence
            (`GenerationStatistics.citations_resolved`).
        citations_unresolved: Citation labels that did not
            (`GenerationStatistics.citations_unresolved`).

    Returns:
        `citations_resolved / (citations_resolved + citations_unresolved)`,
        or `1.0` if no citation was attempted at all -- vacuously nothing
        was wrong, since nothing was attempted (mirrors the same vacuous
        case Module 10's own `citation_coverage` signal uses).
    """
    total = citations_resolved + citations_unresolved
    if total == 0:
        return 1.0
    return citations_resolved / total
