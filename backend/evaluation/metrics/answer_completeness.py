"""Answer Completeness: of the ground-truth expected citations, what
fraction actually made it into the generated answer.

Recall-flavored counterpart to `evidence_coverage.py`'s precision-flavored
measure of the same two sets.
"""


def answer_completeness(cited_knowledge_unit_ids: set[str], expected_citations: set[str]) -> float:
    """Compute answer completeness for one generated answer.

    Args:
        cited_knowledge_unit_ids: Knowledge unit ids the answer's resolved
            citations actually reference.
        expected_citations: The case's ground-truth expected citation ids.
            Never empty in practice -- validated as non-empty at dataset
            load time.

    Returns:
        `|cited ∩ expected| / |expected|`.
    """
    if not expected_citations:
        return 1.0
    hits = len(cited_knowledge_unit_ids & expected_citations)
    return hits / len(expected_citations)
