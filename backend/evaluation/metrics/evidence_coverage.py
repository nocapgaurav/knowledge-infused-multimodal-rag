"""Evidence Coverage: of what a generated answer actually cited, what
fraction was expected ground-truth evidence.

Precision-flavored, deliberately paired with `answer_completeness.py`'s
recall-flavored measure of the same two sets -- the same precision/recall
symmetry the retrieval metrics already use, applied to citations instead
of ranked results.
"""


def evidence_coverage(cited_knowledge_unit_ids: set[str], expected_citations: set[str]) -> float:
    """Compute evidence coverage for one generated answer.

    Args:
        cited_knowledge_unit_ids: Knowledge unit ids the answer's resolved
            citations actually reference.
        expected_citations: The case's ground-truth expected citation ids.

    Returns:
        `|cited ∩ expected| / |cited|`, or `0.0` if nothing was cited at
        all -- an answer that cites nothing cannot claim any of its
        (nonexistent) citations were correct.
    """
    if not cited_knowledge_unit_ids:
        return 0.0
    hits = len(cited_knowledge_unit_ids & expected_citations)
    return hits / len(cited_knowledge_unit_ids)
