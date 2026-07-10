"""JSON report: the machine-readable, authoritative view of a benchmark run.

The Markdown and HTML reports are derived, human-readable views of
exactly the same `EvaluationSummary` this renders -- never independently
computed, so the three formats can never disagree.
"""

import json

from backend.evaluation.models.evaluation_summary import EvaluationSummary


def render_json(summary: EvaluationSummary) -> str:
    """Render a benchmark run as pretty-printed JSON.

    Args:
        summary: The complete benchmark run to render.

    Returns:
        The JSON text, suitable for writing to `benchmark.json`.
    """
    return json.dumps(summary.model_dump(mode="json"), indent=2)
