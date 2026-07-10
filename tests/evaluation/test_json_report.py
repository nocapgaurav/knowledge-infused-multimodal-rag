"""Unit tests for the JSON report -- the authoritative, machine-readable view."""

import json

from backend.evaluation.models.evaluation_summary import EvaluationSummary
from backend.evaluation.reports.json_report import render_json

from ._helpers import build_summary


def test_render_json_round_trips_through_the_summary_model() -> None:
    summary = build_summary()

    rendered = render_json(summary)
    parsed = EvaluationSummary.model_validate(json.loads(rendered))

    assert parsed == summary


def test_render_json_is_pretty_printed() -> None:
    summary = build_summary()

    rendered = render_json(summary)

    assert "\n" in rendered
