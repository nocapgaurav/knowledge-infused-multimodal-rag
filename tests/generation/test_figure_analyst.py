"""Tests for the figure visual-analysis phase."""

from datetime import UTC, datetime
from uuid import uuid4

from backend.domain import PaperId
from backend.generation.models.prompt_context import ContextSection
from backend.generation.vision.figure_analyst import VISUAL_ANALYSIS_LABEL, FigureAnalyst
from backend.retrieval.models import (
    DiscoveryMethod,
    EvidenceBundle,
    GraphPath,
    RetrievalCandidate,
    RetrievalManifest,
    RetrievalStatistics,
    RetrievalTrace,
)


class _FakeStorage:
    def __init__(self, blobs=None, fail=False):
        self._blobs = blobs or {}
        self._fail = fail

    def read_bytes(self, document_id, relative_path):
        if self._fail:
            raise OSError("boom")
        return self._blobs[relative_path]

    def write_bytes(self, document_id, relative_path, content):  # pragma: no cover
        raise NotImplementedError


def _candidate(document_id, knowledge_unit_id, asset_uri=None, modality="figure"):
    return RetrievalCandidate(
        knowledge_unit_id=knowledge_unit_id,
        document_id=document_id,
        section_id=None,
        modality=modality,
        text="Fig. 1. A caption.",
        asset_uri=asset_uri,
        reading_order=0,
        citation_count=0,
        dense_similarity=0.5,
        discovery_method=DiscoveryMethod.DENSE_RETRIEVAL,
        graph_path=GraphPath(hops=()),
    )


def _bundle(document_id, candidates):
    return EvidenceBundle(
        document_id=document_id,
        query="Explain Figure 1.",
        candidates=tuple(candidates),
        evidence_groups=(),
        trace=RetrievalTrace(phases=(), dropped=()),
        manifest=RetrievalManifest(
            document_id=document_id,
            query="Explain Figure 1.",
            retrieval_version="1.0",
            retrieval_strategy_version="1.1",
            representation_version="r",
            embedding_version="e",
            graph_version="g",
            statistics=RetrievalStatistics(
                candidates_generated=1,
                candidates_expanded=0,
                candidates_scored=1,
                evidence_groups=0,
                evidence_items=0,
                duration_ms=1.0,
            ),
            created_at=datetime.now(UTC),
        ),
    )


def _figure_section(knowledge_unit_id):
    return ContextSection(
        citation_label="KU1",
        knowledge_unit_id=str(knowledge_unit_id),
        text="Fig. 1. A caption.",
        retrieval_context="Figure 1",
        modality="figure",
    )


def test_figure_section_gains_labeled_visual_analysis() -> None:
    document_id = PaperId(uuid4())
    ku = uuid4()
    analyst = FigureAnalyst(
        parsed_storage=_FakeStorage({"figures/a.png": b"png-bytes"}),
        describe=lambda image, instruction: "Three pipelines feed a fusion layer.",
    )

    sections, notes = analyst.augment(
        document_id,
        [_figure_section(ku)],
        _bundle(document_id, [_candidate(document_id, ku, asset_uri="figures/a.png")]),
    )

    assert VISUAL_ANALYSIS_LABEL in sections[0].text
    assert "Three pipelines feed a fusion layer." in sections[0].text
    assert sections[0].text.startswith("Fig. 1. A caption.")
    assert notes == ["KU1: figure image analyzed"]


def test_vision_failure_falls_back_to_caption_only() -> None:
    document_id = PaperId(uuid4())
    ku = uuid4()

    def exploding(image, instruction):
        raise RuntimeError("vision model down")

    analyst = FigureAnalyst(
        parsed_storage=_FakeStorage({"figures/a.png": b"png"}), describe=exploding
    )

    sections, notes = analyst.augment(
        document_id,
        [_figure_section(ku)],
        _bundle(document_id, [_candidate(document_id, ku, asset_uri="figures/a.png")]),
    )

    assert sections[0].text == "Fig. 1. A caption."
    assert notes == ["KU1: visual analysis unavailable; caption only"]


def test_non_figure_sections_and_disabled_analyst_are_untouched() -> None:
    document_id = PaperId(uuid4())
    ku = uuid4()
    text_section = ContextSection(
        citation_label="KU2", knowledge_unit_id=str(uuid4()), text="Body text.", modality="text"
    )
    disabled = FigureAnalyst(
        parsed_storage=_FakeStorage(), describe=lambda i, p: "x", enabled=False
    )

    sections, notes = disabled.augment(
        document_id,
        [_figure_section(ku), text_section],
        _bundle(document_id, [_candidate(document_id, ku, asset_uri="figures/a.png")]),
    )

    assert [s.text for s in sections] == ["Fig. 1. A caption.", "Body text."]
    assert notes == []
