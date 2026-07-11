"""Tests for the knowledge representation service, using real storage
(against tmp_path) and the real builder -- both are fast, pure Python."""

import json
from pathlib import Path
from uuid import uuid4

import pytest

from backend.chunking.builder.knowledge_builder import KnowledgeBuilder
from backend.chunking.exceptions import PaperNotParsedError
from backend.chunking.services.knowledge_representation_service import (
    KnowledgeRepresentationService,
)
from backend.domain import Metadata, Paper, PaperId, Paragraph, Section
from backend.storage.local_filesystem import LocalFilesystemStorage


def _seed_parsed_paper(parsed_storage: LocalFilesystemStorage, paper: Paper) -> None:
    parsed_storage.create_workspace(paper.id)
    parsed_storage.write_json(paper.id, "paper.json", paper.model_dump(mode="json"))


def _sample_paper() -> Paper:
    paper_id = PaperId(uuid4())
    section = Section(paper_id=paper_id, title="Introduction", level=1, order=0)
    paragraph = Paragraph(paper_id=paper_id, section_id=section.id, order=0, text="Body text.")
    return Paper(
        id=paper_id,
        metadata=Metadata(title="A Paper", source_filename="p.pdf"),
        sections=[section],
        paragraphs=[paragraph],
    )


@pytest.fixture
def service(
    tmp_path: Path,
) -> tuple[KnowledgeRepresentationService, LocalFilesystemStorage, LocalFilesystemStorage]:
    parsed_storage = LocalFilesystemStorage(root=tmp_path / "parsed")
    knowledge_storage = LocalFilesystemStorage(root=tmp_path / "knowledge")
    builder = KnowledgeBuilder(max_words_per_chunk=250, min_words_per_chunk=4)
    return (
        KnowledgeRepresentationService(
            parsed_storage=parsed_storage, knowledge_storage=knowledge_storage, builder=builder
        ),
        parsed_storage,
        knowledge_storage,
    )


def test_represent_document_persists_expected_artifacts(
    service: tuple[KnowledgeRepresentationService, LocalFilesystemStorage, LocalFilesystemStorage],
    tmp_path: Path,
) -> None:
    representation_service, parsed_storage, _ = service
    paper = _sample_paper()
    _seed_parsed_paper(parsed_storage, paper)

    chunks, relationships = representation_service.represent_document(paper.id)

    assert len(chunks) == 2  # the paragraph chunk + the title chunk
    workspace = tmp_path / "knowledge" / str(paper.id)
    units_payload = json.loads((workspace / "knowledge_units.json").read_text())
    relationships_payload = json.loads((workspace / "relationships.json").read_text())

    assert units_payload["document_id"] == str(paper.id)
    assert units_payload["count"] == 2
    assert len(units_payload["chunks"]) == 2
    assert relationships_payload["count"] == len(relationships)


def test_represent_document_raises_for_unparsed_document(
    service: tuple[KnowledgeRepresentationService, LocalFilesystemStorage, LocalFilesystemStorage],
) -> None:
    representation_service, _, _ = service

    with pytest.raises(PaperNotParsedError):
        representation_service.represent_document(PaperId(uuid4()))
