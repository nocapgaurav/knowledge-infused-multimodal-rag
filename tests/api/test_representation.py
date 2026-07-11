"""End-to-end tests for the knowledge representation API.

Seeds parsed storage directly with a hand-built `Paper` rather than running
a real parse first -- this test verifies routing, dependency wiring, and
status-code mapping, not the parser (covered elsewhere).
"""

from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.dependencies import get_knowledge_storage, get_parsed_storage
from backend.domain import Metadata, Paper, PaperId, Paragraph, Section
from backend.storage.local_filesystem import LocalFilesystemStorage


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
def client_and_parsed_storage(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, LocalFilesystemStorage]]:
    app = create_app()
    parsed_storage = LocalFilesystemStorage(root=tmp_path / "parsed")
    knowledge_storage = LocalFilesystemStorage(root=tmp_path / "knowledge")
    app.dependency_overrides[get_parsed_storage] = lambda: parsed_storage
    app.dependency_overrides[get_knowledge_storage] = lambda: knowledge_storage
    with TestClient(app) as test_client:
        yield test_client, parsed_storage


def test_represent_document_returns_counts(
    client_and_parsed_storage: tuple[TestClient, LocalFilesystemStorage],
) -> None:
    client, parsed_storage = client_and_parsed_storage
    paper = _sample_paper()
    parsed_storage.create_workspace(paper.id)
    parsed_storage.write_json(paper.id, "paper.json", paper.model_dump(mode="json"))

    response = client.post(f"/documents/{paper.id}/represent")

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == str(paper.id)
    assert body["knowledge_units"] == 2  # the paragraph chunk + the title chunk
    assert body["status"] == "REPRESENTED"
    assert isinstance(body["relationships"], int)


def test_represent_document_returns_404_for_unparsed_document(
    client_and_parsed_storage: tuple[TestClient, LocalFilesystemStorage],
) -> None:
    client, _ = client_and_parsed_storage

    response = client.post(f"/documents/{uuid4()}/represent")

    assert response.status_code == 404
