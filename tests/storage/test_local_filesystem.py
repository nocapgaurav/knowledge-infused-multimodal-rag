"""Tests for the local filesystem storage backend."""

from pathlib import Path
from uuid import uuid4

import pytest

from backend.domain import PaperId
from backend.storage.exceptions import WorkspaceAlreadyExistsError, WorkspaceNotFoundError
from backend.storage.local_filesystem import LocalFilesystemStorage


@pytest.fixture
def storage(tmp_path: Path) -> LocalFilesystemStorage:
    return LocalFilesystemStorage(root=tmp_path / "raw")


def test_create_workspace_creates_a_directory(
    storage: LocalFilesystemStorage, tmp_path: Path
) -> None:
    document_id = PaperId(uuid4())

    storage.create_workspace(document_id)

    assert (tmp_path / "raw" / str(document_id)).is_dir()


def test_create_workspace_raises_if_already_exists(storage: LocalFilesystemStorage) -> None:
    document_id = PaperId(uuid4())
    storage.create_workspace(document_id)

    with pytest.raises(WorkspaceAlreadyExistsError):
        storage.create_workspace(document_id)


def test_write_and_read_json_round_trips(storage: LocalFilesystemStorage) -> None:
    document_id = PaperId(uuid4())
    storage.create_workspace(document_id)

    storage.write_json(document_id, "status.json", {"status": "UPLOADED"})

    assert storage.read_json(document_id, "status.json") == {"status": "UPLOADED"}


def test_write_bytes_persists_file_content(storage: LocalFilesystemStorage, tmp_path: Path) -> None:
    document_id = PaperId(uuid4())
    storage.create_workspace(document_id)

    storage.write_bytes(document_id, "paper.pdf", b"%PDF-1.4 content")

    assert (tmp_path / "raw" / str(document_id) / "paper.pdf").read_bytes() == (b"%PDF-1.4 content")


def test_workspace_exists_reflects_actual_state(storage: LocalFilesystemStorage) -> None:
    document_id = PaperId(uuid4())

    assert storage.workspace_exists(document_id) is False

    storage.create_workspace(document_id)

    assert storage.workspace_exists(document_id) is True


def test_read_json_raises_if_workspace_missing(storage: LocalFilesystemStorage) -> None:
    with pytest.raises(WorkspaceNotFoundError):
        storage.read_json(PaperId(uuid4()), "status.json")


def test_write_bytes_raises_if_workspace_missing(storage: LocalFilesystemStorage) -> None:
    with pytest.raises(WorkspaceNotFoundError):
        storage.write_bytes(PaperId(uuid4()), "paper.pdf", b"content")
