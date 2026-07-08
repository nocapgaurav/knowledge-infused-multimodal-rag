"""Local disk implementation of the workspace storage abstraction."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from backend.domain import PaperId
from backend.storage.exceptions import (
    StorageReadError,
    StorageWriteError,
    WorkspaceAlreadyExistsError,
    WorkspaceNotFoundError,
)
from backend.storage.interfaces import WorkspaceStorage


class LocalFilesystemStorage(WorkspaceStorage):
    """Stores each document's workspace as a directory on local disk.

    Every file is written under a fixed, predictable name (e.g. always
    `paper.pdf`, never the uploaded file's original name) so that no
    caller-supplied string ever becomes part of a filesystem path.

    Attributes:
        root: Base directory all workspaces are created under.
    """

    def __init__(self, root: Path) -> None:
        """Initialize the storage backend.

        Args:
            root: Base directory all workspaces are created under. Created
                on first use if it does not already exist.
        """
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def create_workspace(self, document_id: PaperId) -> None:
        workspace_dir = self._workspace_dir(document_id)
        if workspace_dir.exists():
            raise WorkspaceAlreadyExistsError(workspace_id=str(document_id))
        try:
            workspace_dir.mkdir(parents=True)
        except OSError as exc:
            raise StorageWriteError(workspace_id=str(document_id), relative_path=".") from exc

    def write_bytes(self, document_id: PaperId, relative_path: str, content: bytes) -> None:
        target = self._resolve_existing(document_id, relative_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        except OSError as exc:
            raise StorageWriteError(
                workspace_id=str(document_id), relative_path=relative_path
            ) from exc

    def read_bytes(self, document_id: PaperId, relative_path: str) -> bytes:
        target = self._resolve_existing(document_id, relative_path)
        try:
            return target.read_bytes()
        except OSError as exc:
            raise StorageReadError(
                workspace_id=str(document_id), relative_path=relative_path
            ) from exc

    def write_json(
        self, document_id: PaperId, relative_path: str, payload: Mapping[str, Any]
    ) -> None:
        try:
            content = json.dumps(payload, indent=2).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise StorageWriteError(
                workspace_id=str(document_id), relative_path=relative_path
            ) from exc
        self.write_bytes(document_id, relative_path, content)

    def read_json(self, document_id: PaperId, relative_path: str) -> dict[str, Any]:
        target = self._resolve_existing(document_id, relative_path)
        try:
            raw = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise StorageReadError(
                workspace_id=str(document_id), relative_path=relative_path
            ) from exc
        try:
            parsed: dict[str, Any] = json.loads(raw)
        except ValueError as exc:
            raise StorageReadError(
                workspace_id=str(document_id), relative_path=relative_path
            ) from exc
        return parsed

    def workspace_exists(self, document_id: PaperId) -> bool:
        return self._workspace_dir(document_id).is_dir()

    def _workspace_dir(self, document_id: PaperId) -> Path:
        return self._root / str(document_id)

    def _resolve_existing(self, document_id: PaperId, relative_path: str) -> Path:
        workspace_dir = self._workspace_dir(document_id)
        if not workspace_dir.is_dir():
            raise WorkspaceNotFoundError(workspace_id=str(document_id))
        return workspace_dir / relative_path
