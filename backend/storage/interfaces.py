"""Storage abstraction: the port every concrete backend implements."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from backend.domain import PaperId


class WorkspaceStorage(ABC):
    """Abstraction over a per-document workspace's file storage.

    Every ingested document gets its own workspace, identified by its
    document id, where the original PDF and -- later -- every derived
    artifact (parsed JSON, extracted figures and tables) is stored.
    Concrete implementations decide where a workspace physically lives
    (local disk today; S3, Azure Blob, or GCS are all drop-in replacements
    later) without requiring any change to this interface's callers.

    `relative_path` is a plain string rather than a `pathlib.Path`, since a
    path is a filesystem-specific concept that an object-storage backend
    would not use for its keys.
    """

    @abstractmethod
    def create_workspace(self, document_id: PaperId) -> None:
        """Create a new, empty workspace for a document.

        Args:
            document_id: Identifier of the document the workspace is for.

        Raises:
            WorkspaceAlreadyExistsError: A workspace already exists for
                this document.
            StorageError: The workspace could not be created.
        """

    @abstractmethod
    def write_bytes(self, document_id: PaperId, relative_path: str, content: bytes) -> None:
        """Write raw bytes to a path within a document's workspace.

        Args:
            document_id: Identifier of the document whose workspace to write to.
            relative_path: Path of the file within the workspace.
            content: Bytes to write.

        Raises:
            WorkspaceNotFoundError: The workspace does not exist.
            StorageError: The write failed.
        """

    @abstractmethod
    def read_bytes(self, document_id: PaperId, relative_path: str) -> bytes:
        """Read raw bytes from a path within a document's workspace.

        Args:
            document_id: Identifier of the document whose workspace to read from.
            relative_path: Path of the file within the workspace.

        Returns:
            The file's raw bytes.

        Raises:
            WorkspaceNotFoundError: The workspace does not exist.
            StorageError: The read failed.
        """

    @abstractmethod
    def write_json(
        self, document_id: PaperId, relative_path: str, payload: Mapping[str, Any]
    ) -> None:
        """Write a JSON-serializable mapping to a path within a document's workspace.

        Args:
            document_id: Identifier of the document whose workspace to write to.
            relative_path: Path of the file within the workspace.
            payload: JSON-serializable mapping to write.

        Raises:
            WorkspaceNotFoundError: The workspace does not exist.
            StorageError: The write failed.
        """

    @abstractmethod
    def read_json(self, document_id: PaperId, relative_path: str) -> dict[str, Any]:
        """Read and parse a JSON file within a document's workspace.

        Args:
            document_id: Identifier of the document whose workspace to read from.
            relative_path: Path of the file within the workspace.

        Returns:
            The parsed JSON content.

        Raises:
            WorkspaceNotFoundError: The workspace does not exist.
            StorageError: The read or parse failed.
        """

    @abstractmethod
    def workspace_exists(self, document_id: PaperId) -> bool:
        """Check whether a workspace exists for a document.

        Args:
            document_id: Identifier of the document to check.

        Returns:
            `True` if a workspace exists for this document, `False` otherwise.
        """
