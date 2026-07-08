"""Exceptions raised by the storage layer.

Every concrete `WorkspaceStorage` implementation raises these, not its own
backend-specific errors (e.g. `OSError`, a future S3 client's exceptions),
so callers can handle storage failures without knowing which backend is in use.
"""


class StorageError(Exception):
    """Base class for all storage backend errors."""


class WorkspaceAlreadyExistsError(StorageError):
    """Raised when creating a workspace that already exists."""

    def __init__(self, *, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        super().__init__(f"workspace '{workspace_id}' already exists")


class WorkspaceNotFoundError(StorageError):
    """Raised when operating on a workspace that does not exist."""

    def __init__(self, *, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        super().__init__(f"workspace '{workspace_id}' does not exist")


class StorageWriteError(StorageError):
    """Raised when writing to a workspace fails."""

    def __init__(self, *, workspace_id: str, relative_path: str) -> None:
        self.workspace_id = workspace_id
        self.relative_path = relative_path
        super().__init__(f"failed to write '{relative_path}' in workspace '{workspace_id}'")


class StorageReadError(StorageError):
    """Raised when reading from a workspace fails."""

    def __init__(self, *, workspace_id: str, relative_path: str) -> None:
        self.workspace_id = workspace_id
        self.relative_path = relative_path
        super().__init__(f"failed to read '{relative_path}' in workspace '{workspace_id}'")
