"""Storage layer: an abstraction over where a document's workspace lives.

Everything outside this package depends on `WorkspaceStorage`
(`backend.storage.interfaces`), never on a concrete backend -- that is what
lets the local filesystem implementation used today be replaced by S3,
Azure Blob, or GCS later without touching any caller.
"""
