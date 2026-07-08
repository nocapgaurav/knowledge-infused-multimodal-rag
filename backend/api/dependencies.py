"""Dependency injection providers for the API layer.

Every route depends on an interface (`WorkspaceStorage`) or a service that
itself depends only on that interface -- never on a concrete backend or on
module-level global state. `lru_cache` gives each provider a single
process-wide instance without needing a global variable.
"""

from functools import lru_cache

from fastapi import Depends

from backend.config.settings import Settings, get_settings
from backend.ingestion.service import DocumentIngestionService
from backend.parser.interfaces.document_parser import DocumentParser
from backend.parser.mapper.domain_mapper import DomainMapper
from backend.parser.providers.docling_parser import DoclingDocumentParser
from backend.parser.services.parser_service import ParserService
from backend.storage.interfaces import WorkspaceStorage
from backend.storage.local_filesystem import LocalFilesystemStorage


@lru_cache
def get_workspace_storage() -> WorkspaceStorage:
    """Return the process-wide raw document storage backend.

    Returns:
        A `WorkspaceStorage` implementation configured from application settings.
    """
    settings = get_settings()
    return LocalFilesystemStorage(root=settings.storage_root)


@lru_cache
def get_parsed_storage() -> WorkspaceStorage:
    """Return the process-wide parsed-artifact storage backend.

    A separate `WorkspaceStorage` instance from raw storage, rooted at a
    different directory -- the same abstraction, reused rather than
    duplicated, for a different artifact lifecycle.

    Returns:
        A `WorkspaceStorage` implementation configured from application settings.
    """
    settings = get_settings()
    return LocalFilesystemStorage(root=settings.parsed_storage_root)


@lru_cache
def get_document_parser() -> DocumentParser:
    """Return the process-wide document parser.

    Cached because constructing a `DoclingDocumentParser` loads Docling's
    layout and table-structure models -- an expensive operation that must
    happen once per process, not once per request.

    Returns:
        A `DocumentParser` implementation configured from application settings.
    """
    settings = get_settings()
    return DoclingDocumentParser(ocr_enabled=settings.docling_ocr_enabled)


@lru_cache
def get_domain_mapper() -> DomainMapper:
    """Return the process-wide domain mapper.

    Returns:
        A `DomainMapper` instance. Stateless, so a single shared instance is safe.
    """
    return DomainMapper()


def get_ingestion_service(
    storage: WorkspaceStorage = Depends(get_workspace_storage),
    settings: Settings = Depends(get_settings),
) -> DocumentIngestionService:
    """Return a document ingestion service wired to the configured storage backend.

    Args:
        storage: Workspace storage backend, injected.
        settings: Application settings, injected.

    Returns:
        A configured `DocumentIngestionService`.
    """
    return DocumentIngestionService(
        storage=storage, max_upload_size_bytes=settings.max_upload_size_bytes
    )


def get_parser_service(
    raw_storage: WorkspaceStorage = Depends(get_workspace_storage),
    parsed_storage: WorkspaceStorage = Depends(get_parsed_storage),
    document_parser: DocumentParser = Depends(get_document_parser),
    mapper: DomainMapper = Depends(get_domain_mapper),
) -> ParserService:
    """Return a parser service wired to the configured storage backends and parser.

    Args:
        raw_storage: Storage backend holding ingested documents, injected.
        parsed_storage: Storage backend for parsed artifacts, injected.
        document_parser: Parsing engine, injected.
        mapper: Domain mapper, injected.

    Returns:
        A configured `ParserService`.
    """
    return ParserService(
        raw_storage=raw_storage,
        parsed_storage=parsed_storage,
        document_parser=document_parser,
        mapper=mapper,
    )
