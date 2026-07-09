"""Dependency injection providers for the API layer.

Every route depends on an interface (`WorkspaceStorage`) or a service that
itself depends only on that interface -- never on a concrete backend or on
module-level global state. `lru_cache` gives each provider a single
process-wide instance without needing a global variable.
"""

from functools import lru_cache

from fastapi import Depends

from backend.chunking.builder.knowledge_builder import KnowledgeBuilder
from backend.chunking.services.knowledge_representation_service import (
    KnowledgeRepresentationService,
)
from backend.config.settings import Settings, get_settings
from backend.embeddings.interfaces.embedding_provider import EmbeddingProvider
from backend.embeddings.planner.embedding_planner import EmbeddingPlanner
from backend.embeddings.providers.sentence_transformers_provider import (
    SentenceTransformersProvider,
)
from backend.embeddings.repository.embedding_repository import EmbeddingRepository
from backend.embeddings.services.embedding_service import EmbeddingService
from backend.graph.interfaces.knowledge_graph_store import KnowledgeGraphStore
from backend.graph.providers.neo4j_provider import Neo4jProvider
from backend.graph.repository.graph_repository import GraphRepository
from backend.graph.services.graph_service import GraphService
from backend.ingestion.service import DocumentIngestionService
from backend.parser.interfaces.document_parser import DocumentParser
from backend.parser.mapper.domain_mapper import DomainMapper
from backend.parser.providers.docling_parser import DoclingDocumentParser
from backend.parser.services.parser_service import ParserService
from backend.search.interfaces.vector_store import VectorStore
from backend.search.providers.qdrant_provider import QdrantProvider
from backend.search.repository.index_repository import IndexRepository
from backend.search.services.indexing_service import IndexingService
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


@lru_cache
def get_knowledge_storage() -> WorkspaceStorage:
    """Return the process-wide knowledge representation storage backend.

    A third `WorkspaceStorage` instance, rooted at yet another directory --
    the same reused abstraction as raw and parsed storage.

    Returns:
        A `WorkspaceStorage` implementation configured from application settings.
    """
    settings = get_settings()
    return LocalFilesystemStorage(root=settings.knowledge_storage_root)


def get_knowledge_builder(settings: Settings = Depends(get_settings)) -> KnowledgeBuilder:
    """Return a knowledge builder configured from application settings.

    Args:
        settings: Application settings, injected.

    Returns:
        A configured `KnowledgeBuilder`.
    """
    return KnowledgeBuilder(
        max_words_per_chunk=settings.max_words_per_knowledge_unit,
        min_words_per_chunk=settings.min_words_per_knowledge_unit,
    )


def get_knowledge_representation_service(
    parsed_storage: WorkspaceStorage = Depends(get_parsed_storage),
    knowledge_storage: WorkspaceStorage = Depends(get_knowledge_storage),
    builder: KnowledgeBuilder = Depends(get_knowledge_builder),
) -> KnowledgeRepresentationService:
    """Return a knowledge representation service wired to the configured storage backends.

    Args:
        parsed_storage: Storage backend holding parsed `Paper` artifacts, injected.
        knowledge_storage: Storage backend for representation artifacts, injected.
        builder: Knowledge builder, injected.

    Returns:
        A configured `KnowledgeRepresentationService`.
    """
    return KnowledgeRepresentationService(
        parsed_storage=parsed_storage, knowledge_storage=knowledge_storage, builder=builder
    )


@lru_cache
def get_embeddings_storage() -> WorkspaceStorage:
    """Return the process-wide embedding artifact storage backend.

    A fourth `WorkspaceStorage` instance, rooted at yet another directory --
    the same reused abstraction as every other artifact lifecycle.

    Returns:
        A `WorkspaceStorage` implementation configured from application settings.
    """
    settings = get_settings()
    return LocalFilesystemStorage(root=settings.embeddings_storage_root)


@lru_cache
def get_text_embedding_provider() -> EmbeddingProvider:
    """Return the process-wide text embedding provider.

    Cached because constructing a `SentenceTransformersProvider` loads
    model weights and resolves a pinned revision over the network -- an
    expensive operation that must happen once per process, not once per
    request.

    Returns:
        An `EmbeddingProvider` implementation configured from application settings.
    """
    settings = get_settings()
    return SentenceTransformersProvider(
        model_name=settings.embedding_model_name,
        revision=settings.embedding_model_revision,
        batch_size=settings.embedding_batch_size,
    )


def get_embedding_repository(
    knowledge_storage: WorkspaceStorage = Depends(get_knowledge_storage),
    embeddings_storage: WorkspaceStorage = Depends(get_embeddings_storage),
) -> EmbeddingRepository:
    """Return an embedding repository wired to the configured storage backends.

    Args:
        knowledge_storage: Storage backend holding knowledge representation
            artifacts, injected.
        embeddings_storage: Storage backend for embedding artifacts, injected.

    Returns:
        A configured `EmbeddingRepository`.
    """
    return EmbeddingRepository(
        knowledge_storage=knowledge_storage, embeddings_storage=embeddings_storage
    )


def get_embedding_service(
    repository: EmbeddingRepository = Depends(get_embedding_repository),
    text_provider: EmbeddingProvider = Depends(get_text_embedding_provider),
    settings: Settings = Depends(get_settings),
) -> EmbeddingService:
    """Return an embedding service wired to the configured repository and provider.

    Args:
        repository: Embedding repository, injected.
        text_provider: Text embedding provider, injected.
        settings: Application settings, injected.

    Returns:
        A configured `EmbeddingService`. No image embedding provider is
        configured -- see `interfaces/embedding_provider.py` for why that's
        a deliberate scope boundary.
    """
    return EmbeddingService(
        repository=repository,
        text_provider=text_provider,
        planner=EmbeddingPlanner(),
        batch_size=settings.embedding_batch_size,
    )


@lru_cache
def get_index_storage() -> WorkspaceStorage:
    """Return the process-wide index manifest storage backend.

    A fifth `WorkspaceStorage` instance, rooted at yet another directory --
    the same reused abstraction as every other artifact lifecycle.

    Returns:
        A `WorkspaceStorage` implementation configured from application settings.
    """
    settings = get_settings()
    return LocalFilesystemStorage(root=settings.index_storage_root)


@lru_cache
def get_vector_store() -> VectorStore:
    """Return the process-wide vector store.

    Cached because constructing a `QdrantProvider` opens a client
    connection -- reused across requests rather than reconnecting per call.

    Returns:
        A `VectorStore` implementation configured from application settings.
    """
    settings = get_settings()
    return QdrantProvider(url=settings.qdrant_url)


def get_index_repository(
    embeddings_storage: WorkspaceStorage = Depends(get_embeddings_storage),
    knowledge_storage: WorkspaceStorage = Depends(get_knowledge_storage),
    index_storage: WorkspaceStorage = Depends(get_index_storage),
) -> IndexRepository:
    """Return an index repository wired to the configured storage backends.

    Args:
        embeddings_storage: Storage backend holding embedding artifacts, injected.
        knowledge_storage: Storage backend holding knowledge representation
            artifacts, injected.
        index_storage: Storage backend for index manifests, injected.

    Returns:
        A configured `IndexRepository`.
    """
    return IndexRepository(
        embeddings_storage=embeddings_storage,
        knowledge_storage=knowledge_storage,
        index_storage=index_storage,
    )


def get_indexing_service(
    repository: IndexRepository = Depends(get_index_repository),
    vector_store: VectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_settings),
) -> IndexingService:
    """Return an indexing service wired to the configured repository and vector store.

    Args:
        repository: Index repository, injected.
        vector_store: Vector store, injected.
        settings: Application settings, injected.

    Returns:
        A configured `IndexingService`.
    """
    return IndexingService(
        repository=repository,
        vector_store=vector_store,
        collection_prefix=settings.qdrant_collection_prefix,
        batch_size=settings.index_batch_size,
    )


@lru_cache
def get_graph_storage() -> WorkspaceStorage:
    """Return the process-wide graph manifest storage backend.

    A sixth `WorkspaceStorage` instance, rooted at yet another directory --
    the same reused abstraction as every other artifact lifecycle.

    Returns:
        A `WorkspaceStorage` implementation configured from application settings.
    """
    settings = get_settings()
    return LocalFilesystemStorage(root=settings.graph_storage_root)


@lru_cache
def get_graph_store() -> KnowledgeGraphStore:
    """Return the process-wide knowledge graph store.

    Cached because constructing a `Neo4jProvider` opens a driver connection
    pool -- reused across requests rather than reconnecting per call.

    Returns:
        A `KnowledgeGraphStore` implementation configured from application settings.
    """
    settings = get_settings()
    return Neo4jProvider(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )


def get_graph_repository(
    knowledge_storage: WorkspaceStorage = Depends(get_knowledge_storage),
    graph_storage: WorkspaceStorage = Depends(get_graph_storage),
) -> GraphRepository:
    """Return a graph repository wired to the configured storage backends.

    Args:
        knowledge_storage: Storage backend holding knowledge representation
            artifacts, injected.
        graph_storage: Storage backend for graph manifests, injected.

    Returns:
        A configured `GraphRepository`.
    """
    return GraphRepository(knowledge_storage=knowledge_storage, graph_storage=graph_storage)


def get_graph_service(
    repository: GraphRepository = Depends(get_graph_repository),
    store: KnowledgeGraphStore = Depends(get_graph_store),
) -> GraphService:
    """Return a graph service wired to the configured repository and store.

    Args:
        repository: Graph repository, injected.
        store: Knowledge graph store, injected.

    Returns:
        A configured `GraphService`.
    """
    return GraphService(repository=repository, store=store)
