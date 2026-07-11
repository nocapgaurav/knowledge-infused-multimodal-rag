"""Dependency injection providers for the API layer.

Every route depends on an interface (`WorkspaceStorage`) or a service that
itself depends only on that interface -- never on a concrete backend or on
module-level global state. `lru_cache` gives each provider a single
process-wide instance without needing a global variable.
"""

from functools import lru_cache

from fastapi import Depends
from ollama import Client as OllamaClient

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
from backend.evaluation.benchmark.benchmark_runner import BenchmarkRunner
from backend.evaluation.repository.evaluation_repository import EvaluationRepository
from backend.evaluation.services.evaluation_service import EvaluationService
from backend.evaluation.validation.evaluation_validator import EvaluationValidator
from backend.generation.citations.citation_resolver import CitationResolver
from backend.generation.context.context_optimizer import ContextOptimizer
from backend.generation.formatting.response_formatter import ResponseFormatter
from backend.generation.grounding.grounding_validator import GroundingValidator
from backend.generation.interfaces.generation_provider import GenerationProvider
from backend.generation.models.generation_config import GenerationConfig
from backend.generation.planner.answer_planner import AnswerPlanner
from backend.generation.prompt.prompt_composer import PromptComposer
from backend.generation.prompt.prompt_validator import PromptValidator
from backend.generation.providers.ollama_provider import OllamaProvider
from backend.generation.quality.answer_quality_assessor import AnswerQualityAssessor
from backend.generation.repository.generation_repository import GenerationRepository
from backend.generation.services.generation_service import GenerationService
from backend.generation.validation.generation_validator import GenerationValidator
from backend.generation.vision.figure_analyst import FigureAnalyst
from backend.graph.interfaces.knowledge_graph_store import KnowledgeGraphStore
from backend.graph.providers.neo4j_provider import Neo4jProvider
from backend.graph.repository.graph_repository import GraphRepository
from backend.graph.services.graph_service import GraphService
from backend.ingestion.service import DocumentIngestionService
from backend.parser.interfaces.document_parser import DocumentParser
from backend.parser.mapper.domain_mapper import DomainMapper
from backend.parser.providers.docling_parser import DoclingDocumentParser
from backend.parser.services.parser_service import ParserService
from backend.retrieval.assembly.evidence_assembler import AssemblyBudget, EvidenceAssembler
from backend.retrieval.candidate.candidate_generator import CandidateGenerator
from backend.retrieval.evaluation.evidence_evaluator import EvidenceEvaluator
from backend.retrieval.expansion.graph_expander import ExpansionBudget, GraphExpander
from backend.retrieval.interfaces.graph_retriever import GraphRetriever
from backend.retrieval.interfaces.vector_retriever import VectorRetriever
from backend.retrieval.providers.neo4j_retriever import Neo4jRetriever
from backend.retrieval.providers.qdrant_retriever import QdrantRetriever
from backend.retrieval.repository.retrieval_repository import RetrievalRepository
from backend.retrieval.services.retrieval_service import RetrievalService
from backend.retrieval.validation.retrieval_validator import RetrievalValidator
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


@lru_cache
def get_retrieval_storage() -> WorkspaceStorage:
    """Return the process-wide retrieval manifest storage backend.

    A seventh `WorkspaceStorage` instance, rooted at yet another directory --
    the same reused abstraction as every other artifact lifecycle.

    Returns:
        A `WorkspaceStorage` implementation configured from application settings.
    """
    settings = get_settings()
    return LocalFilesystemStorage(root=settings.retrieval_storage_root)


@lru_cache
def get_vector_retriever() -> VectorRetriever:
    """Return the process-wide, read-only vector retriever.

    A separate connection from Module 7's `VectorStore`, deliberately:
    depending on this narrower interface is what makes retrieval logic
    structurally incapable of writing to Qdrant.

    Returns:
        A `VectorRetriever` implementation configured from application settings.
    """
    settings = get_settings()
    return QdrantRetriever(url=settings.qdrant_url)


@lru_cache
def get_graph_retriever() -> GraphRetriever:
    """Return the process-wide, read-only graph retriever.

    A separate connection from Module 8's `KnowledgeGraphStore`,
    deliberately: depending on this narrower interface is what makes
    retrieval logic structurally incapable of writing to Neo4j.

    Returns:
        A `GraphRetriever` implementation configured from application settings.
    """
    settings = get_settings()
    return Neo4jRetriever(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )


def get_candidate_generator(
    text_provider: EmbeddingProvider = Depends(get_text_embedding_provider),
    vector_retriever: VectorRetriever = Depends(get_vector_retriever),
    settings: Settings = Depends(get_settings),
) -> CandidateGenerator:
    """Return a candidate generator wired to the configured embedding provider and retriever.

    Args:
        text_provider: Text embedding provider, injected.
        vector_retriever: Read-only vector retriever, injected.
        settings: Application settings, injected.

    Returns:
        A configured `CandidateGenerator`.
    """
    return CandidateGenerator(
        embedding_provider=text_provider,
        vector_retriever=vector_retriever,
        top_k=settings.retrieval_top_k,
    )


def get_graph_expander(
    graph_retriever: GraphRetriever = Depends(get_graph_retriever),
    vector_retriever: VectorRetriever = Depends(get_vector_retriever),
) -> GraphExpander:
    """Return a graph expander wired to the configured retrievers.

    Args:
        graph_retriever: Read-only graph retriever, injected.
        vector_retriever: Read-only vector retriever, injected.

    Returns:
        A configured `GraphExpander`.
    """
    return GraphExpander(graph_retriever=graph_retriever, vector_retriever=vector_retriever)


@lru_cache
def get_evidence_evaluator() -> EvidenceEvaluator:
    """Return the process-wide evidence evaluator.

    Returns:
        An `EvidenceEvaluator` instance. Stateless, so a single shared instance is safe.
    """
    return EvidenceEvaluator()


@lru_cache
def get_evidence_assembler() -> EvidenceAssembler:
    """Return the process-wide evidence assembler.

    Returns:
        An `EvidenceAssembler` instance. Stateless, so a single shared instance is safe.
    """
    return EvidenceAssembler()


@lru_cache
def get_retrieval_validator() -> RetrievalValidator:
    """Return the process-wide retrieval validator.

    Returns:
        A `RetrievalValidator` instance. Stateless, so a single shared instance is safe.
    """
    return RetrievalValidator()


def get_retrieval_repository(
    embeddings_storage: WorkspaceStorage = Depends(get_embeddings_storage),
    index_storage: WorkspaceStorage = Depends(get_index_storage),
    graph_storage: WorkspaceStorage = Depends(get_graph_storage),
    retrieval_storage: WorkspaceStorage = Depends(get_retrieval_storage),
) -> RetrievalRepository:
    """Return a retrieval repository wired to the configured storage backends.

    Args:
        embeddings_storage: Storage backend holding embedding manifests, injected.
        index_storage: Storage backend holding index manifests, injected.
        graph_storage: Storage backend holding graph manifests, injected.
        retrieval_storage: Storage backend for retrieval manifests, injected.

    Returns:
        A configured `RetrievalRepository`.
    """
    return RetrievalRepository(
        embeddings_storage=embeddings_storage,
        index_storage=index_storage,
        graph_storage=graph_storage,
        retrieval_storage=retrieval_storage,
    )


def get_retrieval_service(
    repository: RetrievalRepository = Depends(get_retrieval_repository),
    candidate_generator: CandidateGenerator = Depends(get_candidate_generator),
    graph_expander: GraphExpander = Depends(get_graph_expander),
    evaluator: EvidenceEvaluator = Depends(get_evidence_evaluator),
    assembler: EvidenceAssembler = Depends(get_evidence_assembler),
    validator: RetrievalValidator = Depends(get_retrieval_validator),
    settings: Settings = Depends(get_settings),
) -> RetrievalService:
    """Return a retrieval service wired to the configured phases and budgets.

    Args:
        repository: Retrieval repository, injected.
        candidate_generator: Phase 1, injected.
        graph_expander: Phase 2, injected.
        evaluator: Phase 3, injected.
        assembler: Phase 4, injected.
        validator: Structural validator, injected.
        settings: Application settings, injected.

    Returns:
        A configured `RetrievalService`.
    """
    return RetrievalService(
        repository=repository,
        candidate_generator=candidate_generator,
        graph_expander=graph_expander,
        evaluator=evaluator,
        assembler=assembler,
        validator=validator,
        expansion_budget=ExpansionBudget(
            max_depth=settings.retrieval_max_expansion_depth,
            max_neighbors_per_node=settings.retrieval_max_neighbors_per_node,
            max_total_evidence=settings.retrieval_max_total_evidence,
            max_traversal_cost=settings.retrieval_max_traversal_cost,
        ),
        assembly_budget=AssemblyBudget(
            max_evidence_groups=settings.retrieval_max_evidence_groups,
            max_primaries_per_section=settings.retrieval_max_primaries_per_section,
        ),
    )


def get_dense_retrieval_service(
    repository: RetrievalRepository = Depends(get_retrieval_repository),
    candidate_generator: CandidateGenerator = Depends(get_candidate_generator),
    graph_expander: GraphExpander = Depends(get_graph_expander),
    evaluator: EvidenceEvaluator = Depends(get_evidence_evaluator),
    assembler: EvidenceAssembler = Depends(get_evidence_assembler),
    validator: RetrievalValidator = Depends(get_retrieval_validator),
    settings: Settings = Depends(get_settings),
) -> RetrievalService:
    """Return a dense-only retrieval service, for evaluation comparison.

    The same phases and assembly budget as `get_retrieval_service`, but
    with a zero-depth expansion budget -- confirmed by real testing that
    Module 9's own expansion loop discovers nothing new at `max_depth=0`,
    so this is genuine dense-only retrieval, not a special code path.

    Args:
        repository: Retrieval repository, injected.
        candidate_generator: Phase 1, injected.
        graph_expander: Phase 2, injected.
        evaluator: Phase 3, injected.
        assembler: Phase 4, injected.
        validator: Structural validator, injected.
        settings: Application settings, injected.

    Returns:
        A `RetrievalService` configured for dense-only retrieval.
    """
    return RetrievalService(
        repository=repository,
        candidate_generator=candidate_generator,
        graph_expander=graph_expander,
        evaluator=evaluator,
        assembler=assembler,
        validator=validator,
        expansion_budget=ExpansionBudget(
            max_depth=0,
            max_neighbors_per_node=0,
            max_total_evidence=0,
            max_traversal_cost=0,
        ),
        assembly_budget=AssemblyBudget(
            max_evidence_groups=settings.retrieval_max_evidence_groups,
            max_primaries_per_section=settings.retrieval_max_primaries_per_section,
        ),
    )


@lru_cache
def get_generation_storage() -> WorkspaceStorage:
    """Return the process-wide generation manifest storage backend.

    An eighth `WorkspaceStorage` instance, rooted at yet another directory --
    the same reused abstraction as every other artifact lifecycle.

    Returns:
        A `WorkspaceStorage` implementation configured from application settings.
    """
    settings = get_settings()
    return LocalFilesystemStorage(root=settings.generation_storage_root)


@lru_cache
def get_generation_provider() -> GenerationProvider:
    """Return the process-wide generation provider.

    Cached because constructing an `OllamaProvider` opens a client
    connection -- reused across requests rather than reconnecting per call.

    Returns:
        A `GenerationProvider` implementation configured from application settings.
    """
    settings = get_settings()
    return OllamaProvider(host=settings.ollama_host)


def get_generation_config(settings: Settings = Depends(get_settings)) -> GenerationConfig:
    """Return the process-wide generation configuration.

    Args:
        settings: Application settings, injected.

    Returns:
        A `GenerationConfig` built from application settings -- never a
        hardcoded model or provider.
    """
    return GenerationConfig(
        provider=settings.generation_provider,
        model=settings.generation_model,
        temperature=settings.generation_temperature,
        top_p=settings.generation_top_p,
        max_tokens=settings.generation_max_tokens,
        context_window=settings.generation_context_window,
    )


@lru_cache
def get_answer_planner() -> AnswerPlanner:
    """Return the process-wide answer planner.

    Returns:
        An `AnswerPlanner` instance. Stateless, so a single shared instance is safe.
    """
    return AnswerPlanner()


@lru_cache
def get_context_optimizer() -> ContextOptimizer:
    """Return the process-wide context optimizer.

    Returns:
        A `ContextOptimizer` instance. Stateless, so a single shared instance is safe.
    """
    return ContextOptimizer()


@lru_cache
def get_prompt_composer() -> PromptComposer:
    """Return the process-wide prompt composer.

    Returns:
        A `PromptComposer` instance. Stateless, so a single shared instance is safe.
    """
    return PromptComposer()


@lru_cache
def get_prompt_validator() -> PromptValidator:
    """Return the process-wide prompt validator.

    Returns:
        A `PromptValidator` instance. Stateless, so a single shared instance is safe.
    """
    return PromptValidator()


@lru_cache
def get_grounding_validator() -> GroundingValidator:
    """Return the process-wide grounding validator.

    Returns:
        A `GroundingValidator` instance. Stateless, so a single shared instance is safe.
    """
    return GroundingValidator()


@lru_cache
def get_citation_resolver() -> CitationResolver:
    """Return the process-wide citation resolver.

    Returns:
        A `CitationResolver` instance. Stateless, so a single shared instance is safe.
    """
    return CitationResolver()


@lru_cache
def get_answer_quality_assessor() -> AnswerQualityAssessor:
    """Return the process-wide answer quality assessor.

    Returns:
        An `AnswerQualityAssessor` instance. Stateless, so a single shared instance is safe.
    """
    return AnswerQualityAssessor()


@lru_cache
def get_response_formatter() -> ResponseFormatter:
    """Return the process-wide response formatter.

    Returns:
        A `ResponseFormatter` instance. Stateless, so a single shared instance is safe.
    """
    return ResponseFormatter()


@lru_cache
def get_generation_validator() -> GenerationValidator:
    """Return the process-wide generation validator.

    Returns:
        A `GenerationValidator` instance. Stateless, so a single shared instance is safe.
    """
    return GenerationValidator()


def get_generation_repository(
    generation_storage: WorkspaceStorage = Depends(get_generation_storage),
) -> GenerationRepository:
    """Return a generation repository wired to the configured storage backend.

    Args:
        generation_storage: Storage backend for generation manifests, injected.

    Returns:
        A configured `GenerationRepository`.
    """
    return GenerationRepository(generation_storage=generation_storage)


def get_figure_analyst(
    parsed_storage: WorkspaceStorage = Depends(get_parsed_storage),
) -> FigureAnalyst:
    """Return the figure analyst wired to the local Ollama vision model.

    Args:
        parsed_storage: Storage the parser wrote figure images to, injected.

    Returns:
        A configured `FigureAnalyst` (a no-op when disabled in settings).
    """
    settings = get_settings()

    def describe(image: bytes, instruction: str) -> str:
        client = OllamaClient(host=settings.ollama_host)
        response = client.chat(
            model=settings.generation_vision_model,
            messages=[{"role": "user", "content": instruction, "images": [image]}],
            options={"temperature": 0.1, "num_predict": 400},
        )
        return response["message"]["content"]

    return FigureAnalyst(
        parsed_storage=parsed_storage,
        describe=describe,
        enabled=settings.generation_vision_enabled,
    )


def get_generation_service(
    repository: GenerationRepository = Depends(get_generation_repository),
    provider: GenerationProvider = Depends(get_generation_provider),
    planner: AnswerPlanner = Depends(get_answer_planner),
    context_optimizer: ContextOptimizer = Depends(get_context_optimizer),
    prompt_composer: PromptComposer = Depends(get_prompt_composer),
    prompt_validator: PromptValidator = Depends(get_prompt_validator),
    grounding_validator: GroundingValidator = Depends(get_grounding_validator),
    citation_resolver: CitationResolver = Depends(get_citation_resolver),
    quality_assessor: AnswerQualityAssessor = Depends(get_answer_quality_assessor),
    response_formatter: ResponseFormatter = Depends(get_response_formatter),
    generation_validator: GenerationValidator = Depends(get_generation_validator),
    figure_analyst: FigureAnalyst = Depends(get_figure_analyst),
) -> GenerationService:
    """Return a generation service wired to the configured phases.

    Args:
        repository: Generation repository, injected.
        provider: Generation provider, injected.
        planner: Phase 2, injected.
        context_optimizer: Phase 3, injected.
        prompt_composer: Phase 4, injected.
        prompt_validator: Phase 5, injected.
        grounding_validator: Phase 7, injected.
        citation_resolver: Phase 8, injected.
        quality_assessor: Phase 9, injected.
        response_formatter: Phase 10, injected.
        generation_validator: Whole-response structural validator, injected.

    Returns:
        A configured `GenerationService`.
    """
    return GenerationService(
        repository=repository,
        provider=provider,
        planner=planner,
        context_optimizer=context_optimizer,
        prompt_composer=prompt_composer,
        prompt_validator=prompt_validator,
        grounding_validator=grounding_validator,
        citation_resolver=citation_resolver,
        quality_assessor=quality_assessor,
        response_formatter=response_formatter,
        generation_validator=generation_validator,
        figure_analyst=figure_analyst,
    )


def get_evaluation_repository(settings: Settings = Depends(get_settings)) -> EvaluationRepository:
    """Return an evaluation repository wired to the configured storage root.

    Args:
        settings: Application settings, injected.

    Returns:
        A configured `EvaluationRepository`.
    """
    return EvaluationRepository(storage_root=settings.evaluation_storage_root)


@lru_cache
def get_evaluation_validator() -> EvaluationValidator:
    """Return the process-wide evaluation dataset validator.

    Returns:
        An `EvaluationValidator` instance. Stateless, so a single shared instance is safe.
    """
    return EvaluationValidator()


def get_benchmark_runner(
    dense_retrieval_service: RetrievalService = Depends(get_dense_retrieval_service),
    hybrid_retrieval_service: RetrievalService = Depends(get_retrieval_service),
    generation_service: GenerationService = Depends(get_generation_service),
    generation_config: GenerationConfig = Depends(get_generation_config),
) -> BenchmarkRunner:
    """Return a benchmark runner wired to the real production services.

    Reuses the exact same `RetrievalService` and `GenerationService`
    instances (and configuration) a real caller would get -- evaluation
    measures what production actually does, never a separately tuned copy.

    Args:
        dense_retrieval_service: Zero-depth-expansion retrieval service, injected.
        hybrid_retrieval_service: The real, normally configured retrieval service, injected.
        generation_service: The real generation service, injected.
        generation_config: The real generation configuration, injected.

    Returns:
        A configured `BenchmarkRunner`.
    """
    return BenchmarkRunner(
        dense_retrieval_service=dense_retrieval_service,
        hybrid_retrieval_service=hybrid_retrieval_service,
        generation_service=generation_service,
        generation_config=generation_config,
    )


def get_evaluation_service(
    repository: EvaluationRepository = Depends(get_evaluation_repository),
    validator: EvaluationValidator = Depends(get_evaluation_validator),
    runner: BenchmarkRunner = Depends(get_benchmark_runner),
    settings: Settings = Depends(get_settings),
) -> EvaluationService:
    """Return an evaluation service wired to the configured dataset path.

    Args:
        repository: Evaluation repository, injected.
        validator: Evaluation dataset validator, injected.
        runner: Benchmark runner, injected.
        settings: Application settings, injected.

    Returns:
        A configured `EvaluationService`.
    """
    return EvaluationService(
        repository=repository,
        validator=validator,
        runner=runner,
        dataset_path=settings.evaluation_dataset_path,
    )
