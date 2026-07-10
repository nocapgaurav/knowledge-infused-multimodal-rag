"""The real, no-mock, full-pipeline integration test.

Runs the actual Module 3-9 pipeline (ingest, parse, represent, embed,
index, build graph, retrieve) against the permanent fixture PDF, using
real Docling parsing, real BGE-M3 embeddings, a real Qdrant instance, and
a real Neo4j instance to produce a real `EvidenceBundle` -- then runs
Module 10's `GenerationService` with the real `OllamaProvider` against
that real bundle. No component in this file is a fake; every other test
file's fakes exist specifically so this is the only place the full, real
stack (through generation) has to run.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from backend.chunking.builder.knowledge_builder import KnowledgeBuilder
from backend.chunking.services.knowledge_representation_service import (
    KnowledgeRepresentationService,
)
from backend.domain import PaperId
from backend.embeddings.planner.embedding_planner import EmbeddingPlanner
from backend.embeddings.providers.sentence_transformers_provider import (
    SentenceTransformersProvider,
)
from backend.embeddings.repository.embedding_repository import EmbeddingRepository
from backend.embeddings.services.embedding_service import EmbeddingService
from backend.generation.citations.citation_resolver import CitationResolver
from backend.generation.context.context_optimizer import ContextOptimizer
from backend.generation.formatting.response_formatter import ResponseFormatter
from backend.generation.grounding.grounding_validator import GroundingValidator
from backend.generation.models.answer_status import AnswerStatus
from backend.generation.models.generation_config import GenerationConfig
from backend.generation.planner.answer_planner import AnswerPlanner
from backend.generation.prompt.prompt_composer import PromptComposer
from backend.generation.prompt.prompt_validator import PromptValidator
from backend.generation.providers.ollama_provider import OllamaProvider
from backend.generation.quality.answer_quality_assessor import AnswerQualityAssessor
from backend.generation.repository.generation_repository import GenerationRepository
from backend.generation.services.generation_service import GenerationService
from backend.generation.validation.generation_validator import GenerationValidator
from backend.graph.providers.neo4j_provider import Neo4jProvider
from backend.graph.repository.graph_repository import GraphRepository
from backend.graph.services.graph_service import GraphService
from backend.ingestion.service import DocumentIngestionService
from backend.parser.mapper.domain_mapper import DomainMapper
from backend.parser.providers.docling_parser import DoclingDocumentParser
from backend.parser.services.parser_service import ParserService
from backend.retrieval.assembly.evidence_assembler import AssemblyBudget, EvidenceAssembler
from backend.retrieval.candidate.candidate_generator import CandidateGenerator
from backend.retrieval.evaluation.evidence_evaluator import EvidenceEvaluator
from backend.retrieval.expansion.graph_expander import ExpansionBudget, GraphExpander
from backend.retrieval.providers.neo4j_retriever import Neo4jRetriever
from backend.retrieval.providers.qdrant_retriever import QdrantRetriever
from backend.retrieval.repository.retrieval_repository import RetrievalRepository
from backend.retrieval.services.retrieval_service import RetrievalService
from backend.retrieval.validation.retrieval_validator import RetrievalValidator
from backend.search.providers.qdrant_provider import QdrantProvider
from backend.search.repository.index_repository import IndexRepository
from backend.search.services.indexing_service import IndexingService
from backend.storage.local_filesystem import LocalFilesystemStorage

QDRANT_URL = "http://localhost:6333"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "kimrag-dev-password"
OLLAMA_HOST = "http://localhost:11434"
GENERATION_MODEL = "qwen2.5:7b-instruct"
FIXTURE_PDF = Path(__file__).parent.parent / "parser" / "fixtures" / "sample_paper.pdf"
COLLECTION_PREFIX = "kimrag_generation_e2e"


@pytest.fixture(scope="module")
def real_bundle(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict]:
    """Run the full Module 3-9 pipeline once, real services throughout,
    producing a real EvidenceBundle for the generation tests to share."""
    root = tmp_path_factory.mktemp("generation-e2e")
    storages = {
        name: LocalFilesystemStorage(root=root / name)
        for name in (
            "raw",
            "parsed",
            "knowledge",
            "embeddings",
            "index",
            "graph",
            "retrieval",
            "generation",
        )
    }

    ingestion = DocumentIngestionService(
        storage=storages["raw"], max_upload_size_bytes=50 * 1024 * 1024
    )
    job = ingestion.ingest(
        filename="sample_paper.pdf",
        content_type="application/pdf",
        content=FIXTURE_PDF.read_bytes(),
    )
    document_id = job.document_id

    parser_service = ParserService(
        raw_storage=storages["raw"],
        parsed_storage=storages["parsed"],
        document_parser=DoclingDocumentParser(ocr_enabled=False),
        mapper=DomainMapper(),
    )
    parser_service.parse_document(document_id)

    representation_service = KnowledgeRepresentationService(
        parsed_storage=storages["parsed"],
        knowledge_storage=storages["knowledge"],
        builder=KnowledgeBuilder(max_words_per_chunk=250, min_words_per_chunk=4),
    )
    representation_service.represent_document(document_id)

    text_provider = SentenceTransformersProvider(model_name="BAAI/bge-m3")
    embedding_service = EmbeddingService(
        repository=EmbeddingRepository(
            knowledge_storage=storages["knowledge"], embeddings_storage=storages["embeddings"]
        ),
        text_provider=text_provider,
        planner=EmbeddingPlanner(),
        batch_size=32,
    )
    embedding_service.embed_document(document_id)

    index_repository = IndexRepository(
        embeddings_storage=storages["embeddings"],
        knowledge_storage=storages["knowledge"],
        index_storage=storages["index"],
    )
    write_vector_store = QdrantProvider(url=QDRANT_URL)
    indexing_service = IndexingService(
        repository=index_repository,
        vector_store=write_vector_store,
        collection_prefix=COLLECTION_PREFIX,
        batch_size=64,
    )
    indexing_service.index_document(document_id)

    write_graph_store = Neo4jProvider(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    graph_service = GraphService(
        repository=GraphRepository(
            knowledge_storage=storages["knowledge"], graph_storage=storages["graph"]
        ),
        store=write_graph_store,
    )
    graph_service.build_graph(document_id)

    retrieval_vector_retriever = QdrantRetriever(url=QDRANT_URL)
    retrieval_graph_retriever = Neo4jRetriever(
        uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD
    )
    retrieval_service = RetrievalService(
        repository=RetrievalRepository(
            embeddings_storage=storages["embeddings"],
            index_storage=storages["index"],
            graph_storage=storages["graph"],
            retrieval_storage=storages["retrieval"],
        ),
        candidate_generator=CandidateGenerator(text_provider, retrieval_vector_retriever, top_k=10),
        graph_expander=GraphExpander(retrieval_graph_retriever, retrieval_vector_retriever),
        evaluator=EvidenceEvaluator(),
        assembler=EvidenceAssembler(),
        validator=RetrievalValidator(),
        expansion_budget=ExpansionBudget(),
        assembly_budget=AssemblyBudget(),
    )
    bundle = retrieval_service.retrieve(
        document_id, "What are the main results described in this paper?"
    )

    yield {"document_id": document_id, "storages": storages, "bundle": bundle}

    index_manifest = index_repository.load_index_manifest(document_id)
    if index_manifest is not None and write_vector_store.collection_exists(
        index_manifest.collection_name
    ):
        write_vector_store._client.delete_collection(index_manifest.collection_name)  # test-only
    with write_graph_store._driver.session(database=write_graph_store._database) as session:
        session.run(
            "MATCH (n:KGNode {paper_id: $paper_id}) DETACH DELETE n", paper_id=str(document_id)
        )  # test-only cleanup
    write_graph_store.close()


def _config() -> GenerationConfig:
    return GenerationConfig(
        provider="ollama",
        model=GENERATION_MODEL,
        temperature=0.0,
        top_p=0.9,
        max_tokens=300,
        context_window=4096,
    )


def _real_service(real_bundle: dict) -> GenerationService:
    return GenerationService(
        repository=GenerationRepository(generation_storage=real_bundle["storages"]["generation"]),
        provider=OllamaProvider(host=OLLAMA_HOST),
        planner=AnswerPlanner(),
        context_optimizer=ContextOptimizer(),
        prompt_composer=PromptComposer(),
        prompt_validator=PromptValidator(),
        grounding_validator=GroundingValidator(),
        citation_resolver=CitationResolver(),
        quality_assessor=AnswerQualityAssessor(),
        response_formatter=ResponseFormatter(),
        generation_validator=GenerationValidator(),
    )


def test_real_pipeline_produces_a_valid_grounded_response(real_bundle: dict) -> None:
    document_id: PaperId = real_bundle["document_id"]
    service = _real_service(real_bundle)

    response = service.generate(real_bundle["bundle"], _config())

    assert response.document_id == document_id
    assert response.answer.strip() != ""
    assert response.executive_summary.strip() != ""
    assert response.answer_status in {
        AnswerStatus.SUFFICIENT_EVIDENCE,
        AnswerStatus.PARTIALLY_SUFFICIENT_EVIDENCE,
        AnswerStatus.INSUFFICIENT_EVIDENCE,
    }
    assert 0.0 <= response.confidence <= 1.0
    assert {phase.phase for phase in response.generation_trace.phases} == {
        "answer_planning",
        "context_optimization",
        "prompt_composition",
        "prompt_validation",
        "generation",
        "grounding_validation",
        "citation_resolution",
        "answer_quality_assessment",
        "response_formatting",
    }


def test_real_pipeline_every_resolved_citation_traces_to_real_evidence(
    real_bundle: dict,
) -> None:
    service = _real_service(real_bundle)
    bundle_knowledge_unit_ids = {
        str(candidate.knowledge_unit_id) for candidate in real_bundle["bundle"].candidates
    }

    response = service.generate(real_bundle["bundle"], _config())

    for citation in response.resolved_citations:
        assert citation.knowledge_unit_id in bundle_knowledge_unit_ids


def test_real_pipeline_provenance_matches_source_bundle(real_bundle: dict) -> None:
    service = _real_service(real_bundle)
    bundle = real_bundle["bundle"]

    response = service.generate(bundle, _config())

    assert response.answer_provenance.retrieval_version == bundle.manifest.retrieval_version
    assert response.answer_provenance.embedding_version == bundle.manifest.embedding_version
    assert response.answer_provenance.graph_version == bundle.manifest.graph_version
    assert response.answer_provenance.document_id == bundle.document_id


def test_real_pipeline_persists_generation_manifest(real_bundle: dict) -> None:
    document_id: PaperId = real_bundle["document_id"]
    service = _real_service(real_bundle)

    response = service.generate(real_bundle["bundle"], _config())

    persisted = real_bundle["storages"]["generation"].read_json(
        document_id, "generation_manifest.json"
    )
    assert persisted["model_name"] == GENERATION_MODEL
    assert persisted["answer_status"] == response.answer_status.value
    assert persisted["confidence"] == response.confidence


def test_real_pipeline_model_version_is_a_real_ollama_digest(real_bundle: dict) -> None:
    service = _real_service(real_bundle)

    response = service.generate(real_bundle["bundle"], _config())

    assert len(response.model_version) > 10
