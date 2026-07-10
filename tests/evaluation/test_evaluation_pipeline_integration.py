"""The real, no-mock, full-pipeline integration test for Module 11.

Runs the actual Module 3-10 pipeline (ingest, parse, represent, embed,
index, build graph, retrieve, generate) against the permanent fixture
PDF, using real Docling parsing, real BGE-M3 embeddings, a real Qdrant
instance, a real Neo4j instance, and a real Ollama model -- then builds a
one-case evaluation dataset referencing a real knowledge unit id
discovered from that real bundle, and runs it through the real
`EvaluationService`. No component in this file is a fake; every other
evaluation test file's fakes exist specifically so this is the only place
the full, real stack (through evaluation) has to run.
"""

import json
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
from backend.evaluation.benchmark.benchmark_runner import BenchmarkRunner
from backend.evaluation.repository.evaluation_repository import EvaluationRepository
from backend.evaluation.services.evaluation_service import EvaluationService
from backend.evaluation.validation.evaluation_validator import EvaluationValidator
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
COLLECTION_PREFIX = "kimrag_evaluation_e2e"
QUESTION = "What are the main results described in this paper?"


def _generation_config() -> GenerationConfig:
    return GenerationConfig(
        provider="ollama",
        model=GENERATION_MODEL,
        temperature=0.0,
        top_p=0.9,
        max_tokens=300,
        context_window=4096,
    )


@pytest.fixture(scope="module")
def real_pipeline(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict]:
    """Run the full Module 3-9 pipeline once, real services throughout, to
    obtain a real document id and a real knowledge unit id to build an
    evaluation dataset around."""
    root = tmp_path_factory.mktemp("evaluation-e2e")
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
    document_id: PaperId = job.document_id

    ParserService(
        raw_storage=storages["raw"],
        parsed_storage=storages["parsed"],
        document_parser=DoclingDocumentParser(ocr_enabled=False),
        mapper=DomainMapper(),
    ).parse_document(document_id)

    KnowledgeRepresentationService(
        parsed_storage=storages["parsed"],
        knowledge_storage=storages["knowledge"],
        builder=KnowledgeBuilder(max_words_per_chunk=250, min_words_per_chunk=4),
    ).represent_document(document_id)

    text_provider = SentenceTransformersProvider(model_name="BAAI/bge-m3")
    EmbeddingService(
        repository=EmbeddingRepository(
            knowledge_storage=storages["knowledge"], embeddings_storage=storages["embeddings"]
        ),
        text_provider=text_provider,
        planner=EmbeddingPlanner(),
        batch_size=32,
    ).embed_document(document_id)

    index_repository = IndexRepository(
        embeddings_storage=storages["embeddings"],
        knowledge_storage=storages["knowledge"],
        index_storage=storages["index"],
    )
    write_vector_store = QdrantProvider(url=QDRANT_URL)
    IndexingService(
        repository=index_repository,
        vector_store=write_vector_store,
        collection_prefix=COLLECTION_PREFIX,
        batch_size=64,
    ).index_document(document_id)

    write_graph_store = Neo4jProvider(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    GraphService(
        repository=GraphRepository(
            knowledge_storage=storages["knowledge"], graph_storage=storages["graph"]
        ),
        store=write_graph_store,
    ).build_graph(document_id)

    vector_retriever = QdrantRetriever(url=QDRANT_URL)
    graph_retriever = Neo4jRetriever(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    retrieval_repository = RetrievalRepository(
        embeddings_storage=storages["embeddings"],
        index_storage=storages["index"],
        graph_storage=storages["graph"],
        retrieval_storage=storages["retrieval"],
    )
    discovery_bundle = RetrievalService(
        repository=retrieval_repository,
        candidate_generator=CandidateGenerator(text_provider, vector_retriever, top_k=10),
        graph_expander=GraphExpander(graph_retriever, vector_retriever),
        evaluator=EvidenceEvaluator(),
        assembler=EvidenceAssembler(),
        validator=RetrievalValidator(),
        expansion_budget=ExpansionBudget(),
        assembly_budget=AssemblyBudget(),
    ).retrieve(document_id, QUESTION)
    real_knowledge_unit_id = str(
        discovery_bundle.evidence_groups[0].primary.candidate.knowledge_unit_id
    )

    yield {
        "document_id": document_id,
        "storages": storages,
        "evaluation_storage_root": root / "evaluation",
        "retrieval_repository": retrieval_repository,
        "vector_retriever": vector_retriever,
        "graph_retriever": graph_retriever,
        "text_provider": text_provider,
        "real_knowledge_unit_id": real_knowledge_unit_id,
    }

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


def _build_evaluation_service(real_pipeline: dict, dataset_path: Path) -> EvaluationService:
    storages = real_pipeline["storages"]
    vector_retriever = real_pipeline["vector_retriever"]
    graph_retriever = real_pipeline["graph_retriever"]

    def _retrieval_service(expansion_budget: ExpansionBudget) -> RetrievalService:
        return RetrievalService(
            repository=real_pipeline["retrieval_repository"],
            candidate_generator=CandidateGenerator(
                real_pipeline["text_provider"], vector_retriever, top_k=10
            ),
            graph_expander=GraphExpander(graph_retriever, vector_retriever),
            evaluator=EvidenceEvaluator(),
            assembler=EvidenceAssembler(),
            validator=RetrievalValidator(),
            expansion_budget=expansion_budget,
            assembly_budget=AssemblyBudget(),
        )

    generation_service = GenerationService(
        repository=GenerationRepository(generation_storage=storages["generation"]),
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
    runner = BenchmarkRunner(
        dense_retrieval_service=_retrieval_service(
            ExpansionBudget(
                max_depth=0, max_neighbors_per_node=0, max_total_evidence=0, max_traversal_cost=0
            )
        ),
        hybrid_retrieval_service=_retrieval_service(ExpansionBudget()),
        generation_service=generation_service,
        generation_config=_generation_config(),
    )
    return EvaluationService(
        repository=EvaluationRepository(storage_root=real_pipeline["evaluation_storage_root"]),
        validator=EvaluationValidator(),
        runner=runner,
        dataset_path=dataset_path,
    )


def _write_dataset(path: Path, *, document_id: PaperId, knowledge_unit_id: str) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "case_id": "case-001",
                    "question": QUESTION,
                    "document_id": str(document_id),
                    "ground_truth_answer": "A reference answer about the paper's main results.",
                    "expected_knowledge_units": [knowledge_unit_id],
                    "expected_citations": [knowledge_unit_id],
                    "expected_answer_status": "sufficient_evidence",
                    "difficulty": "easy",
                    "category": "factual",
                }
            ]
        ),
        encoding="utf-8",
    )


def test_real_pipeline_run_benchmark_produces_a_complete_summary(
    real_pipeline: dict, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(
        dataset_path,
        document_id=real_pipeline["document_id"],
        knowledge_unit_id=real_pipeline["real_knowledge_unit_id"],
    )
    service = _build_evaluation_service(real_pipeline, dataset_path)

    summary = service.run_benchmark()

    assert summary.manifest.dataset_case_count == 1
    assert len(summary.case_results) == 1
    assert not summary.failed_cases
    result = summary.case_results[0]
    assert result.hybrid_retrieval_metrics.recall_at_k[10] == 1.0
    assert 0.0 <= result.generation_metrics.grounding_accuracy <= 1.0
    assert result.answer_status in {
        AnswerStatus.SUFFICIENT_EVIDENCE,
        AnswerStatus.PARTIALLY_SUFFICIENT_EVIDENCE,
        AnswerStatus.INSUFFICIENT_EVIDENCE,
    }
    assert summary.manifest.retrieval_strategy_version is not None
    assert summary.manifest.generation_prompt_version is not None
    assert "mrr" in summary.hybrid_retrieval_aggregate
    assert "grounding_accuracy" in summary.generation_aggregate
    assert "end_to_end_latency_ms" in summary.performance_aggregate


def test_real_pipeline_get_latest_report_returns_the_persisted_summary(
    real_pipeline: dict, tmp_path: Path
) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(
        dataset_path,
        document_id=real_pipeline["document_id"],
        knowledge_unit_id=real_pipeline["real_knowledge_unit_id"],
    )
    service = _build_evaluation_service(real_pipeline, dataset_path)
    run_summary = service.run_benchmark()

    latest = service.get_latest_report()

    assert latest.manifest.benchmark_id == run_summary.manifest.benchmark_id
