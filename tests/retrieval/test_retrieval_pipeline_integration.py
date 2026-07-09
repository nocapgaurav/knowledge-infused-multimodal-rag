"""The real, no-mock, full-pipeline integration test.

Runs the actual Module 3-8 pipeline (ingest, parse, represent, embed,
index, build graph) against the permanent fixture PDF, using real Docling
parsing, real BGE-M3 embeddings, a real Qdrant instance, and a real Neo4j
instance -- then runs Module 9's `RetrievalService` with real
`QdrantRetriever`/`Neo4jRetriever` providers against that real data. No
component in this file is a fake; every other test file's fakes exist
specifically so this is the only place the full, real stack has to run.
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
FIXTURE_PDF = Path(__file__).parent.parent / "parser" / "fixtures" / "sample_paper.pdf"
COLLECTION_PREFIX = "kimrag_retrieval_e2e"


@pytest.fixture(scope="module")
def real_pipeline(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict]:
    """Run the full Module 3-8 pipeline once, real services throughout."""
    root = tmp_path_factory.mktemp("retrieval-e2e")
    storages = {
        name: LocalFilesystemStorage(root=root / name)
        for name in ("raw", "parsed", "knowledge", "embeddings", "index", "graph", "retrieval")
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

    yield {
        "document_id": document_id,
        "storages": storages,
        "text_provider": text_provider,
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


def _real_service(real_pipeline: dict) -> RetrievalService:
    storages = real_pipeline["storages"]
    repository = RetrievalRepository(
        embeddings_storage=storages["embeddings"],
        index_storage=storages["index"],
        graph_storage=storages["graph"],
        retrieval_storage=storages["retrieval"],
    )
    vector_retriever = QdrantRetriever(url=QDRANT_URL)
    graph_retriever = Neo4jRetriever(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    return RetrievalService(
        repository=repository,
        candidate_generator=CandidateGenerator(
            real_pipeline["text_provider"], vector_retriever, top_k=10
        ),
        graph_expander=GraphExpander(graph_retriever, vector_retriever),
        evaluator=EvidenceEvaluator(),
        assembler=EvidenceAssembler(),
        validator=RetrievalValidator(),
        expansion_budget=ExpansionBudget(),
        assembly_budget=AssemblyBudget(),
    )


def test_real_pipeline_produces_a_valid_evidence_bundle(real_pipeline: dict) -> None:
    document_id: PaperId = real_pipeline["document_id"]
    service = _real_service(real_pipeline)

    bundle = service.retrieve(document_id, "What are the main results described in this paper?")

    assert bundle.document_id == document_id
    assert len(bundle.candidates) > 0
    assert bundle.evidence_groups
    assert bundle.manifest.statistics.candidates_generated > 0
    assert bundle.manifest.retrieval_strategy_version == "1.0"
    assert {phase.phase for phase in bundle.trace.phases} == {
        "candidate_generation",
        "expansion",
        "evaluation",
        "assembly",
    }


def test_real_pipeline_candidates_carry_real_content_and_provenance(real_pipeline: dict) -> None:
    document_id: PaperId = real_pipeline["document_id"]
    service = _real_service(real_pipeline)

    bundle = service.retrieve(document_id, "What methodology does this paper use?")

    for candidate in bundle.candidates:
        assert candidate.document_id == document_id
        assert candidate.text.strip() != ""
        if candidate.graph_path.depth == 0:
            assert candidate.dense_similarity is not None
        else:
            assert candidate.dense_similarity is None
            assert candidate.graph_path.hops[-1].target_id == str(candidate.knowledge_unit_id)


def test_real_pipeline_evidence_groups_have_no_duplicate_members(real_pipeline: dict) -> None:
    document_id: PaperId = real_pipeline["document_id"]
    service = _real_service(real_pipeline)

    bundle = service.retrieve(document_id, "What are the limitations of this study?")

    all_member_ids = [
        str(member.candidate.knowledge_unit_id)
        for group in bundle.evidence_groups
        for member in (group.primary, *group.supporting)
    ]
    assert len(all_member_ids) == len(set(all_member_ids))


def test_real_pipeline_manifest_persisted_with_real_upstream_versions(
    real_pipeline: dict, request: pytest.FixtureRequest
) -> None:
    document_id: PaperId = real_pipeline["document_id"]
    service = _real_service(real_pipeline)

    bundle = service.retrieve(document_id, "What data was used in this paper?")

    storages = real_pipeline["storages"]
    persisted = storages["retrieval"].read_json(document_id, "retrieval_manifest.json")
    assert persisted["representation_version"] == bundle.manifest.representation_version
    assert persisted["embedding_version"] == bundle.manifest.embedding_version
    assert persisted["graph_version"] == bundle.manifest.graph_version
    assert bundle.manifest.embedding_version == real_pipeline["text_provider"].model_version


def test_real_pipeline_retrieval_is_deterministic(real_pipeline: dict) -> None:
    document_id: PaperId = real_pipeline["document_id"]
    service = _real_service(real_pipeline)

    first = service.retrieve(document_id, "What are the main results described in this paper?")
    second = service.retrieve(document_id, "What are the main results described in this paper?")

    assert [c.knowledge_unit_id for c in first.candidates] == [
        c.knowledge_unit_id for c in second.candidates
    ]
    assert [c.dense_similarity for c in first.candidates] == [
        c.dense_similarity for c in second.candidates
    ]
    assert [g.group_id for g in first.evidence_groups] == [
        g.group_id for g in second.evidence_groups
    ]
