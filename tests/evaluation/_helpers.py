"""Shared factories for constructing minimal-but-real evaluation model
instances in tests, without threading every field through each call site.

Not a test module itself (no `test_` prefix) -- pytest never collects it.
"""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from backend.domain import PaperId
from backend.evaluation.models.benchmark_manifest import BenchmarkManifest
from backend.evaluation.models.benchmark_result import (
    BenchmarkResult,
    GenerationCaseMetrics,
    PerformanceCaseMetrics,
    RetrievalCaseMetrics,
)
from backend.evaluation.models.evaluation_case import Difficulty
from backend.evaluation.models.evaluation_summary import CaseFailure, EvaluationSummary
from backend.generation.models import (
    AnswerProvenance,
    AnswerStatus,
    GenerationStatistics,
    GenerationTrace,
    GroundedResponse,
    ResolvedCitation,
    SupportingEvidenceItem,
)
from backend.retrieval.models import (
    DiscoveryMethod,
    EvidenceBundle,
    EvidenceGroup,
    RankingExplanation,
    RetrievalCandidate,
    RetrievalManifest,
    RetrievalStatistics,
    RetrievalTrace,
    ScoredCandidate,
    SignalScore,
)

K_VALUES = (1, 3, 5, 10)


def build_manifest(
    *,
    benchmark_id: str = "benchmark-1",
    dataset_version: str = "dataset-hash-1",
    dataset_case_count: int = 1,
    retrieval_strategy_version: str | None = "1.0",
    generation_prompt_version: str | None = "1.0",
) -> BenchmarkManifest:
    return BenchmarkManifest(
        benchmark_id=benchmark_id,
        benchmark_version="1.0",
        evaluation_strategy_version="1.0",
        dataset_version=dataset_version,
        dataset_case_count=dataset_case_count,
        retrieval_strategy_version=retrieval_strategy_version,
        generation_prompt_version=generation_prompt_version,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def build_retrieval_case_metrics(
    *, reciprocal_rank: float = 1.0, value: float = 1.0
) -> RetrievalCaseMetrics:
    return RetrievalCaseMetrics(
        precision_at_k={k: value for k in K_VALUES},
        recall_at_k={k: value for k in K_VALUES},
        reciprocal_rank=reciprocal_rank,
        ndcg_at_k={k: value for k in K_VALUES},
        hit_rate_at_k={k: value for k in K_VALUES},
    )


def build_generation_case_metrics(
    *,
    grounding_accuracy: float = 1.0,
    citation_accuracy: float = 1.0,
    evidence_coverage: float = 1.0,
    answer_completeness: float = 1.0,
    unsupported_claim_rate: float = 0.0,
    answer_status_correct: bool = True,
) -> GenerationCaseMetrics:
    return GenerationCaseMetrics(
        grounding_accuracy=grounding_accuracy,
        citation_accuracy=citation_accuracy,
        evidence_coverage=evidence_coverage,
        answer_completeness=answer_completeness,
        unsupported_claim_rate=unsupported_claim_rate,
        answer_status_correct=answer_status_correct,
    )


def build_performance_case_metrics(
    *,
    retrieval_latency_ms: float = 100.0,
    generation_latency_ms: float = 900.0,
    end_to_end_latency_ms: float = 1000.0,
    peak_memory_mb: float = 50.0,
    cpu_time_ms: float = 200.0,
) -> PerformanceCaseMetrics:
    return PerformanceCaseMetrics(
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        end_to_end_latency_ms=end_to_end_latency_ms,
        peak_memory_mb=peak_memory_mb,
        cpu_time_ms=cpu_time_ms,
    )


def build_benchmark_result(
    *,
    case_id: str = "case-001",
    document_id: PaperId | None = None,
    difficulty: Difficulty = Difficulty.EASY,
    category: str = "factual",
    answer_status: AnswerStatus = AnswerStatus.SUFFICIENT_EVIDENCE,
    expected_answer_status: AnswerStatus = AnswerStatus.SUFFICIENT_EVIDENCE,
    generation_metrics: GenerationCaseMetrics | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        case_id=case_id,
        question="What are the main results?",
        document_id=document_id if document_id is not None else PaperId(uuid4()),
        difficulty=difficulty,
        category=category,
        dense_retrieval_metrics=build_retrieval_case_metrics(value=0.5),
        hybrid_retrieval_metrics=build_retrieval_case_metrics(value=1.0),
        generation_metrics=generation_metrics or build_generation_case_metrics(),
        performance_metrics=build_performance_case_metrics(),
        generated_answer="A generated answer.",
        ground_truth_answer="A ground truth answer.",
        answer_status=answer_status,
        expected_answer_status=expected_answer_status,
    )


def build_summary(
    *,
    benchmark_id: str = "benchmark-1",
    dataset_version: str = "dataset-hash-1",
    case_results: tuple[BenchmarkResult, ...] | None = None,
    failed_cases: tuple[CaseFailure, ...] = (),
    dense_retrieval_aggregate: dict[str, float] | None = None,
    hybrid_retrieval_aggregate: dict[str, float] | None = None,
    generation_aggregate: dict[str, float] | None = None,
    performance_aggregate: dict[str, float] | None = None,
    answer_status_accuracy: float = 1.0,
) -> EvaluationSummary:
    results = case_results if case_results is not None else (build_benchmark_result(),)
    return EvaluationSummary(
        manifest=build_manifest(
            benchmark_id=benchmark_id,
            dataset_version=dataset_version,
            dataset_case_count=len(results),
        ),
        case_results=results,
        failed_cases=failed_cases,
        dense_retrieval_aggregate=dense_retrieval_aggregate or {"mrr": 0.5, "precision@1": 0.5},
        hybrid_retrieval_aggregate=hybrid_retrieval_aggregate or {"mrr": 1.0, "precision@1": 1.0},
        generation_aggregate=generation_aggregate
        or {
            "grounding_accuracy": 1.0,
            "citation_accuracy": 1.0,
            "evidence_coverage": 1.0,
            "answer_completeness": 1.0,
            "unsupported_claim_rate": 0.0,
        },
        performance_aggregate=performance_aggregate
        or {
            "retrieval_latency_ms": 100.0,
            "generation_latency_ms": 900.0,
            "end_to_end_latency_ms": 1000.0,
            "peak_memory_mb": 50.0,
            "cpu_time_ms": 200.0,
            "throughput_cases_per_second": 1.0,
        },
        answer_status_accuracy=answer_status_accuracy,
        breakdown_by_difficulty={"easy": generation_aggregate or {"grounding_accuracy": 1.0}},
        breakdown_by_category={"factual": generation_aggregate or {"grounding_accuracy": 1.0}},
    )


def build_evidence_bundle(
    *,
    document_id: PaperId,
    query: str = "What are the main results?",
    knowledge_unit_ids: tuple[UUID, ...] | None = None,
    retrieval_strategy_version: str = "1.0",
) -> EvidenceBundle:
    """Build a real `EvidenceBundle`, one evidence group per knowledge unit
    id, in the given rank order -- exercises the same
    `_ranked_knowledge_unit_ids` flattening `BenchmarkRunner` relies on.

    `knowledge_unit_id` is a real `ChunkId` (a `UUID` NewType) in
    production, never a free-form string -- callers that need a specific,
    predictable id to assert against should generate one with `uuid4()`
    and pass its string form into `EvaluationCase.expected_knowledge_units`.
    """
    knowledge_unit_ids = knowledge_unit_ids or (uuid4(), uuid4())
    candidates = tuple(
        RetrievalCandidate(
            knowledge_unit_id=knowledge_unit_id,
            document_id=document_id,
            section_id=None,
            modality="text",
            text=f"evidence text for {knowledge_unit_id}",
            asset_uri=None,
            reading_order=index,
            citation_count=0,
            dense_similarity=1.0 - (index * 0.1),
            discovery_method=DiscoveryMethod.DENSE_RETRIEVAL,
        )
        for index, knowledge_unit_id in enumerate(knowledge_unit_ids)
    )
    groups = tuple(
        EvidenceGroup(
            group_id=str(candidate.knowledge_unit_id),
            primary=ScoredCandidate(
                candidate=candidate,
                ranking=RankingExplanation(
                    signals=(
                        SignalScore(
                            name="dense_similarity", raw_value=candidate.dense_similarity, rank=1
                        ),
                    ),
                    fused_score=candidate.dense_similarity,
                    final_rank=rank,
                ),
            ),
            supporting=(),
            modalities=("text",),
        )
        for rank, candidate in enumerate(candidates, start=1)
    )
    return EvidenceBundle(
        document_id=document_id,
        query=query,
        candidates=candidates,
        evidence_groups=groups,
        trace=RetrievalTrace(phases=(), dropped=()),
        manifest=RetrievalManifest(
            document_id=document_id,
            query=query,
            retrieval_version="1.0",
            retrieval_strategy_version=retrieval_strategy_version,
            representation_version="repr-hash",
            embedding_version="embed-hash",
            graph_version="1.0",
            statistics=RetrievalStatistics(
                candidates_generated=len(candidates),
                candidates_expanded=0,
                candidates_scored=len(candidates),
                evidence_groups=len(groups),
                evidence_items=len(candidates),
                duration_ms=5.0,
            ),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )


def build_grounded_response(
    *,
    bundle: EvidenceBundle,
    prompt_version: str = "1.0",
    cited_knowledge_unit_ids: tuple[str, ...] = (),
    claims_grounded: int = 1,
    claims_total: int = 1,
    citations_resolved: int = 1,
    citations_unresolved: int = 0,
    answer_status: AnswerStatus = AnswerStatus.SUFFICIENT_EVIDENCE,
) -> GroundedResponse:
    """Build a real `GroundedResponse` referencing the given cited
    knowledge unit ids -- every generation metric under test reads these
    fields exactly as Module 10 itself would populate them."""
    resolved_citations = tuple(
        ResolvedCitation(
            label=f"KU{index + 1}",
            knowledge_unit_id=knowledge_unit_id,
            text_excerpt=f"excerpt for {knowledge_unit_id}",
        )
        for index, knowledge_unit_id in enumerate(cited_knowledge_unit_ids)
    )
    return GroundedResponse(
        document_id=bundle.document_id,
        query=bundle.query,
        answer="A generated answer citing the evidence above.",
        executive_summary="A short summary of the generated answer.",
        supporting_evidence=tuple(
            SupportingEvidenceItem(
                label=f"KU{index + 1}",
                knowledge_unit_id=knowledge_unit_id,
                text=f"excerpt for {knowledge_unit_id}",
                modality="text",
            )
            for index, knowledge_unit_id in enumerate(cited_knowledge_unit_ids)
        ),
        resolved_citations=resolved_citations,
        limitations=(),
        references=tuple(f"[KU{i + 1}] excerpt" for i in range(len(cited_knowledge_unit_ids))),
        warnings=(),
        confidence=0.9,
        answer_status=answer_status,
        generation_metadata={},
        prompt_version=prompt_version,
        model_name="fake-model",
        model_version="fake-model-version-1",
        generation_trace=GenerationTrace(phases=()),
        generation_statistics=GenerationStatistics(
            context_sections_used=len(bundle.evidence_groups),
            context_sections_dropped=0,
            claims_total=claims_total,
            claims_grounded=claims_grounded,
            citations_resolved=citations_resolved,
            citations_unresolved=citations_unresolved,
            prompt_tokens=100,
            completion_tokens=50,
            duration_ms=5.0,
        ),
        answer_provenance=AnswerProvenance(
            document_id=bundle.document_id,
            retrieval_version=bundle.manifest.retrieval_version,
            retrieval_strategy_version=bundle.manifest.retrieval_strategy_version,
            representation_version=bundle.manifest.representation_version,
            embedding_version=bundle.manifest.embedding_version,
            graph_version=bundle.manifest.graph_version,
            knowledge_unit_ids=tuple(
                str(candidate.knowledge_unit_id) for candidate in bundle.candidates
            ),
            evidence_bundle_checksum="checksum",
        ),
    )
