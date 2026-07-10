"""Unit tests for `BenchmarkRunner`, run against fake retrieval and
generation services that return real, fully-populated `EvidenceBundle`/
`GroundedResponse` objects -- the same test-double pattern
`tests/api/test_generation.py` uses, exercising `BenchmarkRunner`'s own
metric computation and aggregation logic without touching Qdrant, Neo4j,
or Ollama. The real, no-fake, full-pipeline run lives in
`test_evaluation_pipeline_integration.py`.
"""

from uuid import uuid4

import pytest

from backend.domain import PaperId
from backend.evaluation.benchmark.benchmark_runner import BenchmarkRunner
from backend.evaluation.models.evaluation_case import Difficulty, EvaluationCase
from backend.generation.exceptions import GenerationProviderError
from backend.generation.models import AnswerStatus, GenerationConfig, GroundedResponse
from backend.generation.services.generation_service import GenerationService
from backend.retrieval.exceptions import DocumentNotIndexedError
from backend.retrieval.models import EvidenceBundle
from backend.retrieval.services.retrieval_service import RetrievalService

from ._helpers import build_evidence_bundle, build_grounded_response

DOCUMENT_ID = PaperId(uuid4())
KU_1, KU_2, KU_3 = uuid4(), uuid4(), uuid4()


def _config() -> GenerationConfig:
    return GenerationConfig(
        provider="ollama",
        model="fake-model",
        temperature=0.0,
        top_p=0.9,
        max_tokens=300,
        context_window=4096,
    )


def _case(
    *,
    case_id: str = "case-001",
    expected_knowledge_units: tuple[str, ...] = (str(KU_1), str(KU_2)),
    expected_citations: tuple[str, ...] = (str(KU_1),),
    expected_answer_status: AnswerStatus = AnswerStatus.SUFFICIENT_EVIDENCE,
    difficulty: Difficulty = Difficulty.EASY,
    category: str = "factual",
) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        question="What are the main results?",
        document_id=DOCUMENT_ID,
        ground_truth_answer="A reference answer.",
        expected_knowledge_units=expected_knowledge_units,
        expected_citations=expected_citations,
        expected_answer_status=expected_answer_status,
        difficulty=difficulty,
        category=category,
    )


class _FakeRetrievalService(RetrievalService):
    """A real `RetrievalService` subclass that overrides `retrieve` and
    deliberately skips the parent `__init__` (no Qdrant/Neo4j connection
    is ever constructed) -- a true `RetrievalService` for `BenchmarkRunner`
    to depend on, returning a fixed, real bundle instead of querying live
    infrastructure. Dense and hybrid instances can be configured to
    differ, exactly as `ExpansionBudget(max_depth=0)` makes the real
    dense-only service discover strictly fewer candidates."""

    def __init__(
        self,
        *,
        knowledge_unit_ids: tuple = (KU_1, KU_2),
        retrieval_strategy_version: str = "1.0",
        raise_not_indexed: bool = False,
    ) -> None:
        self._knowledge_unit_ids = knowledge_unit_ids
        self._retrieval_strategy_version = retrieval_strategy_version
        self._raise_not_indexed = raise_not_indexed

    def retrieve(self, document_id: PaperId, query: str) -> EvidenceBundle:
        if self._raise_not_indexed:
            raise DocumentNotIndexedError(document_id=document_id)
        return build_evidence_bundle(
            document_id=document_id,
            query=query,
            knowledge_unit_ids=self._knowledge_unit_ids,
            retrieval_strategy_version=self._retrieval_strategy_version,
        )


class _FakeGenerationService(GenerationService):
    """A real `GenerationService` subclass, for the same reason
    `_FakeRetrievalService` subclasses `RetrievalService` above."""

    def __init__(
        self,
        *,
        cited_knowledge_unit_ids: tuple[str, ...] = (str(KU_1),),
        claims_grounded: int = 4,
        claims_total: int = 5,
        answer_status: AnswerStatus = AnswerStatus.SUFFICIENT_EVIDENCE,
        prompt_version: str = "1.0",
        raise_provider_error: bool = False,
    ) -> None:
        self._cited_knowledge_unit_ids = cited_knowledge_unit_ids
        self._claims_grounded = claims_grounded
        self._claims_total = claims_total
        self._answer_status = answer_status
        self._prompt_version = prompt_version
        self._raise_provider_error = raise_provider_error

    def generate(self, bundle: EvidenceBundle, config: GenerationConfig) -> GroundedResponse:
        if self._raise_provider_error:
            raise GenerationProviderError(reason="simulated failure")
        return build_grounded_response(
            bundle=bundle,
            prompt_version=self._prompt_version,
            cited_knowledge_unit_ids=self._cited_knowledge_unit_ids,
            claims_grounded=self._claims_grounded,
            claims_total=self._claims_total,
            citations_resolved=len(self._cited_knowledge_unit_ids),
            citations_unresolved=0,
            answer_status=self._answer_status,
        )


def test_run_computes_retrieval_and_generation_metrics_for_a_perfect_case() -> None:
    runner = BenchmarkRunner(
        dense_retrieval_service=_FakeRetrievalService(knowledge_unit_ids=(KU_2,)),
        hybrid_retrieval_service=_FakeRetrievalService(knowledge_unit_ids=(KU_1, KU_2)),
        generation_service=_FakeGenerationService(
            cited_knowledge_unit_ids=(str(KU_1),), claims_grounded=5, claims_total=5
        ),
        generation_config=_config(),
    )

    result = runner.run([_case()])

    assert len(result.case_results) == 1
    assert not result.failed_cases
    case_result = result.case_results[0]
    assert case_result.hybrid_retrieval_metrics.recall_at_k[10] == 1.0
    assert case_result.dense_retrieval_metrics.recall_at_k[10] == pytest.approx(0.5)
    assert case_result.generation_metrics.grounding_accuracy == 1.0
    assert case_result.generation_metrics.evidence_coverage == 1.0
    assert case_result.generation_metrics.answer_completeness == 1.0
    assert case_result.generation_metrics.answer_status_correct is True
    assert result.answer_status_accuracy == 1.0
    assert result.retrieval_strategy_version == "1.0"
    assert result.generation_prompt_version == "1.0"
    assert result.total_duration_seconds > 0


def test_run_records_a_case_failure_without_aborting_the_run() -> None:
    runner = BenchmarkRunner(
        dense_retrieval_service=_FakeRetrievalService(),
        hybrid_retrieval_service=_FakeRetrievalService(raise_not_indexed=True),
        generation_service=_FakeGenerationService(),
        generation_config=_config(),
    )

    result = runner.run([_case(case_id="case-001"), _case(case_id="case-002")])

    assert result.case_results == []
    assert len(result.failed_cases) == 2
    assert {failure.case_id for failure in result.failed_cases} == {"case-001", "case-002"}


def test_run_excludes_failed_cases_from_every_aggregate() -> None:
    good_retrieval = _FakeRetrievalService()
    bad_generation = _FakeGenerationService(raise_provider_error=True)
    good_generation = _FakeGenerationService()

    class _MixedGenerationService(GenerationService):
        def __init__(self) -> None:
            self._calls = 0

        def generate(self, bundle: EvidenceBundle, config: GenerationConfig) -> GroundedResponse:
            self._calls += 1
            service = bad_generation if self._calls == 1 else good_generation
            return service.generate(bundle, config)

    runner = BenchmarkRunner(
        dense_retrieval_service=good_retrieval,
        hybrid_retrieval_service=good_retrieval,
        generation_service=_MixedGenerationService(),
        generation_config=_config(),
    )

    result = runner.run([_case(case_id="case-001"), _case(case_id="case-002")])

    assert len(result.case_results) == 1
    assert result.case_results[0].case_id == "case-002"
    assert len(result.failed_cases) == 1
    assert result.failed_cases[0].case_id == "case-001"


def test_run_of_an_empty_dataset_produces_empty_aggregates() -> None:
    runner = BenchmarkRunner(
        dense_retrieval_service=_FakeRetrievalService(),
        hybrid_retrieval_service=_FakeRetrievalService(),
        generation_service=_FakeGenerationService(),
        generation_config=_config(),
    )

    result = runner.run([])

    assert result.case_results == []
    assert result.dense_retrieval_aggregate == {}
    assert result.hybrid_retrieval_aggregate == {}
    assert result.generation_aggregate == {}
    assert result.performance_aggregate == {}
    assert result.answer_status_accuracy == 0.0
    assert result.retrieval_strategy_version is None
    assert result.generation_prompt_version is None


def test_run_breaks_down_metrics_by_difficulty_and_category() -> None:
    runner = BenchmarkRunner(
        dense_retrieval_service=_FakeRetrievalService(),
        hybrid_retrieval_service=_FakeRetrievalService(),
        generation_service=_FakeGenerationService(),
        generation_config=_config(),
    )

    result = runner.run(
        [
            _case(case_id="case-001", difficulty=Difficulty.EASY, category="factual"),
            _case(case_id="case-002", difficulty=Difficulty.HARD, category="comparative"),
        ]
    )

    assert set(result.breakdown_by_difficulty) == {"easy", "hard"}
    assert set(result.breakdown_by_category) == {"factual", "comparative"}


def test_run_flags_answer_status_mismatch() -> None:
    runner = BenchmarkRunner(
        dense_retrieval_service=_FakeRetrievalService(),
        hybrid_retrieval_service=_FakeRetrievalService(),
        generation_service=_FakeGenerationService(answer_status=AnswerStatus.INSUFFICIENT_EVIDENCE),
        generation_config=_config(),
    )

    result = runner.run([_case(expected_answer_status=AnswerStatus.SUFFICIENT_EVIDENCE)])

    assert result.case_results[0].generation_metrics.answer_status_correct is False
    assert result.answer_status_accuracy == 0.0
