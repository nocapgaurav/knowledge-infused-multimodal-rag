"""Orchestrates the nine generation phases (Answer Planning through
Response Formatting) between the input EvidenceBundle and the output
GroundedResponse.

Consumes only the EvidenceBundle passed to it -- never calls Module 9's
retrieval service, never reads Qdrant or Neo4j, never regenerates
embeddings. Every version fact `AnswerProvenance` needs (retrieval,
representation, embedding, graph versions) is copied straight from
`bundle.manifest`, never recomputed.
"""

import hashlib
import json
import logging
import time
from datetime import UTC, datetime

from backend.generation.citations.citation_resolver import CitationResolver
from backend.generation.context.context_optimizer import ContextOptimizer
from backend.generation.formatting.response_formatter import FormattingInput, ResponseFormatter
from backend.generation.grounding.grounding_validator import GroundingValidator
from backend.generation.interfaces.generation_provider import GenerationProvider
from backend.generation.models.answer_plan import QuestionType
from backend.generation.models.answer_provenance import AnswerProvenance
from backend.generation.models.generation_config import GenerationConfig
from backend.generation.models.generation_manifest import GenerationManifest, GenerationStatistics
from backend.generation.models.generation_trace import GenerationTrace, PhaseTrace
from backend.generation.models.grounded_response import GroundedResponse
from backend.generation.models.grounding_report import ClaimGroundingStatus
from backend.generation.planner.answer_planner import AnswerPlanner
from backend.generation.prompt.prompt_composer import PROMPT_VERSION, PromptComposer
from backend.generation.prompt.prompt_validator import PromptValidator
from backend.generation.quality.answer_quality_assessor import AnswerQualityAssessor
from backend.generation.repository.generation_repository import GenerationRepository
from backend.generation.validation.generation_validator import GenerationValidator
from backend.generation.vision.figure_analyst import FigureAnalyst
from backend.retrieval.models import EvidenceBundle

logger = logging.getLogger(__name__)

GENERATION_ARTIFACT_VERSION = "1.0"
GENERATION_STRATEGY_VERSION = "1.0"
"""Version of this module's own planning/optimization/grounding/scoring
rules -- bumped when the strategy changes, independently of the manifest
schema or the prompt template version."""


class GenerationService:
    """Runs the complete grounded answer generation pipeline for one evidence bundle."""

    def __init__(
        self,
        repository: GenerationRepository,
        provider: GenerationProvider,
        planner: AnswerPlanner,
        context_optimizer: ContextOptimizer,
        prompt_composer: PromptComposer,
        prompt_validator: PromptValidator,
        grounding_validator: GroundingValidator,
        citation_resolver: CitationResolver,
        quality_assessor: AnswerQualityAssessor,
        response_formatter: ResponseFormatter,
        generation_validator: GenerationValidator,
        figure_analyst: "FigureAnalyst | None" = None,
    ) -> None:
        """Initialize the service.

        Args:
            repository: Persists generation manifests.
            provider: The LLM backend.
            planner: Phase 2.
            context_optimizer: Phase 3.
            prompt_composer: Phase 4.
            prompt_validator: Phase 5.
            grounding_validator: Phase 7.
            citation_resolver: Phase 8.
            quality_assessor: Phase 9.
            response_formatter: Phase 10.
            generation_validator: Whole-response structural checks, run last.
        """
        self._repository = repository
        self._provider = provider
        self._planner = planner
        self._context_optimizer = context_optimizer
        self._prompt_composer = prompt_composer
        self._prompt_validator = prompt_validator
        self._grounding_validator = grounding_validator
        self._citation_resolver = citation_resolver
        self._quality_assessor = quality_assessor
        self._response_formatter = response_formatter
        self._generation_validator = generation_validator
        self._figure_analyst = figure_analyst

    def generate(self, bundle: EvidenceBundle, config: GenerationConfig) -> GroundedResponse:
        """Generate a grounded answer for one evidence bundle.

        Args:
            bundle: The evidence retrieved for this question (Module 9's
                output) -- the sole input this module consumes.
            config: Generation parameters for this call.

        Returns:
            The complete, grounded, structurally-validated response.

        Raises:
            PromptValidationError: The composed prompt failed validation.
            GenerationProviderError: The provider failed to produce a response.
            NoClaimsExtractedError: The generated answer had no extractable claims.
            GenerationValidationError: The assembled response failed a
                whole-response consistency check.
            GenerationStorageError: The manifest could not be persisted.
        """
        started = time.perf_counter()
        phases: list[PhaseTrace] = []

        phase_started = time.perf_counter()
        plan = self._planner.plan(bundle.query, bundle)
        phases.append(_phase_trace("answer_planning", 1, 1, phase_started))

        phase_started = time.perf_counter()
        context_result = self._context_optimizer.optimize(bundle, plan, config)
        phases.append(
            _phase_trace(
                "context_optimization",
                context_result.total_candidates_considered,
                len(context_result.context_sections),
                phase_started,
                tuple(context_result.notes),
            )
        )

        context_sections = context_result.context_sections
        if self._figure_analyst is not None and plan.question_type is QuestionType.FIGURE_CENTRIC:
            phase_started = time.perf_counter()
            context_sections, vision_notes = self._figure_analyst.augment(
                bundle.document_id, context_sections, bundle
            )
            phases.append(
                _phase_trace(
                    "visual_analysis",
                    len(context_result.context_sections),
                    len(context_sections),
                    phase_started,
                    tuple(vision_notes),
                )
            )

        phase_started = time.perf_counter()
        prompt_context = self._prompt_composer.compose(bundle.query, plan, context_sections)
        phases.append(
            _phase_trace(
                "prompt_composition",
                len(context_result.context_sections),
                len(prompt_context.context_sections),
                phase_started,
            )
        )

        phase_started = time.perf_counter()
        self._prompt_validator.validate(prompt_context, config)
        phases.append(
            _phase_trace(
                "prompt_validation",
                len(prompt_context.context_sections),
                len(prompt_context.context_sections),
                phase_started,
            )
        )

        phase_started = time.perf_counter()
        provider_response = self._provider.generate(prompt_context, config)
        phases.append(_phase_trace("generation", 1, 1, phase_started))

        phase_started = time.perf_counter()
        grounding_report = self._grounding_validator.validate(
            provider_response.text, prompt_context
        )
        grounded_count = sum(
            1 for claim in grounding_report.claims if claim.status is ClaimGroundingStatus.GROUNDED
        )
        phases.append(
            _phase_trace(
                "grounding_validation", len(grounding_report.claims), grounded_count, phase_started
            )
        )

        phase_started = time.perf_counter()
        citation_report = self._citation_resolver.resolve(provider_response.text, prompt_context)
        phases.append(
            _phase_trace(
                "citation_resolution",
                len(citation_report.resolved) + len(citation_report.unresolved),
                len(citation_report.resolved),
                phase_started,
            )
        )

        phase_started = time.perf_counter()
        quality = self._quality_assessor.assess(
            bundle, plan, context_sections, grounding_report, citation_report
        )
        phases.append(_phase_trace("answer_quality_assessment", 1, 1, phase_started))

        model_version = self._provider.resolve_model_version(config.model)
        statistics = GenerationStatistics(
            context_sections_used=len(context_result.context_sections),
            context_sections_dropped=(
                context_result.total_candidates_considered - len(context_result.context_sections)
            ),
            claims_total=len(grounding_report.claims),
            claims_grounded=grounded_count,
            citations_resolved=len(citation_report.resolved),
            citations_unresolved=len(citation_report.unresolved),
            prompt_tokens=provider_response.prompt_tokens,
            completion_tokens=provider_response.completion_tokens,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        answer_provenance = AnswerProvenance(
            document_id=bundle.document_id,
            retrieval_version=bundle.manifest.retrieval_version,
            retrieval_strategy_version=bundle.manifest.retrieval_strategy_version,
            representation_version=bundle.manifest.representation_version,
            embedding_version=bundle.manifest.embedding_version,
            graph_version=bundle.manifest.graph_version,
            knowledge_unit_ids=tuple(
                citation.knowledge_unit_id for citation in citation_report.resolved
            ),
            evidence_bundle_checksum=_hash_bundle(bundle),
        )

        phase_started = time.perf_counter()
        formatting_input = FormattingInput(
            document_id=bundle.document_id,
            query=bundle.query,
            answer_text=provider_response.text,
            plan=plan,
            context_sections=context_sections,
            context_optimization_notes=context_result.notes,
            grounding_report=grounding_report,
            citation_report=citation_report,
            quality=quality,
            prompt_version=PROMPT_VERSION,
            model_name=config.model,
            model_version=model_version,
            generation_trace=GenerationTrace(phases=tuple(phases)),
            generation_statistics=statistics,
            answer_provenance=answer_provenance,
        )
        response = self._response_formatter.format(formatting_input, bundle)
        phases.append(
            _phase_trace(
                "response_formatting",
                len(context_sections),
                1,
                phase_started,
            )
        )

        final_trace = GenerationTrace(phases=tuple(phases))
        response = response.model_copy(update={"generation_trace": final_trace})

        self._generation_validator.validate(response, context_sections)

        manifest = GenerationManifest(
            document_id=bundle.document_id,
            query=bundle.query,
            generation_version=GENERATION_ARTIFACT_VERSION,
            prompt_version=PROMPT_VERSION,
            provider=self._provider.provider_name,
            model_name=config.model,
            model_version=model_version,
            answer_status=quality.answer_status,
            confidence=quality.confidence,
            statistics=statistics,
            created_at=datetime.now(UTC),
        )
        self._repository.save_generation_manifest(bundle.document_id, manifest)

        logger.info(
            "grounded response generated",
            extra={
                "document_id": str(bundle.document_id),
                "answer_status": quality.answer_status.value,
                "confidence": quality.confidence,
            },
        )
        return response


def _phase_trace(
    phase: str,
    input_count: int,
    output_count: int,
    started_at: float,
    notes: tuple[str, ...] = (),
) -> PhaseTrace:
    return PhaseTrace(
        phase=phase,
        input_count=input_count,
        output_count=output_count,
        duration_ms=(time.perf_counter() - started_at) * 1000,
        notes=notes,
    )


def _hash_bundle(bundle: EvidenceBundle) -> str:
    canonical = json.dumps(bundle.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
