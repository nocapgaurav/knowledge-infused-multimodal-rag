"""Orchestrates the five retrieval phases.

Resolve collection and upstream versions -> generate candidates -> expand
through the graph -> evaluate and rank -> assemble evidence groups ->
validate at each boundary -> persist the manifest -> return the bundle.
Each phase is independently testable; this class's only job is calling
them in the right order and recording what happened.
"""

import logging
import time
from datetime import UTC, datetime

from backend.domain import PaperId
from backend.retrieval.assembly.evidence_assembler import AssemblyBudget, EvidenceAssembler
from backend.retrieval.candidate.candidate_generator import CandidateGenerator, normalize_query
from backend.retrieval.evaluation.evidence_evaluator import EvidenceEvaluator
from backend.retrieval.expansion.graph_expander import ExpansionBudget, GraphExpander
from backend.retrieval.models import (
    EvidenceBundle,
    PhaseTrace,
    RetrievalCandidate,
    RetrievalManifest,
    RetrievalStatistics,
    RetrievalTrace,
)
from backend.retrieval.repository.retrieval_repository import RetrievalRepository
from backend.retrieval.validation.retrieval_validator import RetrievalValidator

logger = logging.getLogger(__name__)

RETRIEVAL_ARTIFACT_VERSION = "1.0"
RETRIEVAL_STRATEGY_VERSION = "1.1"
"""Version of this module's own candidate generation, expansion,
evaluation, and assembly rules -- bumped when the retrieval strategy
itself changes, independently of the manifest schema or any upstream
artifact version."""


class RetrievalService:
    """Runs the complete hybrid evidence retrieval pipeline for one query."""

    def __init__(
        self,
        repository: RetrievalRepository,
        candidate_generator: CandidateGenerator,
        graph_expander: GraphExpander,
        evaluator: EvidenceEvaluator,
        assembler: EvidenceAssembler,
        validator: RetrievalValidator,
        expansion_budget: ExpansionBudget,
        assembly_budget: AssemblyBudget,
    ) -> None:
        """Initialize the service.

        Args:
            repository: Reads upstream manifests and persists retrieval manifests.
            candidate_generator: Phase 1.
            graph_expander: Phase 2.
            evaluator: Phase 3.
            assembler: Phase 4.
            validator: Structural checks run at every phase boundary.
            expansion_budget: Hard limits on Phase 2's traversal.
            assembly_budget: Hard limits on Phase 4's group count.
        """
        self._repository = repository
        self._candidate_generator = candidate_generator
        self._graph_expander = graph_expander
        self._evaluator = evaluator
        self._assembler = assembler
        self._validator = validator
        self._expansion_budget = expansion_budget
        self._assembly_budget = assembly_budget

    def retrieve(self, document_id: PaperId, query: str) -> EvidenceBundle:
        """Run the complete retrieval pipeline for one query against one document.

        Args:
            document_id: Identifier of the document to retrieve evidence from.
            query: The user's question, verbatim.

        Returns:
            The complete evidence bundle -- never an answer.

        Raises:
            DocumentNotIndexedError: The document has no embeddings/index.
            DocumentNotGraphedError: The document has no knowledge graph.
            QueryEmbeddingError: The query could not be embedded.
            VectorRetrieverError: A vector search or retrieval call failed.
            GraphRetrieverError: A graph traversal call failed.
            RetrievalValidationError: A structural check failed at some phase boundary.
            RetrievalStorageError: The manifest could not be persisted.
        """
        started = time.perf_counter()
        collection = self._repository.resolve_collection(document_id)
        embedding_manifest = self._repository.read_embedding_manifest(document_id)
        graph_manifest = self._repository.read_graph_manifest(document_id)

        phases: list[PhaseTrace] = []

        phase_started = time.perf_counter()
        seed_candidates = self._candidate_generator.generate(document_id, query, collection)
        phases.append(_phase_trace("candidate_generation", 0, len(seed_candidates), phase_started))

        phase_started = time.perf_counter()
        expansion_result = self._graph_expander.expand(
            seed_candidates, collection, self._expansion_budget
        )
        all_candidates: list[RetrievalCandidate] = [*seed_candidates, *expansion_result.candidates]
        self._validator.validate_candidates(all_candidates)
        for candidate in expansion_result.candidates:
            self._validator.validate_graph_path(candidate.graph_path)
        notes = (
            (f"budget_exhausted={expansion_result.budget_exhausted}",)
            if expansion_result.budget_exhausted
            else ()
        )
        phases.append(
            _phase_trace(
                "expansion", len(seed_candidates), len(all_candidates), phase_started, notes
            )
        )

        phase_started = time.perf_counter()
        scored_candidates = self._evaluator.evaluate(all_candidates, normalize_query(query))
        self._validator.validate_ranking(scored_candidates)
        phases.append(
            _phase_trace("evaluation", len(all_candidates), len(scored_candidates), phase_started)
        )

        phase_started = time.perf_counter()
        assembly_result = self._assembler.assemble(scored_candidates, self._assembly_budget)
        self._validator.validate_evidence_groups(assembly_result.groups, all_candidates)
        phases.append(
            _phase_trace(
                "assembly", len(scored_candidates), len(assembly_result.groups), phase_started
            )
        )

        trace = RetrievalTrace(phases=tuple(phases), dropped=tuple(assembly_result.dropped))
        evidence_items = sum(1 + len(group.supporting) for group in assembly_result.groups)
        manifest = RetrievalManifest(
            document_id=document_id,
            query=query,
            retrieval_version=RETRIEVAL_ARTIFACT_VERSION,
            retrieval_strategy_version=RETRIEVAL_STRATEGY_VERSION,
            representation_version=embedding_manifest.source_representation_version,
            embedding_version=embedding_manifest.model_version,
            graph_version=graph_manifest.graph_version,
            statistics=RetrievalStatistics(
                candidates_generated=len(seed_candidates),
                candidates_expanded=len(expansion_result.candidates),
                candidates_scored=len(scored_candidates),
                evidence_groups=len(assembly_result.groups),
                evidence_items=evidence_items,
                duration_ms=(time.perf_counter() - started) * 1000,
            ),
            created_at=datetime.now(UTC),
        )
        self._repository.save_retrieval_manifest(document_id, manifest)

        bundle = EvidenceBundle(
            document_id=document_id,
            query=query,
            candidates=tuple(all_candidates),
            evidence_groups=tuple(assembly_result.groups),
            trace=trace,
            manifest=manifest,
        )
        self._validator.validate_bundle(bundle)

        logger.info(
            "evidence retrieved",
            extra={
                "document_id": str(document_id),
                "candidates": len(all_candidates),
                "evidence_groups": len(assembly_result.groups),
            },
        )
        return bundle


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
