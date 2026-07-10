"""Grounded answer generation's own data models -- not part of
`backend.domain` for the same reason `EmbeddingArtifact`/`IndexManifest`/
`GraphManifest`/`RetrievalManifest` aren't: this is versioned,
strategy-dependent infrastructure output, not a permanent fact about a
paper. Deliberately expose no LLM provider type.
"""

from backend.generation.models.answer_plan import (
    AnswerPlan,
    AnswerSection,
    ExpectedAnswerType,
    QuestionType,
)
from backend.generation.models.answer_provenance import AnswerProvenance
from backend.generation.models.answer_status import AnswerStatus
from backend.generation.models.citation import (
    CitationResolutionReport,
    ResolvedCitation,
    UnresolvedCitation,
)
from backend.generation.models.generation_config import GenerationConfig
from backend.generation.models.generation_manifest import GenerationManifest, GenerationStatistics
from backend.generation.models.generation_trace import GenerationTrace, PhaseTrace
from backend.generation.models.grounded_response import GroundedResponse, SupportingEvidenceItem
from backend.generation.models.grounding_report import (
    ClaimGroundingStatus,
    ClaimVerdict,
    GroundingReport,
)
from backend.generation.models.prompt_context import ContextSection, PromptContext
from backend.generation.models.provider_response import ProviderResponse

__all__ = [
    "AnswerPlan",
    "AnswerProvenance",
    "AnswerSection",
    "AnswerStatus",
    "CitationResolutionReport",
    "ClaimGroundingStatus",
    "ClaimVerdict",
    "ContextSection",
    "ExpectedAnswerType",
    "GenerationConfig",
    "GenerationManifest",
    "GenerationStatistics",
    "GenerationTrace",
    "GroundedResponse",
    "GroundingReport",
    "PhaseTrace",
    "PromptContext",
    "ProviderResponse",
    "QuestionType",
    "ResolvedCitation",
    "SupportingEvidenceItem",
    "UnresolvedCitation",
]
