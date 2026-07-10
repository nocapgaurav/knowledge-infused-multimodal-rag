"""Exceptions raised by the grounded answer generation engine.

Notably absent: an exception for "some claims were unsupported" or "some
citations failed to resolve." Those are normal, expected outcomes this
module is explicitly designed to handle -- they lower `confidence` and can
produce `PARTIALLY_SUFFICIENT_EVIDENCE`/`INSUFFICIENT_EVIDENCE`, never a
crash. Exceptions here are reserved for genuine structural defects: a
phase that could not produce its required output shape at all.
"""

from backend.domain import PaperId


class GenerationError(Exception):
    """Base class for all generation engine errors."""


class AnswerPlanningError(GenerationError):
    """Raised when Answer Planning cannot produce a plan for a question."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"answer planning failed: {reason}")


class ContextOptimizationError(GenerationError):
    """Raised when Context Optimization cannot produce any usable context."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"context optimization failed: {reason}")


class PromptCompositionError(GenerationError):
    """Raised when Prompt Composition cannot assemble a valid prompt."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"prompt composition failed: {reason}")


class PromptValidationError(GenerationError):
    """Base class for Prompt Validation failures. Fails loudly, never silently truncates."""


class TokenBudgetExceededError(PromptValidationError):
    """Raised when a composed prompt exceeds the configured context window."""

    def __init__(self, *, estimated_tokens: int, context_window: int) -> None:
        self.estimated_tokens = estimated_tokens
        self.context_window = context_window
        super().__init__(
            f"prompt requires an estimated {estimated_tokens} tokens, "
            f"exceeding the context window of {context_window}"
        )


class MissingEvidenceError(PromptValidationError):
    """Raised when a prompt intended to require evidence carries none."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"prompt is missing required evidence: {reason}")


class DuplicatePromptEvidenceError(PromptValidationError):
    """Raised when the same knowledge unit appears more than once in a prompt's context."""

    def __init__(self, *, knowledge_unit_id: str) -> None:
        self.knowledge_unit_id = knowledge_unit_id
        super().__init__(f"duplicate evidence in prompt: {knowledge_unit_id}")


class CitationPlaceholderError(PromptValidationError):
    """Raised when a citation label is malformed, duplicated, or unresolvable within the prompt."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"invalid citation placeholder: {reason}")


class ContextIntegrityError(PromptValidationError):
    """Raised when the prompt's structural components are internally inconsistent."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"prompt context integrity check failed: {reason}")


class GenerationProviderError(GenerationError):
    """Raised by a GenerationProvider when it fails to produce a response."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"generation provider failed: {reason}")


class NoClaimsExtractedError(GenerationError):
    """Raised when Grounding Validation cannot extract any claim from a generated answer.

    A structural defect (an empty or unparseable answer), never raised
    merely because claims failed their grounding check.
    """

    def __init__(self) -> None:
        super().__init__("no claims could be extracted from the generated answer")


class GenerationValidationError(GenerationError):
    """Base class for structural defects found in a freshly assembled GroundedResponse.

    Every subclass here is unreachable through correct phase
    implementations -- these check whole-response consistency, not
    per-claim grounding outcomes.
    """


class StatisticsMismatchError(GenerationValidationError):
    """Raised when the manifest's recorded statistics don't match the response's actual contents."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"generation statistics are inconsistent: {reason}")


class CitationConsistencyError(GenerationValidationError):
    """Raised when a resolved citation or supporting evidence item is internally inconsistent."""

    def __init__(self, *, reason: str) -> None:
        self.reason = reason
        super().__init__(f"citation data is inconsistent: {reason}")


class GenerationTraceIncompleteError(GenerationValidationError):
    """Raised when the generation trace does not cover every expected phase."""

    def __init__(self, *, missing_phases: list[str]) -> None:
        self.missing_phases = missing_phases
        super().__init__(f"generation trace is missing phases: {missing_phases}")


class GenerationStorageError(GenerationError):
    """Raised when a storage failure prevents a generation manifest from being persisted."""

    def __init__(self, *, document_id: PaperId) -> None:
        self.document_id = document_id
        super().__init__(
            f"a storage error occurred while persisting the generation manifest for "
            f"document {document_id}"
        )
