"""PromptContext: the structured, vendor-independent contents of one prompt.

Never a raw string and never a provider-specific message list -- each
component is its own field, independently testable and validatable.
Rendering this into a specific backend's wire format (a `messages` list
for Ollama's `/api/chat`, a single templated string for a raw-completion
backend) is the provider's job, not this module's.
"""

from pydantic import BaseModel, ConfigDict, Field

from backend.domain import ChunkModality
from backend.generation.models.answer_plan import AnswerPlan


class ContextSection(BaseModel):
    """One piece of optimized evidence, ready to be shown to the model.

    Attributes:
        citation_label: The short, deterministic label (e.g. "KU1") the
            model is instructed to cite inline -- never the raw
            `knowledge_unit_id` itself, which is long, easy to mistype,
            and gives a model nothing to gain from copying exactly.
            Citation Resolution (Phase 8) maps this label back to
            `knowledge_unit_id`.
        knowledge_unit_id: Identifier of the underlying knowledge unit,
            carried alongside the label so resolution never has to guess.
        text: The evidence's text content.
        modality: The kind of content this evidence represents.
        section_id: Identifier of the section this evidence belongs to, if any.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    citation_label: str = Field(min_length=1)
    knowledge_unit_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    modality: ChunkModality
    section_id: str | None = None


class PromptContext(BaseModel):
    """The complete, structured contents of one prompt, before rendering.

    Attributes:
        system_prompt: The model's role and identity statement.
        grounding_rules: Explicit, individually statable rules constraining
            the model to the provided evidence (e.g. "cite every factual
            claim using the given labels", "state explicitly when evidence
            is insufficient").
        answer_plan: The deterministic plan this prompt was built to satisfy.
        context_sections: The optimized evidence, in presentation order.
        formatting_rules: Explicit rules describing the expected output
            structure.
        user_question: The original question, verbatim.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    system_prompt: str = Field(min_length=1)
    grounding_rules: tuple[str, ...] = Field(min_length=1)
    answer_plan: AnswerPlan
    context_sections: tuple[ContextSection, ...]
    formatting_rules: tuple[str, ...] = Field(min_length=1)
    user_question: str = Field(min_length=1)
