"""Shared base class for every domain model."""

from pydantic import BaseModel, ConfigDict


class DomainModel(BaseModel):
    """Base class for every domain model (entities and value objects).

    Domain models are immutable: they represent facts established once, at
    construction time (typically by the parser, the chunker, or the
    generation module), and are read by every downstream module. Making
    them immutable means a section handed to the chunking module cannot be
    silently altered in a way that the parser or the API layer never sees.

    `extra="forbid"` turns a misspelled field name into a validation error
    at construction time instead of a silently discarded value.

    Note:
        Immutability here means fields cannot be reassigned
        (`entity.field = x` raises). It does not deep-freeze mutable field
        values -- `entity.some_list.append(x)` is still possible. Enforcing
        that fully would require exposing tuples instead of lists, which
        would make this layer considerably less convenient to consume (and
        to serialize to JSON) for no real safety benefit at this stage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
