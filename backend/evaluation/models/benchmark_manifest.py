"""BenchmarkManifest: describes one complete benchmark run, for reproducibility."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class BenchmarkManifest(BaseModel):
    """Describes one complete benchmark run.

    Persisted alongside every report -- the record two runs must agree on
    before a regression comparison between them means anything.

    Attributes:
        benchmark_id: Unique identifier for this run (used to reference it
            as a baseline or candidate in a regression comparison).
        benchmark_version: Schema version of this persisted manifest shape.
        evaluation_strategy_version: Version of this module's own
            metric-computation rules -- bumped when a metric's definition
            changes, independently of the manifest schema.
        dataset_version: Content hash of the evaluation dataset actually
            used -- two benchmark runs are only comparable if this matches.
        dataset_case_count: Number of cases in the dataset.
        retrieval_strategy_version: Module 9's construction-rules version,
            as recorded by the first evaluated case's retrieval result.
            `None` if the dataset was empty.
        generation_prompt_version: Module 10's prompt template version,
            as recorded by the first evaluated case's generation result.
            `None` if the dataset was empty.
        created_at: Timestamp this manifest was generated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark_id: str = Field(min_length=1)
    benchmark_version: str
    evaluation_strategy_version: str
    dataset_version: str
    dataset_case_count: int = Field(ge=0)
    retrieval_strategy_version: str | None
    generation_prompt_version: str | None
    created_at: datetime
