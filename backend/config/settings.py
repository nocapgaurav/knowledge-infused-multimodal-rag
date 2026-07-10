"""Centralized application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the backend application.

    Values are loaded from environment variables prefixed with ``KIMRAG_``,
    optionally via a local ``.env`` file. Instances should be obtained
    through :func:`get_settings` rather than constructed directly, so that
    configuration is read once per process.

    Attributes:
        app_name: Human-readable application name, used in API metadata.
        app_version: Application version, used in API metadata.
        environment: Deployment environment the process is running in.
        host: Interface the HTTP server binds to.
        port: Port the HTTP server binds to.
        reload: Whether the server should reload on code changes.
        log_level: Minimum severity of log records emitted by the application.
        storage_root: Base directory document workspaces are created under.
        max_upload_size_bytes: Maximum permitted size for an uploaded document.
        parsed_storage_root: Base directory parsed document artifacts are
            written under.
        docling_ocr_enabled: Whether Docling runs OCR for scanned/image
            -based pages. Disabled by default: most scientific papers are
            born-digital with a real text layer, and enabling this pulls in
            an extra OCR model download in fresh environments.
        knowledge_storage_root: Base directory knowledge representation
            artifacts (knowledge units, relationships) are written under.
        max_words_per_knowledge_unit: Word-count threshold above which a
            paragraph is split at sentence boundaries into multiple
            knowledge units.
        min_words_per_knowledge_unit: Word-count floor below which a
            paragraph is merged into a neighbor rather than becoming its
            own knowledge unit.
        embeddings_storage_root: Base directory embedding artifacts are
            written under.
        embedding_model_name: HuggingFace identifier of the text embedding
            model.
        embedding_model_revision: Explicit commit revision to pin the
            embedding model to. If `None`, the current revision is
            resolved once at load time via the HuggingFace Hub API.
        embedding_batch_size: Number of knowledge units embedded per
            provider call.
        index_storage_root: Base directory index manifests are written under.
        qdrant_url: URL of the Qdrant HTTP API.
        qdrant_collection_prefix: Namespace prefix for collection names,
            for operational safety if a Qdrant instance is ever shared
            across deployments.
        index_batch_size: Number of vectors upserted per vector store call.
        graph_storage_root: Base directory graph manifests are written under.
        neo4j_uri: Bolt URI of the Neo4j instance.
        neo4j_user: Username to authenticate to Neo4j with.
        neo4j_password: Password to authenticate to Neo4j with.
        neo4j_database: Name of the Neo4j database to use.
        retrieval_storage_root: Base directory retrieval manifests are written under.
        retrieval_top_k: Maximum number of Phase 1 dense-retrieval candidates.
        retrieval_max_expansion_depth: Maximum graph traversal depth in
            Phase 2 (evidence expansion).
        retrieval_max_neighbors_per_node: Maximum neighbors explored from
            any single node during expansion.
        retrieval_max_total_evidence: Maximum new candidates expansion may
            discover, across the whole traversal.
        retrieval_max_traversal_cost: Maximum total neighbor-edges examined
            during expansion, before truncation.
        retrieval_max_evidence_groups: Maximum evidence groups in the final bundle.
        retrieval_max_primaries_per_section: Maximum evidence groups whose
            primary candidate comes from the same section.
        generation_storage_root: Base directory generation manifests are written under.
        ollama_host: Base URL of the Ollama server.
        generation_provider: Name of the generation backend.
        generation_model: Model identifier as the provider understands it.
            Development default (`qwen2.5:7b-instruct`) runs comfortably
            on Apple Silicon with 8GB RAM; a production deployment
            overrides this to a larger model without any code change.
        generation_temperature: Sampling temperature.
        generation_top_p: Nucleus sampling threshold.
        generation_max_tokens: Maximum tokens the provider may generate for the answer.
        generation_context_window: Maximum total tokens (prompt + completion) the model supports.
        evaluation_storage_root: Base directory benchmark runs are written under.
        evaluation_dataset_path: Path to the evaluation dataset JSON file.
        cors_allowed_origins: Origins the browser-based frontend (Module 12)
            is served from and is therefore allowed to call this API
            from. Cross-origin browser requests are rejected by the
            browser itself without this -- not an API behavior change,
            just the transport-level permission every separately-hosted
            frontend needs.
    """

    app_name: str = "Knowledge-Infused Multimodal RAG"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    storage_root: Path = Path("data/raw")
    max_upload_size_bytes: int = 50 * 1024 * 1024
    parsed_storage_root: Path = Path("data/parsed")
    docling_ocr_enabled: bool = False
    knowledge_storage_root: Path = Path("data/knowledge")
    max_words_per_knowledge_unit: int = 250
    min_words_per_knowledge_unit: int = 4
    embeddings_storage_root: Path = Path("data/embeddings")
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_model_revision: str | None = None
    embedding_batch_size: int = 32
    index_storage_root: Path = Path("data/index")
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_prefix: str = "kimrag"
    index_batch_size: int = 64
    graph_storage_root: Path = Path("data/graph")
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "kimrag-dev-password"
    neo4j_database: str = "neo4j"
    retrieval_storage_root: Path = Path("data/retrieval")
    retrieval_top_k: int = 20
    retrieval_max_expansion_depth: int = 2
    retrieval_max_neighbors_per_node: int = 10
    retrieval_max_total_evidence: int = 50
    retrieval_max_traversal_cost: int = 500
    retrieval_max_evidence_groups: int = 5
    retrieval_max_primaries_per_section: int = 2
    generation_storage_root: Path = Path("data/generation")
    ollama_host: str = "http://localhost:11434"
    generation_provider: str = "ollama"
    generation_model: str = "qwen2.5:7b-instruct"
    generation_temperature: float = 0.1
    generation_top_p: float = 0.9
    generation_max_tokens: int = 800
    generation_context_window: int = 4096
    evaluation_storage_root: Path = Path("data/evaluation")
    evaluation_dataset_path: Path = Path("data/evaluation_dataset.json")
    cors_allowed_origins: list[str] = ["http://localhost:3000"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="KIMRAG_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide, cached application settings.

    Returns:
        The single :class:`Settings` instance for this process.
    """
    return Settings()
