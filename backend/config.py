"""Configuration settings for the Ask Render Anything Assistant."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # extra="ignore": tolerate SDK-only env vars that live in the shared .env but aren't
    # Settings fields (e.g. RENDER_USE_LOCAL_DEV / RENDER_LOCAL_DEV_URL for local dev, and the
    # platform-injected RENDER_SDK_MODE / RENDER_SDK_SOCKET_PATH). Without this, pydantic-settings
    # defaults to extra="forbid" and crashes on startup when those keys appear in the .env file.
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")
    
    # API Keys
    openai_api_key: str
    anthropic_api_key: str
    logfire_token: str
    logfire_read_token: str = ""  # Optional: for fetching logs via API
    # Logfire Query API base URL. Use https://logfire-eu.pydantic.dev for EU-region projects.
    logfire_api_base: str = "https://logfire-us.pydantic.dev"
    
    # Database
    database_url: str

    # Render Workflows (gateway -> workflow service)
    render_api_key: str = ""  # Required to trigger/poll workflow runs
    workflow_slug: str = ""  # e.g. "pydantic-agents-pipeline" (from the Workflow's Dashboard page)
    
    # Pipeline Configuration
    max_tokens: int = 4000  # Answer generation budget; raised from 2000 so broad answers aren't truncated
    timeout_seconds: int = 30
    
    # RAG Configuration
    rag_top_k: int = 20  # Hard ceiling / backstop on retrieved docs, NOT a fixed quota.
    # Absolute floor: a doc is returned only if its cosine similarity >= this value,
    # so the result count varies with the question. The relevance gate lives in
    # hybrid_search; the adaptive cutoff below tightens it further per query.
    similarity_threshold: float = 0.3
    # Adaptive relative cutoff anchored to the best match: keep a doc only if its
    # cosine >= max(similarity_threshold, top_score * relevance_cutoff_fraction).
    # Strong topics filter aggressively; weaker-but-valid topics keep their cluster.
    relevance_cutoff_fraction: float = 0.75
    verification_threshold: float = 0.30  # Similarity threshold for claim verification (lowered to catch explicit facts)
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    
    # Model Selection
    answer_model: str = "claude-sonnet-4-6"
    claims_model: str = "gpt-5.4-mini"
    accuracy_model: str = "claude-sonnet-4-6"
    eval_model_openai: str = "gpt-5.4-mini"
    eval_model_anthropic: str = "claude-sonnet-4-6"
    query_expansion_model: str = "gpt-4.1-nano"
    
    # Performance
    enable_caching: bool = True
    log_level: str = "INFO"
    
    # CORS
    cors_origins: list[str] = ["*"]


class PipelineConfig:
    """Static pipeline configuration constants."""

    # Stage names for tracing
    STAGE_EMBEDDING = "question_embedding"
    STAGE_RETRIEVAL = "rag_retrieval"
    STAGE_GENERATION = "answer_generation"
    STAGE_CLAIMS = "claims_extraction"
    STAGE_VERIFICATION = "claims_verification"
    STAGE_ACCURACY = "technical_accuracy"
    STAGE_EVALUATION = "quality_evaluation"


# Global settings instance
settings = Settings()

