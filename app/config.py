from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Only what migrations need, so `alembic upgrade` works without the API keys."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Plain libpq URL, e.g. postgresql://postgres.<ref>:<password>@aws-0-eu-central-1.pooler.supabase.com:5432/postgres
    database_url: str


class Settings(DatabaseSettings):
    environment: str = "local"
    log_level: str = "INFO"

    # Supabase's pooler allows 15 connections in total and Spring uses up to 10 of them.
    db_pool_max: int = 3
    # 0 keeps asyncpg working behind PgBouncer in transaction mode (port 6543) too.
    db_statement_cache_size: int = 0

    # Shared secret: checked on our endpoints, and sent to Spring's /internal/ai/**.
    service_token: SecretStr
    spring_base_url: str = "http://localhost:2424"
    # Render free instances can take about a minute to wake up.
    spring_timeout_seconds: float = 90.0

    # Google AI Studio key (free tier): https://aistudio.google.com/apikey
    # Used for embeddings and, unless LLM_API_KEY is set, for the LLM too.
    gemini_api_key: SecretStr | None = None

    # LLM: any OpenAI-compatible chat API. Defaults to Gemini's free tier.
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-3.5-flash-lite"
    llm_api_key: SecretStr | None = None
    # Language of generated overviews, summaries and answers (the storefront is in English).
    content_language: str = "English"
    # Review summaries need at least this many reviews.
    review_summary_min_reviews: int = 3
    # Don't regenerate a review summary more often than this (bursts of new reviews).
    review_summary_min_interval_minutes: int = 10

    # Embeddings: "gemini" (free) or "voyage". Switching re-embeds everything on the next
    # reconcile, because the stored hash includes the model name.
    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 1024
    embedding_batch_size: int = 50
    voyage_api_key: SecretStr | None = None

    @property
    def effective_llm_api_key(self) -> SecretStr | None:
        return self.llm_api_key or self.gemini_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
