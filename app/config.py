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

    voyage_api_key: SecretStr | None = None
    embedding_model: str = "voyage-3.5"
    embedding_dim: int = 1024
    embedding_batch_size: int = 64


@lru_cache
def get_settings() -> Settings:
    return Settings()
