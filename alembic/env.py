"""Alembic environment: raw-SQL migrations for the `ai` schema (no ORM models)."""

from logging.config import fileConfig

from sqlalchemy import create_engine, pool, text

from alembic import context
from app.config import DatabaseSettings

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

SCHEMA = "ai"


def sqlalchemy_url(database_url: str) -> str:
    """postgresql://… (what asyncpg and Supabase use) → postgresql+psycopg://… for SQLAlchemy."""
    for prefix in ("postgresql://", "postgres://"):
        if database_url.startswith(prefix):
            return "postgresql+psycopg://" + database_url[len(prefix) :]
    return database_url


def run_migrations_online() -> None:
    engine = create_engine(sqlalchemy_url(DatabaseSettings().database_url), poolclass=pool.NullPool)
    with engine.connect() as connection:
        # The version table lives in the ai schema, so the schema must exist first.
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        connection.commit()

        context.configure(connection=connection, target_metadata=None, version_table_schema=SCHEMA)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    raise SystemExit("Offline (--sql) migrations aren't supported; run against a database.")
run_migrations_online()
