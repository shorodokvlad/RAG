import asyncpg

from app.config import Settings


async def create_pool(settings: Settings) -> asyncpg.Pool:
    """Small, lazy pool: min_size=0 so an idle service holds no Supabase connections."""
    return await asyncpg.create_pool(
        dsn=settings.database_url,
        min_size=0,
        max_size=settings.db_pool_max,
        max_inactive_connection_lifetime=60,
        statement_cache_size=settings.db_statement_cache_size,
        command_timeout=60,
    )


def to_pgvector(values: list[float]) -> str:
    """Text form of a pgvector value; pass it with a `::vector` cast (no type codec needed)."""
    return "[" + ",".join(format(v, ".7g") for v in values) + "]"
