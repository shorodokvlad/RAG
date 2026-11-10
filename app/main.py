import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, index
from app.clients.embeddings import VoyageEmbedder
from app.clients.spring import SpringClient
from app.config import get_settings
from app.db import create_pool
from app.services import Services

log = logging.getLogger("shv-ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    pool = await create_pool(settings)
    spring = SpringClient(
        settings.spring_base_url, settings.service_token.get_secret_value(), settings.spring_timeout_seconds
    )
    embedder = None
    if settings.voyage_api_key is not None:
        embedder = VoyageEmbedder(
            settings.voyage_api_key.get_secret_value(),
            settings.embedding_model,
            settings.embedding_dim,
            settings.embedding_batch_size,
        )
    else:
        log.warning("VOYAGE_API_KEY is not set: indexing endpoints will answer 503")

    app.state.services = Services(pool=pool, spring=spring, embedder=embedder)
    try:
        yield
    finally:
        await spring.aclose()
        if embedder is not None:
            await embedder.aclose()
        await pool.close()


def create_app() -> FastAPI:
    app = FastAPI(title="SHV Assistant", version="0.1.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(index.router)
    return app


app = create_app()
