import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, index, products, search
from app.clients.spring import SpringClient
from app.config import get_settings
from app.db import create_pool
from app.providers import build_embedder, build_llm, missing_embedder_key
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
    embedder = build_embedder(settings)
    llm = build_llm(settings)
    if embedder is None:
        log.warning("%s is not set: indexing and Q&A answer 503", missing_embedder_key(settings))
    if llm is None:
        log.warning("No LLM key (GEMINI_API_KEY / LLM_API_KEY): overviews, summaries and Q&A answer 503")

    app.state.services = Services(pool=pool, spring=spring, embedder=embedder, llm=llm, settings=settings)
    try:
        yield
    finally:
        await spring.aclose()
        if embedder is not None:
            await embedder.aclose()
        if llm is not None:
            await llm.aclose()
        await pool.close()


def create_app() -> FastAPI:
    app = FastAPI(title="SHV Assistant", version="0.2.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(index.router)
    app.include_router(products.router)
    app.include_router(search.router)
    return app


app = create_app()
