import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, index, products, search
from app.api.index import sync_kb
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

    services = Services(pool=pool, spring=spring, embedder=embedder, llm=llm, settings=settings)
    app.state.services = services
    # Pick up knowledge-base edits shipped with this deploy (only changed sections are embedded).
    kb_task = asyncio.create_task(_startup_kb_sync(services)) if services.indexer is not None else None
    try:
        yield
    finally:
        if kb_task is not None:
            kb_task.cancel()
        await spring.aclose()
        if embedder is not None:
            await embedder.aclose()
        if llm is not None:
            await llm.aclose()
        await pool.close()


async def _startup_kb_sync(services: Services) -> None:
    try:
        await sync_kb(services.indexer)
    except Exception:
        log.exception("knowledge base sync at startup failed; the nightly reconcile retries it")


def create_app() -> FastAPI:
    app = FastAPI(title="SHV Assistant", version="0.2.0", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(index.router)
    app.include_router(products.router)
    app.include_router(search.router)
    return app


app = create_app()
