"""Push-based indexing: Spring calls these after a product or review commit (PLAN.md §1.5)."""

import logging
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.indexing.indexer import Indexer, KbStats, delete_product_data, reindex_all
from app.indexing.kb import load_documents
from app.models import Product, Review
from app.security import require_service_token
from app.services import Services, get_services, require_indexer

log = logging.getLogger(__name__)

router = APIRouter(prefix="/index", tags=["indexing"], dependencies=[Depends(require_service_token)])

ServicesDep = Annotated[Services, Depends(get_services)]


async def refresh_generated(services: Services, product: Product, reviews: list[Review]) -> None:
    """Regenerate the overview / review summary if their inputs changed (no-op otherwise,
    since both compare a hash of their inputs before calling the LLM)."""
    if services.overviews is not None:
        try:
            await services.overviews.get_or_generate(product)
        except Exception:
            log.exception("overview refresh for product %s failed", product.id)
    if services.review_summaries is not None:
        try:
            await services.review_summaries.get_or_generate(product, reviews)
        except Exception:
            log.exception("review summary refresh for product %s failed", product.id)


async def _reindex_product_task(services: Services, product_id: int) -> None:
    try:
        product = await services.spring.get_product(product_id)
        if product is None:
            deleted = await delete_product_data(services.pool, product_id)
            log.info("product %s is gone from Spring: deleted %s chunks", product_id, deleted)
            return
        reviews = await services.spring.get_reviews(product_id)
        indexer = services.indexer
        if indexer is not None:
            log.info("indexed product %s: %s", product_id, await indexer.index_product(product, reviews))
        await refresh_generated(services, product, reviews)
    except Exception:
        # The nightly reconcile catches anything that fails here.
        log.exception("background re-index of product %s failed", product_id)


async def sync_kb(indexer: Indexer) -> KbStats | None:
    """Index kb/*.md. Skipped (not wiped) when no documents are found, e.g. a missing folder."""
    documents = load_documents()
    if not documents:
        log.warning("no knowledge base documents found; skipping the KB sync")
        return None
    stats = await indexer.index_kb(documents)
    log.info("knowledge base synced: %s", stats)
    return stats


async def _reindex_all_task(services: Services) -> None:
    try:
        indexer = require_indexer(services)
        await sync_kb(indexer)
        await reindex_all(services.spring, indexer)
    except Exception:
        log.exception("reconcile failed")
    finally:
        services.reconcile_running = False


@router.post("/products/{product_id}", status_code=status.HTTP_202_ACCEPTED)
async def index_product(product_id: int, services: ServicesDep, background: BackgroundTasks) -> dict:
    """Re-read the product and its reviews from Spring; update its chunks and AI content
    (or delete everything if the product is gone)."""
    if services.embedder is None and services.llm is None:
        require_indexer(services)  # raises a 503 naming the missing key
    background.add_task(_reindex_product_task, services, product_id)
    return {"status": "accepted", "product_id": product_id}


@router.delete("/products/{product_id}")
async def delete_product(product_id: int, services: ServicesDep) -> dict:
    deleted = await delete_product_data(services.pool, product_id)
    return {"status": "deleted", "product_id": product_id, "chunks": deleted}


@router.post("/reconcile", status_code=status.HTTP_202_ACCEPTED)
async def reconcile(services: ServicesDep, background: BackgroundTasks) -> dict:
    """Full re-index + prune; triggered nightly by a GitHub Actions schedule.
    Doesn't generate AI content: that happens on change or on first view."""
    require_indexer(services)
    if services.reconcile_running:
        raise HTTPException(status.HTTP_409_CONFLICT, "A reconcile is already running")
    services.reconcile_running = True
    background.add_task(_reindex_all_task, services)
    return {"status": "accepted"}


@router.post("/kb")
async def index_kb(services: ServicesDep) -> dict:
    """Re-read kb/*.md now (runs at startup and on reconcile too). Only edited sections are embedded."""
    stats = await sync_kb(require_indexer(services))
    if stats is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No knowledge base documents found")
    return {"status": "synced"} | asdict(stats)
