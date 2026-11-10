"""Push-based indexing: Spring calls these after a product or review commit (PLAN.md §1.5)."""

import logging
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.indexing.indexer import reindex_all, reindex_product
from app.security import require_service_token
from app.services import Services, get_services, require_indexer

log = logging.getLogger(__name__)

router = APIRouter(prefix="/index", tags=["indexing"], dependencies=[Depends(require_service_token)])

ServicesDep = Annotated[Services, Depends(get_services)]


async def _reindex_product_task(services: Services, product_id: int) -> None:
    try:
        await reindex_product(services.spring, require_indexer(services), product_id)
    except Exception:
        # The nightly reconcile catches anything that fails here.
        log.exception("background re-index of product %s failed", product_id)


async def _reindex_all_task(services: Services) -> None:
    try:
        await reindex_all(services.spring, require_indexer(services))
    except Exception:
        log.exception("reconcile failed")
    finally:
        services.reconcile_running = False


@router.post("/products/{product_id}", status_code=status.HTTP_202_ACCEPTED)
async def index_product(product_id: int, services: ServicesDep, background: BackgroundTasks) -> dict:
    """Re-read the product and its reviews from Spring and update its chunks (deletes them if it's gone)."""
    require_indexer(services)
    background.add_task(_reindex_product_task, services, product_id)
    return {"status": "accepted", "product_id": product_id}


@router.delete("/products/{product_id}")
async def delete_product(product_id: int, services: ServicesDep) -> dict:
    async with services.pool.acquire() as conn:
        result = await conn.execute("DELETE FROM ai.chunks WHERE product_id = $1", product_id)
    return {"status": "deleted", "product_id": product_id, "chunks": int(result.split()[-1])}


@router.post("/reconcile", status_code=status.HTTP_202_ACCEPTED)
async def reconcile(services: ServicesDep, background: BackgroundTasks) -> dict:
    """Full re-index + prune; triggered nightly by a GitHub Actions schedule."""
    require_indexer(services)
    if services.reconcile_running:
        raise HTTPException(status.HTTP_409_CONFLICT, "A reconcile is already running")
    services.reconcile_running = True
    background.add_task(_reindex_all_task, services)
    return {"status": "accepted"}
