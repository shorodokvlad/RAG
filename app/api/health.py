from typing import Annotated

from fastapi import APIRouter, Depends

from app.security import require_service_token
from app.services import Services, get_services

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Liveness for Render's health check: no database call, so it stays cheap."""
    return {"status": "ok"}


@router.get("/health/ready", dependencies=[Depends(require_service_token)])
async def ready(services: Annotated[Services, Depends(get_services)]) -> dict:
    async with services.pool.acquire() as conn:
        chunks = await conn.fetchval("SELECT count(*) FROM ai.chunks")
        products = await conn.fetchval("SELECT count(DISTINCT product_id) FROM ai.chunks")
    return {
        "status": "ok",
        "database": "ok",
        "indexing_enabled": services.embedder is not None,
        "chunks": chunks,
        "indexed_products": products,
    }
