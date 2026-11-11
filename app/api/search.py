"""Catalog search endpoints (F5, F7 similar). Called by Spring's /ai gateway and the chat agent."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.features.search import MAX_RESULTS, ProductResult, SearchFilters
from app.retrieval.hybrid import KbPassage
from app.security import require_service_token
from app.services import Services, get_services, require_searcher

router = APIRouter(prefix="/search", tags=["search"], dependencies=[Depends(require_service_token)])

ServicesDep = Annotated[Services, Depends(get_services)]


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=200)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = Field(default=6, ge=1, le=MAX_RESULTS)


class SearchResponse(BaseModel):
    query: str
    items: list[ProductResult]


@router.post("")
async def search(body: SearchRequest, services: ServicesDep) -> SearchResponse:
    searcher = require_searcher(services)
    query = body.query.strip()
    return SearchResponse(query=query, items=await searcher.search(query, body.filters, body.limit))


@router.get("/similar/{product_id}")
async def similar(
    product_id: int, services: ServicesDep, limit: Annotated[int, Query(ge=1, le=MAX_RESULTS)] = 6
) -> list[ProductResult]:
    return await require_searcher(services).similar(product_id, limit)


class KbSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=200)
    limit: int = Field(default=4, ge=1, le=8)


@router.post("/kb")
async def search_kb(body: KbSearchRequest, services: ServicesDep) -> list[KbPassage]:
    """Store policies and FAQ passages (returns, delivery, payment, ...)."""
    return await require_searcher(services).hybrid.search_kb(body.query.strip(), body.limit)
