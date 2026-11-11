"""Product-page AI endpoints (F2, F3, F4). Called by Spring's /ai gateway, never by browsers."""

import logging
from collections.abc import Awaitable
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.clients.llm import LLMError, LLMUnavailable
from app.features.ask import MAX_QUESTION_CHARS, MIN_QUESTION_CHARS, ProductAnswer
from app.features.overview import ProductOverview
from app.features.review_summary import ReviewSummary
from app.models import Product
from app.security import require_service_token
from app.services import (
    Services,
    get_services,
    require_asker,
    require_overviews,
    require_review_summaries,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/products", tags=["product ai"], dependencies=[Depends(require_service_token)])

ServicesDep = Annotated[Services, Depends(get_services)]


class OverviewResponse(BaseModel):
    product_id: int
    status: Literal["ready", "not_enough_information"]
    overview: ProductOverview | None = None
    model: str | None = None
    generated_at: datetime | None = None


class ReviewSummaryResponse(BaseModel):
    product_id: int
    status: Literal["ready", "not_enough_reviews"]
    min_reviews: int
    summary: ReviewSummary | None = None
    stale: bool = False
    model: str | None = None
    generated_at: datetime | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=MIN_QUESTION_CHARS, max_length=MAX_QUESTION_CHARS)


async def _product(services: Services, product_id: int) -> Product:
    product = await services.spring.get_product(product_id)
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Product not found")
    return product


async def _generate[T](work: Awaitable[T]) -> T:
    """Map LLM failures to HTTP: quota/outage → 503 (retry later), bad output → 502."""
    try:
        return await work
    except LLMUnavailable as exc:
        log.warning("LLM unavailable: %s", exc)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "The AI assistant is busy, try again soon"
        ) from exc
    except LLMError as exc:
        log.warning("LLM error: %s", exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "The AI assistant couldn't answer this time"
        ) from exc


@router.get("/{product_id}/overview")
async def overview(product_id: int, services: ServicesDep) -> OverviewResponse:
    overviews = require_overviews(services)
    product = await _product(services, product_id)
    record = await _generate(overviews.get_or_generate(product))
    if record is None:
        return OverviewResponse(product_id=product_id, status="not_enough_information")
    return OverviewResponse(
        product_id=product_id,
        status="ready",
        overview=record.overview,
        model=record.model,
        generated_at=record.generated_at,
    )


@router.get("/{product_id}/review-summary")
async def review_summary(
    product_id: int, services: ServicesDep, force: Annotated[bool, Query()] = False
) -> ReviewSummaryResponse:
    summaries = require_review_summaries(services)
    product = await _product(services, product_id)
    reviews = await services.spring.get_reviews(product_id)
    record = await _generate(summaries.get_or_generate(product, reviews, force=force))
    if record is None:
        return ReviewSummaryResponse(
            product_id=product_id, status="not_enough_reviews", min_reviews=summaries.min_reviews
        )
    return ReviewSummaryResponse(
        product_id=product_id,
        status="ready",
        min_reviews=summaries.min_reviews,
        summary=record.summary,
        stale=record.stale,
        model=record.model,
        generated_at=record.generated_at,
    )


@router.post("/{product_id}/ask")
async def ask(product_id: int, body: AskRequest, services: ServicesDep) -> ProductAnswer:
    asker = require_asker(services)
    product = await _product(services, product_id)
    return await _generate(asker.answer(product, body.question.strip()))
