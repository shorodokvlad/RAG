from dataclasses import dataclass, field
from datetime import timedelta

import asyncpg
from fastapi import HTTPException, Request, status

from app.agent.loop import ChatAgent
from app.agent.memory import ChatMemory
from app.clients.embeddings import Embedder
from app.clients.llm import LLM
from app.clients.spring import SpringClient
from app.config import Settings
from app.features.ask import AskService
from app.features.overview import OverviewService
from app.features.review_summary import ReviewSummaryService
from app.features.search import SearchService
from app.indexing.indexer import Indexer
from app.providers import missing_embedder_key
from app.retrieval.hybrid import HybridSearch


@dataclass
class Services:
    """Long-lived clients and feature services, created once in the app lifespan."""

    pool: asyncpg.Pool
    spring: SpringClient
    embedder: Embedder | None
    llm: LLM | None
    settings: Settings | None = None
    # Set synchronously by the endpoint, so two quick calls can't both start a reconcile.
    reconcile_running: bool = False
    overviews: OverviewService | None = field(default=None, init=False)
    review_summaries: ReviewSummaryService | None = field(default=None, init=False)
    asker: AskService | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.llm is None or self.settings is None:
            return
        language = self.settings.content_language
        self.overviews = OverviewService(self.pool, self.llm, language)
        self.review_summaries = ReviewSummaryService(
            self.pool,
            self.llm,
            language,
            min_reviews=self.settings.review_summary_min_reviews,
            min_interval=timedelta(minutes=self.settings.review_summary_min_interval_minutes),
        )
        if self.embedder is not None:
            self.asker = AskService(self.pool, self.llm, self.embedder)

    @property
    def chat_agent(self) -> ChatAgent | None:
        return ChatAgent(self.llm, ChatMemory(self.pool), self) if self.llm is not None else None

    @property
    def searcher(self) -> SearchService | None:
        if self.embedder is None:
            return None
        return SearchService(HybridSearch(self.pool, self.embedder), self.spring)

    @property
    def indexer(self) -> Indexer | None:
        return Indexer(self.pool, self.embedder) if self.embedder is not None else None


def get_services(request: Request) -> Services:
    return request.app.state.services


def _unavailable(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail)


def require_indexer(services: Services) -> Indexer:
    indexer = services.indexer
    if indexer is None:
        key = missing_embedder_key(services.settings) if services.settings else "the embedding API key"
        raise _unavailable(f"Indexing is disabled: {key} is not set")
    return indexer


def require_overviews(services: Services) -> OverviewService:
    if services.overviews is None:
        raise _unavailable("AI generation is disabled: no LLM API key is set")
    return services.overviews


def require_review_summaries(services: Services) -> ReviewSummaryService:
    if services.review_summaries is None:
        raise _unavailable("AI generation is disabled: no LLM API key is set")
    return services.review_summaries


def require_asker(services: Services) -> AskService:
    if services.asker is None:
        raise _unavailable("Product Q&A is disabled: it needs both the LLM and embedding keys")
    return services.asker


def require_searcher(services: Services) -> SearchService:
    searcher = services.searcher
    if searcher is None:
        key = missing_embedder_key(services.settings) if services.settings else "the embedding API key"
        raise _unavailable(f"Search is disabled: {key} is not set")
    return searcher


def require_chat_agent(services: Services) -> ChatAgent:
    agent = services.chat_agent
    if agent is None:
        raise _unavailable("The chat assistant is disabled: no LLM API key is set")
    return agent
