from dataclasses import dataclass

import asyncpg
from fastapi import HTTPException, Request, status

from app.clients.embeddings import VoyageEmbedder
from app.clients.spring import SpringClient
from app.indexing.indexer import Indexer


@dataclass
class Services:
    """Long-lived clients, created once in the app lifespan."""

    pool: asyncpg.Pool
    spring: SpringClient
    embedder: VoyageEmbedder | None
    # Set synchronously by the endpoint, so two quick calls can't both start a reconcile.
    reconcile_running: bool = False

    @property
    def indexer(self) -> Indexer | None:
        return Indexer(self.pool, self.embedder) if self.embedder is not None else None


def get_services(request: Request) -> Services:
    return request.app.state.services


def require_indexer(services: Services) -> Indexer:
    indexer = services.indexer
    if indexer is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Indexing is disabled: VOYAGE_API_KEY is not set"
        )
    return indexer
