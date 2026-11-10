"""Write chunks to ai.chunks, embedding only what changed."""

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import asyncpg

from app.clients.embeddings import Embedder
from app.clients.spring import SpringClient
from app.db import to_pgvector
from app.indexing.chunkers import Chunk, build_product_chunks
from app.models import Product, Review

log = logging.getLogger(__name__)

ChunkKey = tuple[str, int, int]


@dataclass(frozen=True)
class ExistingChunk:
    id: int
    key: ChunkKey
    content_hash: str


@dataclass
class ChangePlan:
    to_embed: list[Chunk] = field(default_factory=list)  # new or changed content
    to_touch: list[Chunk] = field(default_factory=list)  # same content: refresh metadata only
    to_delete: list[int] = field(default_factory=list)  # ai.chunks ids that no longer exist


def plan_changes(existing: Iterable[ExistingChunk], chunks: Sequence[Chunk]) -> ChangePlan:
    by_key = {row.key: row for row in existing}
    plan = ChangePlan()
    wanted = set()
    for chunk in chunks:
        wanted.add(chunk.key)
        row = by_key.get(chunk.key)
        if row is not None and row.content_hash == chunk.content_hash:
            plan.to_touch.append(chunk)
        else:
            plan.to_embed.append(chunk)
    plan.to_delete = [row.id for key, row in by_key.items() if key not in wanted]
    return plan


@dataclass
class IndexStats:
    product_id: int
    chunks: int
    embedded: int
    touched: int
    deleted: int


_EXISTING_SQL = """
SELECT id, source_type, source_id, chunk_index, content_hash
FROM ai.chunks
WHERE product_id = $1
"""

_UPSERT_SQL = """
INSERT INTO ai.chunks
    (source_type, source_id, product_id, category_id, chunk_index, content, content_hash, metadata, embedding)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::vector)
ON CONFLICT (source_type, source_id, chunk_index) DO UPDATE SET
    product_id   = EXCLUDED.product_id,
    category_id  = EXCLUDED.category_id,
    content      = EXCLUDED.content,
    content_hash = EXCLUDED.content_hash,
    metadata     = EXCLUDED.metadata,
    embedding    = EXCLUDED.embedding,
    updated_at   = now()
"""

_TOUCH_SQL = """
UPDATE ai.chunks
SET product_id = $4, category_id = $5, metadata = $6::jsonb, updated_at = now()
WHERE source_type = $1 AND source_id = $2 AND chunk_index = $3
  AND (product_id IS DISTINCT FROM $4 OR category_id IS DISTINCT FROM $5 OR metadata <> $6::jsonb)
"""


class Indexer:
    def __init__(self, pool: asyncpg.Pool, embedder: Embedder):
        self._pool = pool
        self._embedder = embedder

    async def index_product(self, product: Product, reviews: Sequence[Review]) -> IndexStats:
        chunks = build_product_chunks(product, reviews)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_EXISTING_SQL, product.id)
        existing = [
            ExistingChunk(r["id"], (r["source_type"], r["source_id"], r["chunk_index"]), r["content_hash"])
            for r in rows
        ]
        plan = plan_changes(existing, chunks)

        # Call the embedding API before taking a connection: never hold one during network I/O.
        vectors = (
            await self._embedder.embed([c.content for c in plan.to_embed], "document")
            if plan.to_embed
            else []
        )

        async with self._pool.acquire() as conn, conn.transaction():
            if plan.to_embed:
                await conn.executemany(
                    _UPSERT_SQL,
                    [
                        (
                            c.source_type,
                            c.source_id,
                            c.product_id,
                            c.category_id,
                            c.chunk_index,
                            c.content,
                            c.content_hash,
                            c.metadata_json,
                            to_pgvector(v),
                        )
                        for c, v in zip(plan.to_embed, vectors, strict=True)
                    ],
                )
            if plan.to_touch:
                await conn.executemany(
                    _TOUCH_SQL,
                    [
                        (
                            c.source_type,
                            c.source_id,
                            c.chunk_index,
                            c.product_id,
                            c.category_id,
                            c.metadata_json,
                        )
                        for c in plan.to_touch
                    ],
                )
            if plan.to_delete:
                await conn.execute("DELETE FROM ai.chunks WHERE id = ANY($1::bigint[])", plan.to_delete)

        return IndexStats(
            product.id, len(chunks), len(plan.to_embed), len(plan.to_touch), len(plan.to_delete)
        )

    async def delete_product(self, product_id: int) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute("DELETE FROM ai.chunks WHERE product_id = $1", product_id)
        return int(result.split()[-1])

    async def prune_products(self, keep_ids: set[int]) -> int:
        """Delete chunks of products that no longer exist in Spring."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM ai.chunks WHERE product_id IS NOT NULL AND NOT (product_id = ANY($1::bigint[]))",
                list(keep_ids),
            )
        return int(result.split()[-1])


async def reindex_product(spring: SpringClient, indexer: Indexer, product_id: int) -> IndexStats | int:
    """Re-index one product from Spring; if it's gone, delete its chunks (returns the count)."""
    product = await spring.get_product(product_id)
    if product is None:
        deleted = await indexer.delete_product(product_id)
        log.info("product %s not found in Spring, deleted %s chunks", product_id, deleted)
        return deleted
    stats = await indexer.index_product(product, await spring.get_reviews(product_id))
    log.info("indexed product %s: %s", product_id, stats)
    return stats


@dataclass
class ReindexSummary:
    products: int = 0
    failed: list[int] = field(default_factory=list)
    embedded: int = 0
    touched: int = 0
    deleted: int = 0
    pruned: int = 0


async def reindex_all(spring: SpringClient, indexer: Indexer, page_size: int = 50) -> ReindexSummary:
    """Full reconcile: index every product, then drop chunks of deleted products.

    Pruning only runs when every product was indexed, so a failure can't wipe the index.
    """
    summary = ReindexSummary()
    seen: set[int] = set()
    async for product in spring.iter_products(page_size):
        seen.add(product.id)
        try:
            stats = await indexer.index_product(product, await spring.get_reviews(product.id))
        except Exception:
            log.exception("failed to index product %s", product.id)
            summary.failed.append(product.id)
            continue
        summary.products += 1
        summary.embedded += stats.embedded
        summary.touched += stats.touched
        summary.deleted += stats.deleted

    if not summary.failed:
        summary.pruned = await indexer.prune_products(seen)
    log.info("reindex finished: %s", summary)
    return summary
