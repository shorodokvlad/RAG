"""Hybrid catalog search: vector similarity + full-text keywords, merged with Reciprocal Rank Fusion.

Vectors find meaning ("phone for photos" → camera specs); keywords find exact terms the
embedding blurs (model numbers like "S11", "256GB"). Each list ranks chunks; a product's
score sums 1 / (k + rank) of its best chunk in each list, so a product that both lists
like wins over one that only one list likes.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import asyncpg

from app.clients.embeddings import Embedder
from app.db import to_pgvector

RRF_K = 60
CANDIDATES = 40
PRODUCT_SOURCE_TYPES = ("product_card", "product_spec", "product_desc", "review")
# Keyword search ignores filler words so "a phone for photos" matches on phone/photos.
_STOPWORD_TEXT = (
    "a an and are as at be best but by can do for from good has have i in is it me my need "
    "of on or show that the this to under over with want looking find which what some any "
    "cheap cheapest price lei euro eur sub pentru si cu un o de la"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())
_WORD = re.compile(r"[\w.]+", re.UNICODE)
MAX_KEYWORDS = 12


@dataclass(frozen=True)
class ProductHit:
    product_id: int
    score: float
    # Best-matching text for this product, used to explain why it matched.
    snippet: str
    source_type: str


def keyword_query(text: str) -> str | None:
    """OR-query for to_tsquery('simple', ...): every meaningful word, safe to embed in SQL params."""
    words: list[str] = []
    for raw in _WORD.findall(text.lower()):
        word = raw.strip("._")
        if len(word) < 2 or word in _STOPWORDS or word.isdigit() and len(word) < 3:
            continue
        if word not in words:
            words.append(word)
    if not words:
        return None
    # Quoting makes each word a literal lexeme, so characters like ':' or '!' can't form operators.
    return " | ".join("'" + w.replace("'", "") + "'" for w in words[:MAX_KEYWORDS])


def fuse(ranked_lists: Sequence[Sequence[Mapping]], limit: int, k: int = RRF_K) -> list[ProductHit]:
    """Reciprocal Rank Fusion over chunk lists, aggregated per product.

    Each list must be ordered best-first. A product counts once per list (its best chunk), and
    the snippet is the product's best-ranked chunk across all lists.
    """
    scores: dict[int, float] = {}
    best: dict[int, tuple[int, Mapping]] = {}
    for rows in ranked_lists:
        seen: set[int] = set()
        for rank, row in enumerate(rows, start=1):
            product_id = row["product_id"]
            if product_id is None or product_id in seen:
                continue
            seen.add(product_id)
            scores[product_id] = scores.get(product_id, 0.0) + 1.0 / (k + rank)
            if product_id not in best or rank < best[product_id][0]:
                best[product_id] = (rank, row)
    ordered = sorted(scores, key=lambda pid: (-scores[pid], best[pid][0], pid))
    return [
        ProductHit(
            product_id=pid,
            score=round(scores[pid], 6),
            snippet=best[pid][1]["content"],
            source_type=best[pid][1]["source_type"],
        )
        for pid in ordered[:limit]
    ]


class HybridSearch:
    def __init__(self, pool: asyncpg.Pool, embedder: Embedder):
        self._pool = pool
        self._embedder = embedder

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        category_id: int | None = None,
        exclude_product_ids: Sequence[int] = (),
        candidates: int = CANDIDATES,
    ) -> list[ProductHit]:
        [vector] = await self._embedder.embed([query], "query")
        keywords = keyword_query(query)
        filters = "source_type = ANY($1) AND product_id IS NOT NULL AND NOT (product_id = ANY($2))"
        filters += " AND category_id = $3" if category_id is not None else " AND $3::bigint IS NULL"
        common = [list(PRODUCT_SOURCE_TYPES), list(exclude_product_ids), category_id, candidates]

        async with self._pool.acquire() as conn:
            by_vector = await conn.fetch(
                f"""
                SELECT product_id, source_type, content
                FROM ai.chunks
                WHERE {filters}
                ORDER BY embedding <=> $5::vector
                LIMIT $4
                """,
                *common,
                to_pgvector(vector),
            )
            by_keyword = []
            if keywords:
                by_keyword = await conn.fetch(
                    f"""
                    SELECT product_id, source_type, content
                    FROM ai.chunks
                    WHERE {filters} AND tsv @@ to_tsquery('simple', $5)
                    ORDER BY ts_rank(tsv, to_tsquery('simple', $5)) DESC
                    LIMIT $4
                    """,
                    *common,
                    keywords,
                )
        return fuse([by_vector, by_keyword], limit)

    async def similar(self, product_id: int, limit: int = 6) -> list[ProductHit]:
        """Nearest products to this product's card (same embedding space as search)."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT c.product_id, c.source_type, c.content
                FROM ai.chunks c,
                     (SELECT embedding FROM ai.chunks
                      WHERE source_type = 'product_card' AND product_id = $1 LIMIT 1) me
                WHERE c.source_type = 'product_card' AND c.product_id <> $1
                ORDER BY c.embedding <=> me.embedding
                LIMIT $2
                """,
                product_id,
                limit,
            )
        return fuse([rows], limit)
