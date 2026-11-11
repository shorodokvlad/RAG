"""F2: AI product overview, generated once per product version and cached in ai.product_overviews."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime

import asyncpg
from pydantic import BaseModel, Field

from app.clients.llm import LLM
from app.features.common import UNTRUSTED_DATA_RULE, KeyedLocks, fingerprint, product_facts
from app.models import Product

log = logging.getLogger(__name__)

PROMPT_VERSION = "overview-v1"


class Highlight(BaseModel):
    title: str = Field(description="2-4 word feature name, e.g. 'Battery life'")
    text: str = Field(description="One sentence with the concrete facts behind it")


class ProductOverview(BaseModel):
    summary: str = Field(description="2-3 sentences: what the product is and what stands out")
    best_for: list[str] = Field(description="2-4 short phrases naming who or what it suits best")
    highlights: list[Highlight] = Field(description="3-5 strongest points, grounded in the specs")
    watch_outs: list[str] = Field(
        description="0-3 honest limitations visible in the data; empty if none are evident"
    )
    suggested_questions: list[str] = Field(
        description="3 short questions a shopper might ask about this product"
    )


SYSTEM_PROMPT = f"""You write short, factual product overviews for an online electronics store.

Rules:
- Use only facts present in the product data. Never invent specs, prices, awards or comparisons.
- If a fact isn't in the data, leave it out rather than guessing.
- Plain, neutral shopping language: no hype, no exclamation marks, no emojis.
- Never mention price, stock or delivery; those are shown elsewhere on the page.
- Watch-outs must come from the data itself (e.g. a missing feature the specs make explicit);
  leave the list empty when nothing is evident.
- {UNTRUSTED_DATA_RULE}"""


@dataclass(frozen=True)
class OverviewRecord:
    product_id: int
    overview: ProductOverview
    model: str
    generated_at: datetime


class OverviewService:
    def __init__(self, pool: asyncpg.Pool, llm: LLM, language: str):
        self._pool = pool
        self._llm = llm
        self._language = language
        self._locks = KeyedLocks()

    def _source_hash(self, facts: str) -> str:
        return fingerprint(PROMPT_VERSION, self._llm.model, self._language, facts)

    async def _load(self, product_id: int) -> tuple[OverviewRecord, str] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT overview, source_hash, model, generated_at FROM ai.product_overviews "
                "WHERE product_id = $1",
                product_id,
            )
        if row is None:
            return None
        record = OverviewRecord(
            product_id,
            ProductOverview.model_validate_json(row["overview"]),
            row["model"],
            row["generated_at"],
        )
        return record, row["source_hash"]

    async def get_or_generate(self, product: Product) -> OverviewRecord | None:
        """Cached overview if the product text is unchanged; otherwise generate a new one.

        Returns None when the product has too little information for a useful overview.
        """
        facts = product_facts(product)
        if facts is None:
            return None
        source_hash = self._source_hash(facts)

        async with self._locks(product.id):
            cached = await self._load(product.id)
            if cached is not None and cached[1] == source_hash:
                return cached[0]

            result = await self._llm.parse(
                system=SYSTEM_PROMPT,
                user=f"Write the overview in {self._language}.\n\n{facts}",
                output=ProductOverview,
                max_tokens=1500,
            )
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO ai.product_overviews (product_id, lang, overview, source_hash, model)
                    VALUES ($1, $2, $3::jsonb, $4, $5)
                    ON CONFLICT (product_id) DO UPDATE SET
                        lang = EXCLUDED.lang, overview = EXCLUDED.overview,
                        source_hash = EXCLUDED.source_hash, model = EXCLUDED.model,
                        generated_at = now()
                    RETURNING generated_at
                    """,
                    product.id,
                    self._language,
                    json.dumps(result.output.model_dump(), ensure_ascii=False),
                    source_hash,
                    result.model,
                )
            log.info("generated overview for product %s", product.id)
            return OverviewRecord(product.id, result.output, result.model, row["generated_at"])

    async def delete(self, product_id: int) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM ai.product_overviews WHERE product_id = $1", product_id)
