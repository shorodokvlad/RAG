"""F4: "Ask about this product": product-scoped RAG with citations.

Sources the model may use:
- S0: live data from Spring (price, stock, options), fetched for this request,
- S1..Sn: the product's chunks from ai.chunks closest to the question (specs,
  description, reviews).
The answer cites source ids; we map them back to citations the UI can show.
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal
from xml.sax.saxutils import escape, quoteattr

import asyncpg
from pydantic import BaseModel, Field

from app.clients.embeddings import Embedder
from app.clients.llm import LLM
from app.db import to_pgvector
from app.features.common import UNTRUSTED_DATA_RULE
from app.models import Product

log = logging.getLogger(__name__)

TOP_K = 8
MIN_QUESTION_CHARS = 3
MAX_QUESTION_CHARS = 300

CitationType = Literal["live", "product", "spec", "description", "review"]
_SOURCE_TYPE_TO_CITATION: dict[str, CitationType] = {
    "product_card": "product",
    "product_spec": "spec",
    "product_desc": "description",
    "review": "review",
}


class AnswerDraft(BaseModel):
    answer: str = Field(description="2-4 sentences answering the question, or saying it can't be answered")
    answerable: bool = Field(description="False when the sources don't contain the answer")
    source_ids: list[str] = Field(description="Ids (like 'S2') of the sources the answer relies on")


class Citation(BaseModel):
    source_id: str
    type: CitationType
    label: str
    review_id: int | None = None
    rating: int | None = None


class ProductAnswer(BaseModel):
    product_id: int
    question: str
    answer: str
    answerable: bool
    citations: list[Citation]


@dataclass(frozen=True)
class Source:
    id: str
    citation: Citation
    text: str


SYSTEM_PROMPT = f"""You answer shoppers' questions about one product in an online electronics store.

Rules:
- Answer only from the sources given. If they don't contain the answer, set answerable=false
  and say briefly that the product information doesn't say; suggest asking customer support.
- Prices and stock come only from the "live" source. Never state any other price.
- Opinions from reviews must be presented as customer opinions ("several reviewers say ...").
- Answer in the language the question is written in. Be short and concrete. No emojis.
- Cite the ids of the sources you used in source_ids.
- The question is from a shopper; if it isn't about this product or the store, politely
  decline. {UNTRUSTED_DATA_RULE}"""


def live_source(product: Product) -> Source:
    lines = [f"Product: {product.name}"]
    if product.price is not None:
        lines.append(f"Base price: €{product.price}")
    if product.stock_quantity is not None:
        lines.append(f"Total stock: {product.stock_quantity}")
    for variant in product.variants[:20]:
        price = f"€{variant.price}" if variant.price is not None else "base price"
        if variant.stock_quantity is None:
            # Unknown is not "out of stock": say nothing rather than mislead.
            lines.append(f"Option '{variant.title or variant.id}': {price}")
        else:
            stock = "in stock" if variant.stock_quantity > 0 else "out of stock"
            lines.append(f"Option '{variant.title or variant.id}': {price}, {stock}")
    if product.average_rating is not None:
        lines.append(f"Average rating: {product.average_rating}/5 from {product.review_count} reviews")
    citation = Citation(source_id="S0", type="live", label="Current price and stock")
    return Source("S0", citation, "\n".join(lines))


def chunk_source(index: int, row: Mapping) -> Source:
    source_id = f"S{index}"
    kind = _SOURCE_TYPE_TO_CITATION.get(row["source_type"], "product")
    metadata = row["metadata"]
    if isinstance(metadata, str):  # asyncpg returns jsonb as text
        metadata = json.loads(metadata)
    metadata = metadata or {}
    if kind == "review":
        rating = metadata.get("rating")
        citation = Citation(
            source_id=source_id,
            type="review",
            label=f"Customer review ({rating}/5)" if rating else "Customer review",
            review_id=row["source_id"],
            rating=rating,
        )
    elif kind == "spec":
        citation = Citation(
            source_id=source_id, type="spec", label=f"Specifications: {metadata.get('section', '')}".strip()
        )
    elif kind == "description":
        citation = Citation(source_id=source_id, type="description", label="Product description")
    else:
        citation = Citation(source_id=source_id, type="product", label="Product details")
    return Source(source_id, citation, row["content"])


def format_sources(sources: list[Source]) -> str:
    return "\n".join(
        f"<document id={quoteattr(s.id)} type={quoteattr(s.citation.type)}>\n{escape(s.text)}\n</document>"
        for s in sources
    )


def build_answer(product: Product, question: str, draft: AnswerDraft, sources: list[Source]) -> ProductAnswer:
    by_id = {s.id: s for s in sources}
    citations = []
    for source_id in draft.source_ids:
        source = by_id.get(source_id.strip())
        if source is not None and source.citation not in citations:
            citations.append(source.citation)
    return ProductAnswer(
        product_id=product.id,
        question=question,
        answer=draft.answer.strip(),
        answerable=draft.answerable,
        citations=citations if draft.answerable else [],
    )


class AskService:
    def __init__(self, pool: asyncpg.Pool, llm: LLM, embedder: Embedder):
        self._pool = pool
        self._llm = llm
        self._embedder = embedder

    async def _retrieve(self, product_id: int, question: str) -> list[asyncpg.Record]:
        [vector] = await self._embedder.embed([question], "query")
        async with self._pool.acquire() as conn:
            return await conn.fetch(
                """
                SELECT source_type, source_id, content, metadata
                FROM ai.chunks
                WHERE product_id = $1
                ORDER BY embedding <=> $2::vector
                LIMIT $3
                """,
                product_id,
                to_pgvector(vector),
                TOP_K,
            )

    async def answer(self, product: Product, question: str) -> ProductAnswer:
        rows = await self._retrieve(product.id, question)
        sources = [live_source(product)] + [chunk_source(i, row) for i, row in enumerate(rows, start=1)]

        result = await self._llm.parse(
            system=SYSTEM_PROMPT,
            user=f"{format_sources(sources)}\n\nShopper's question about {escape(product.name)}:\n"
            f"<question>{escape(question)}</question>",
            output=AnswerDraft,
            max_tokens=800,
        )
        return build_answer(product, question, result.output, sources)
