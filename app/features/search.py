"""F5: natural-language product search.

Hybrid retrieval ranks products by meaning and keywords; Spring then supplies live price,
stock and images (never stored in the index). Price and stock filters run on that live
data, after hydration, so they're always current.
"""

from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel, Field

from app.clients.spring import SpringClient
from app.models import Product
from app.retrieval.hybrid import HybridSearch, ProductHit

MAX_RESULTS = 12
# Fetch more hits than asked for, so price/stock filters still leave enough results.
OVERFETCH = 4
SNIPPET_CHARS = 180


class SearchFilters(BaseModel):
    category_id: int | None = None
    min_price: Decimal | None = Field(default=None, ge=0)
    max_price: Decimal | None = Field(default=None, ge=0)
    in_stock_only: bool = False


class ProductResult(BaseModel):
    id: int
    name: str
    category_name: str | None = None
    image_url: str | None = None
    # Lowest current price across the product and its variants, in EUR.
    price: Decimal | None = None
    in_stock: bool | None = None
    average_rating: Decimal | None = None
    review_count: int = 0
    # Text from the index that matched the query: "why this result".
    match: str = ""


def lowest_price(product: Product) -> Decimal | None:
    prices = [v.price for v in product.variants if v.price is not None]
    if not prices and product.price is not None:
        prices = [product.price]
    return min(prices) if prices else None


def stock_state(product: Product) -> bool | None:
    """True/False when Spring knows the stock, None when it doesn't."""
    known = [v.stock_quantity for v in product.variants if v.stock_quantity is not None]
    if known:
        return any(q > 0 for q in known)
    if product.stock_quantity is not None:
        return product.stock_quantity > 0
    return None


def snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[: SNIPPET_CHARS - 1].rstrip() + "…"


def to_result(product: Product, hit: ProductHit | None = None) -> ProductResult:
    return ProductResult(
        id=product.id,
        name=product.name,
        category_name=product.category_name,
        image_url=product.image_url,
        price=lowest_price(product),
        in_stock=stock_state(product),
        average_rating=product.average_rating,
        review_count=product.review_count,
        match=snippet(hit.snippet) if hit else "",
    )


def passes(result: ProductResult, filters: SearchFilters) -> bool:
    if filters.in_stock_only and result.in_stock is False:
        return False
    if filters.min_price is not None and (result.price is None or result.price < filters.min_price):
        return False
    return not (filters.max_price is not None and (result.price is None or result.price > filters.max_price))


@dataclass
class SearchService:
    hybrid: HybridSearch
    spring: SpringClient

    async def _hydrate(self, hits: list[ProductHit]) -> list[tuple[Product, ProductHit]]:
        products = {p.id: p for p in await self.spring.get_products_by_ids([h.product_id for h in hits])}
        # Keep the search order; products deleted since indexing are skipped.
        return [(products[h.product_id], h) for h in hits if h.product_id in products]

    async def search(
        self, query: str, filters: SearchFilters | None = None, limit: int = 6
    ) -> list[ProductResult]:
        filters = filters or SearchFilters()
        limit = max(1, min(limit, MAX_RESULTS))
        hits = await self.hybrid.search(query, limit=limit * OVERFETCH, category_id=filters.category_id)
        results = [to_result(product, hit) for product, hit in await self._hydrate(hits)]
        return [r for r in results if passes(r, filters)][:limit]

    async def similar(self, product_id: int, limit: int = 6) -> list[ProductResult]:
        hits = await self.hybrid.similar(product_id, limit=max(1, min(limit, MAX_RESULTS)))
        return [to_result(product, hit) for product, hit in await self._hydrate(hits)]
