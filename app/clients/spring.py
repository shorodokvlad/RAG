"""Client for Spring's /internal/ai/** endpoints (authenticated with X-Service-Token)."""

from collections.abc import AsyncIterator

import httpx
from tenacity import AsyncRetrying

from app.clients.retry import raise_for_status, retry_policy
from app.models import Product, ProductPage, Review
from app.security import SERVICE_TOKEN_HEADER


class SpringClient:
    def __init__(
        self,
        base_url: str,
        service_token: str,
        timeout_seconds: float = 90.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={SERVICE_TOKEN_HEADER: service_token, "Accept": "application/json"},
            timeout=httpx.Timeout(timeout_seconds, connect=timeout_seconds),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _get(self, path: str, params: dict | None = None) -> httpx.Response | None:
        """GET with retries; returns None on 404."""
        async for attempt in AsyncRetrying(**retry_policy()):
            with attempt:
                response = await self._http.get(path, params=params)
                if response.status_code == 404:
                    return None
                raise_for_status(response)
                return response
        raise AssertionError("unreachable")

    async def get_products_page(self, page: int, size: int = 50) -> ProductPage:
        response = await self._get("/internal/ai/products", {"page": page, "size": size})
        if response is None:
            raise RuntimeError("Spring has no /internal/ai/products endpoint; is the backend up to date?")
        return ProductPage.model_validate(response.json())

    async def iter_products(self, page_size: int = 50) -> AsyncIterator[Product]:
        page = 0
        while True:
            result = await self.get_products_page(page, page_size)
            for product in result.items:
                yield product
            page += 1
            if page >= result.total_pages:
                return

    async def get_product(self, product_id: int) -> Product | None:
        response = await self._get(f"/internal/ai/products/{product_id}")
        return Product.model_validate(response.json()) if response is not None else None

    async def get_products_by_ids(self, product_ids: list[int]) -> list[Product]:
        if not product_ids:
            return []
        response = await self._get(
            "/internal/ai/products/by-ids", {"ids": ",".join(str(i) for i in product_ids)}
        )
        return [Product.model_validate(item) for item in response.json()] if response is not None else []

    async def get_reviews(self, product_id: int) -> list[Review]:
        response = await self._get(f"/internal/ai/products/{product_id}/reviews")
        return [Review.model_validate(item) for item in response.json()] if response is not None else []
