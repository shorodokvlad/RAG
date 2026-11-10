import json

import httpx
import pytest
import respx

from app.clients.embeddings import VOYAGE_EMBEDDINGS_URL, VoyageEmbedder
from app.clients.spring import SpringClient


def product_json(pid: int) -> dict:
    return {"id": pid, "name": f"P{pid}", "categoryId": 1, "categoryName": "Phones", "reviewCount": 0}


@respx.mock
async def test_spring_iterates_pages_and_sends_token():
    route = respx.get("http://spring.test/internal/ai/products").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "items": [product_json(1), product_json(2)],
                    "page": 0,
                    "size": 2,
                    "totalPages": 2,
                    "totalElements": 3,
                },
            ),
            httpx.Response(
                200,
                json={"items": [product_json(3)], "page": 1, "size": 2, "totalPages": 2, "totalElements": 3},
            ),
        ]
    )
    spring = SpringClient("http://spring.test/", "secret")

    ids = [p.id async for p in spring.iter_products(page_size=2)]

    assert ids == [1, 2, 3]
    assert route.calls[0].request.headers["X-Service-Token"] == "secret"
    assert route.calls[1].request.url.params["page"] == "1"
    await spring.aclose()


@respx.mock
async def test_spring_missing_product_is_none_and_transient_errors_retry(monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _: None)
    respx.get("http://spring.test/internal/ai/products/9").mock(return_value=httpx.Response(404))
    respx.get("http://spring.test/internal/ai/products/5/reviews").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json=[{"id": 1, "productId": 5, "rating": 5, "content": "ok"}]),
        ]
    )
    spring = SpringClient("http://spring.test", "secret")

    assert await spring.get_product(9) is None
    reviews = await spring.get_reviews(5)
    assert [r.content for r in reviews] == ["ok"]
    await spring.aclose()


@respx.mock
async def test_voyage_batches_and_restores_order():
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "voyage-3.5" and body["input_type"] == "document"
        assert body["output_dimension"] == 2
        data = [{"index": i, "embedding": [float(len(t)), 0.0]} for i, t in enumerate(body["input"])]
        return httpx.Response(200, json={"data": list(reversed(data)), "usage": {"total_tokens": 7}})

    route = respx.post(VOYAGE_EMBEDDINGS_URL).mock(side_effect=respond)
    embedder = VoyageEmbedder("key", dimension=2, batch_size=2)

    vectors = await embedder.embed(["a", "bb", "ccc"], "document")

    assert vectors == [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    assert route.call_count == 2
    assert route.calls[0].request.headers["Authorization"] == "Bearer key"
    assert embedder.total_tokens == 14
    await embedder.aclose()


@respx.mock
async def test_voyage_wrong_dimension_fails():
    respx.post(VOYAGE_EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})
    )
    embedder = VoyageEmbedder("key", dimension=2)
    with pytest.raises(RuntimeError, match="2-dimensional"):
        await embedder.embed(["a"], "query")
    await embedder.aclose()
