import json

import httpx
import httpx2
import openai
import pytest
import respx
from pydantic import BaseModel

from app.clients.embeddings import GEMINI_API, VOYAGE_EMBEDDINGS_URL, GeminiEmbedder, VoyageEmbedder
from app.clients.llm import LLM, LLMRefusal, LLMTruncated, LLMUnavailable
from app.clients.spring import SpringClient

LLM_BASE = "https://llm.test/v1/"


class Verdict(BaseModel):
    verdict: str
    score: int


def chat_response(content: str | None, finish_reason: str = "stop", refusal: str | None = None) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "gemini-test",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def mock_llm(status: int, body: dict, requests: list | None = None) -> LLM:
    """LLM whose OpenAI client talks to an in-memory transport (the SDK uses httpx2, not httpx)."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if requests is not None:
            requests.append(request)
        return httpx2.Response(status, json=body)

    client = openai.AsyncOpenAI(
        api_key="key",
        base_url=LLM_BASE,
        max_retries=0,
        http_client=openai.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return LLM("key", "gemini-test", LLM_BASE, client=client)


async def test_llm_parses_structured_output_and_sends_schema():
    requests = []
    llm = mock_llm(200, chat_response('{"verdict": "good", "score": 4}'), requests)

    result = await llm.parse(system="sys", user="hi", output=Verdict)

    assert result.output == Verdict(verdict="good", score=4)
    assert (result.model, result.input_tokens, result.output_tokens) == ("gemini-test", 10, 5)
    body = json.loads(requests[0].content)
    assert body["response_format"]["type"] == "json_schema"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert requests[0].headers["Authorization"] == "Bearer key"
    assert str(requests[0].url) == f"{LLM_BASE}chat/completions"
    await llm.aclose()


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (chat_response('{"verdict": "go', finish_reason="length"), LLMTruncated),
        (chat_response(None, refusal="I can't help with that"), LLMRefusal),
    ],
)
async def test_llm_truncation_and_refusal_raise(response, error):
    llm = mock_llm(200, response)
    with pytest.raises(error):
        await llm.parse(system="s", user="u", output=Verdict)
    await llm.aclose()


async def test_llm_quota_exhausted_is_unavailable():
    llm = mock_llm(429, {"error": {"message": "quota"}})
    with pytest.raises(LLMUnavailable):
        await llm.parse(system="s", user="u", output=Verdict)
    await llm.aclose()


@respx.mock
async def test_gemini_embedder_sends_task_type_and_normalizes():
    url = f"{GEMINI_API}/models/gemini-embedding-001:batchEmbedContents"

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.headers["x-goog-api-key"] == "gkey"
        assert {r["taskType"] for r in body["requests"]} == {"RETRIEVAL_QUERY"}
        assert {r["outputDimensionality"] for r in body["requests"]} == {2}
        assert body["requests"][0]["model"] == "models/gemini-embedding-001"
        return httpx.Response(200, json={"embeddings": [{"values": [3.0, 4.0]} for _ in body["requests"]]})

    route = respx.post(url).mock(side_effect=respond)
    embedder = GeminiEmbedder("gkey", dimension=2, batch_size=2)

    vectors = await embedder.embed(["a", "b", "c"], "query")

    assert vectors == [[0.6, 0.8]] * 3
    assert route.call_count == 2
    await embedder.aclose()


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
