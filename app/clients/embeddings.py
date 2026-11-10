"""Voyage AI embeddings over plain HTTP (no SDK needed).

API: POST https://api.voyageai.com/v1/embeddings
     {"input": [...], "model": "voyage-3.5", "input_type": "document"|"query", "output_dimension": 1024}
"""

from typing import Literal, Protocol

import httpx
from tenacity import AsyncRetrying

from app.clients.retry import raise_for_status, retry_policy

InputType = Literal["document", "query"]

VOYAGE_EMBEDDINGS_URL = "https://api.voyageai.com/v1/embeddings"


class Embedder(Protocol):
    dimension: int

    async def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]: ...


class VoyageEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = "voyage-3.5",
        dimension: int = 1024,
        batch_size: int = 64,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.total_tokens = 0
        self._http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(60.0),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(await self._embed_batch(texts[start : start + self.batch_size], input_type))
        return vectors

    async def _embed_batch(self, batch: list[str], input_type: InputType) -> list[list[float]]:
        payload = {
            "input": batch,
            "model": self.model,
            "input_type": input_type,
            "output_dimension": self.dimension,
        }
        async for attempt in AsyncRetrying(**retry_policy(attempts=5)):
            with attempt:
                response = await self._http.post(VOYAGE_EMBEDDINGS_URL, json=payload)
                raise_for_status(response)
        body = response.json()
        self.total_tokens += body.get("usage", {}).get("total_tokens", 0)

        data = sorted(body["data"], key=lambda item: item["index"])
        if len(data) != len(batch):
            raise RuntimeError(f"Voyage returned {len(data)} embeddings for {len(batch)} inputs")
        vectors = [item["embedding"] for item in data]
        if any(len(v) != self.dimension for v in vectors):
            raise RuntimeError(f"Voyage returned vectors that aren't {self.dimension}-dimensional")
        return vectors
