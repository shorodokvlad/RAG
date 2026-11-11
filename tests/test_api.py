from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import health, index, products
from app.clients.llm import LLMError, LLMUnavailable
from app.features.ask import ProductAnswer
from app.features.overview import OverviewRecord, ProductOverview
from app.services import Services, get_services
from tests.factories import make_product

TOKEN = {"X-Service-Token": "test-token"}
NOW = datetime(2026, 11, 11, tzinfo=UTC)


class FakeSpring:
    def __init__(self, product=None):
        self.product = product

    async def get_product(self, product_id):
        return self.product if self.product and self.product.id == product_id else None

    async def get_reviews(self, product_id):
        return []


class FakeOverviews:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    async def get_or_generate(self, product):
        if self.error:
            raise self.error
        return self.result


class FakeSummaries:
    min_reviews = 3

    async def get_or_generate(self, product, reviews, force=False):
        return None


class FakeAsker:
    async def answer(self, product, question):
        return ProductAnswer(
            product_id=product.id, question=question, answer="Yes, 5G.", answerable=True, citations=[]
        )


@pytest.fixture
def services():
    # No real pool/clients: endpoints under test only check auth, wiring and error mapping.
    svc = Services(pool=None, spring=FakeSpring(make_product()), embedder=SimpleNamespace(), llm=None)
    svc.overviews = FakeOverviews()
    svc.review_summaries = FakeSummaries()
    svc.asker = FakeAsker()
    return svc


@pytest.fixture
def client(services, monkeypatch):
    calls = []

    async def fake_reindex_product(svc, product_id):
        calls.append(("product", product_id))

    async def fake_reindex_all(svc):
        calls.append(("all",))

    monkeypatch.setattr(index, "_reindex_product_task", fake_reindex_product)
    monkeypatch.setattr(index, "_reindex_all_task", fake_reindex_all)

    app = FastAPI()  # no lifespan: no database needed
    app.include_router(health.router)
    app.include_router(index.router)
    app.include_router(products.router)
    app.dependency_overrides[get_services] = lambda: services
    test_client = TestClient(app)
    test_client.calls = calls
    return test_client


def test_health_is_public(client):
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("post", "/index/products/1"),
        ("delete", "/index/products/1"),
        ("post", "/index/reconcile"),
        ("get", "/health/ready"),
        ("get", "/products/12/overview"),
        ("get", "/products/12/review-summary"),
        ("post", "/products/12/ask"),
    ],
)
def test_protected_endpoints_need_the_token(client, method, path):
    assert getattr(client, method)(path).status_code == 401
    assert getattr(client, method)(path, headers={"X-Service-Token": "wrong"}).status_code == 401


def test_index_product_is_accepted_and_scheduled(client):
    response = client.post("/index/products/12", headers=TOKEN)

    assert response.status_code == 202
    assert client.calls == [("product", 12)]


def test_index_without_any_key_is_503(client, services):
    services.embedder = None
    services.llm = None
    assert client.post("/index/products/12", headers=TOKEN).status_code == 503


def test_index_with_only_llm_key_still_accepted(client, services):
    services.embedder = None
    services.llm = SimpleNamespace()
    assert client.post("/index/products/12", headers=TOKEN).status_code == 202


def test_second_reconcile_while_running_is_409(client, services):
    services.reconcile_running = True
    assert client.post("/index/reconcile", headers=TOKEN).status_code == 409
    services.reconcile_running = False
    assert client.post("/index/reconcile", headers=TOKEN).status_code == 202
    assert client.calls == [("all",)]


def test_overview_ready(client, services):
    overview = ProductOverview(
        summary="A 5G phone.", best_for=["Calls"], highlights=[], watch_outs=[], suggested_questions=[]
    )
    services.overviews = FakeOverviews(OverviewRecord(12, overview, "gemini", NOW))

    body = client.get("/products/12/overview", headers=TOKEN).json()

    assert body["status"] == "ready"
    assert body["overview"]["summary"] == "A 5G phone."


def test_overview_not_enough_information(client):
    body = client.get("/products/12/overview", headers=TOKEN).json()
    assert body == {"product_id": 12, "status": "not_enough_information", "overview": None,
                    "model": None, "generated_at": None}  # fmt: skip


def test_unknown_product_is_404(client):
    assert client.get("/products/999/overview", headers=TOKEN).status_code == 404


@pytest.mark.parametrize(("error", "code"), [(LLMUnavailable("429"), 503), (LLMError("bad json"), 502)])
def test_llm_failures_map_to_http_errors(client, services, error, code):
    services.overviews = FakeOverviews(error=error)
    assert client.get("/products/12/overview", headers=TOKEN).status_code == code


def test_generation_disabled_without_llm_is_503(client, services):
    services.overviews = None
    assert client.get("/products/12/overview", headers=TOKEN).status_code == 503


def test_review_summary_not_enough_reviews(client):
    body = client.get("/products/12/review-summary", headers=TOKEN).json()
    assert body["status"] == "not_enough_reviews" and body["min_reviews"] == 3


def test_ask_validates_question_and_answers(client):
    assert client.post("/products/12/ask", headers=TOKEN, json={"question": "x"}).status_code == 422
    assert client.post("/products/12/ask", headers=TOKEN, json={"question": "a" * 301}).status_code == 422

    body = client.post("/products/12/ask", headers=TOKEN, json={"question": " Has it 5G? "}).json()
    assert body["answer"] == "Yes, 5G." and body["question"] == "Has it 5G?"
