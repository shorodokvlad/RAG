from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api import health, index
from app.services import Services, get_services

TOKEN = {"X-Service-Token": "test-token"}


@pytest.fixture
def services():
    # No real pool/clients: the endpoints under test only check auth and schedule work.
    return Services(pool=None, spring=None, embedder=SimpleNamespace(dimension=1024))


@pytest.fixture
def client(services, monkeypatch):
    from fastapi import FastAPI

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
    ],
)
def test_protected_endpoints_need_the_token(client, method, path):
    assert getattr(client, method)(path).status_code == 401
    assert getattr(client, method)(path, headers={"X-Service-Token": "wrong"}).status_code == 401


def test_index_product_is_accepted_and_scheduled(client):
    response = client.post("/index/products/12", headers=TOKEN)

    assert response.status_code == 202
    assert client.calls == [("product", 12)]


def test_index_without_voyage_key_is_503(client, services):
    services.embedder = None
    assert client.post("/index/products/12", headers=TOKEN).status_code == 503


def test_second_reconcile_while_running_is_409(client, services):
    services.reconcile_running = True
    assert client.post("/index/reconcile", headers=TOKEN).status_code == 409
    services.reconcile_running = False
    assert client.post("/index/reconcile", headers=TOKEN).status_code == 202
    assert client.calls == [("all",)]
