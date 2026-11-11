from decimal import Decimal

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import search as search_api
from app.features.search import SearchFilters, SearchService, lowest_price, stock_state
from app.models import Variant
from app.retrieval.hybrid import ProductHit, fuse, keyword_query
from app.services import get_services
from tests.factories import make_product

TOKEN = {"X-Service-Token": "test-token"}


def row(product_id, content="text", source_type="product_card"):
    return {"product_id": product_id, "content": content, "source_type": source_type}


# --- keyword query --------------------------------------------------------------------------


def test_keyword_query_drops_filler_and_quotes_words():
    assert keyword_query("I need a good phone for photos under 3000 lei") == "'phone' | 'photos' | '3000'"


def test_keyword_query_keeps_model_numbers_and_neutralizes_operators():
    assert keyword_query("Apple S11 & 256GB!: <-> x") == "'apple' | 's11' | '256gb'"


def test_keyword_query_none_without_meaningful_words():
    assert keyword_query("the best for me") is None
    assert keyword_query("'' :: !!") is None


# --- reciprocal rank fusion -----------------------------------------------------------------


def test_fuse_prefers_products_both_lists_like():
    by_vector = [row(1, "v1"), row(2, "v2"), row(3, "v3")]
    by_keyword = [row(2, "k2"), row(4, "k4")]

    hits = fuse([by_vector, by_keyword], limit=10)

    assert [h.product_id for h in hits] == [2, 1, 4, 3]
    assert hits[0].score == round(1 / 62 + 1 / 61, 6)
    assert hits[0].snippet == "k2"  # best rank across lists (1st in keywords)


def test_fuse_counts_each_product_once_per_list_and_skips_missing_ids():
    rows = [row(1, "spec", "product_spec"), row(1, "card"), row(None), row(2)]

    hits = fuse([rows], limit=10)

    assert [(h.product_id, h.snippet, h.source_type) for h in hits] == [
        (1, "spec", "product_spec"),
        (2, "text", "product_card"),
    ]
    # Ranks are chunk positions: product 2 is the 4th row, so it scores 1 / (60 + 4).
    assert hits[1].score == round(1 / 64, 6)


def test_fuse_limit():
    assert len(fuse([[row(i) for i in range(10)]], limit=3)) == 3


# --- live data and filters ------------------------------------------------------------------


def test_lowest_price_and_stock_prefer_variants():
    product = make_product(
        price="999",
        stock_quantity=0,
        variants=[
            Variant(id=1, price=Decimal("1200"), stock_quantity=0),
            Variant(id=2, price=Decimal("1100"), stock_quantity=3),
        ],
    )
    assert lowest_price(product) == Decimal("1100")
    assert stock_state(product) is True


def test_stock_unknown_is_none_not_out_of_stock():
    product = make_product(stock_quantity=None, variants=[Variant(id=1)])
    assert stock_state(product) is None
    assert lowest_price(product) == Decimal("4999.99")


class FakeHybrid:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    async def search(self, query, *, limit, category_id=None):
        self.calls.append((query, limit, category_id))
        return self.hits[:limit]

    async def similar(self, product_id, limit=6):
        return [h for h in self.hits if h.product_id != product_id][:limit]


class FakeSpring:
    def __init__(self, products):
        self.products = {p.id: p for p in products}

    async def get_products_by_ids(self, ids):
        return [self.products[i] for i in ids if i in self.products]


def hit(product_id, snippet="matched text"):
    return ProductHit(product_id=product_id, score=0.1, snippet=snippet, source_type="product_card")


CATALOG = [
    make_product(id=1, name="Cheap phone", price="100", variants=[], stock_quantity=5),
    make_product(id=2, name="Mid phone", price="500", variants=[], stock_quantity=0),
    make_product(id=3, name="Pricey phone", price="1500", variants=[], stock_quantity=2),
]


async def test_search_keeps_order_skips_deleted_and_applies_live_filters():
    hybrid = FakeHybrid([hit(3), hit(99), hit(2), hit(1, "  long\n text  ")])
    service = SearchService(hybrid, FakeSpring(CATALOG))

    everything = await service.search("phone")
    assert [r.id for r in everything] == [3, 2, 1]
    assert everything[2].match == "long text"

    filtered = await service.search(
        "phone", SearchFilters(max_price=Decimal("600"), in_stock_only=True, category_id=4), limit=2
    )
    assert [r.id for r in filtered] == [1]
    assert hybrid.calls[-1] == ("phone", 8, 4)


async def test_similar_excludes_the_product_itself():
    service = SearchService(FakeHybrid([hit(1), hit(2), hit(3)]), FakeSpring(CATALOG))
    assert [r.id for r in await service.similar(2)] == [1, 3]


# --- endpoint -------------------------------------------------------------------------------


class FakeServices:
    def __init__(self, searcher):
        self.searcher = searcher
        self.settings = None


def client(searcher):
    app = FastAPI()
    app.include_router(search_api.router)
    app.dependency_overrides[get_services] = lambda: FakeServices(searcher)
    return TestClient(app)


def test_search_endpoint_requires_token_and_returns_items():
    service = SearchService(FakeHybrid([hit(1)]), FakeSpring(CATALOG))
    api = client(service)

    assert api.post("/search", json={"query": "phone"}).status_code == 401
    response = api.post("/search", json={"query": " phone ", "filters": {"max_price": 200}}, headers=TOKEN)

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "phone"
    assert [i["id"] for i in body["items"]] == [1]
    assert body["items"][0]["price"] == "100"


def test_search_endpoint_503_without_embedder_and_validates_input():
    api = client(None)
    assert api.post("/search", json={"query": "phone"}, headers=TOKEN).status_code == 503
    assert api.post("/search", json={"query": "x"}, headers=TOKEN).status_code == 422
