from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import index as index_api
from app.api import search as search_api
from app.indexing.indexer import KbStats
from app.indexing.kb import KB_DIR, build_faq_chunks, load_documents, parse_document
from app.retrieval.hybrid import KbPassage, fuse_passages
from app.services import get_services

TOKEN = {"X-Service-Token": "test-token"}

DOC = """# Returns and cancellations

Intro text before any section.

## Can I return a product?
Yes, within 14 days.

## Empty section

## When do I get my money back?
Within 14 days.
"""


def test_parse_document_takes_title_and_splits_sections():
    doc = parse_document("returns", DOC)

    assert doc.title == "Returns and cancellations"
    assert [(s.heading, s.body) for s in doc.sections] == [
        ("", "Intro text before any section."),
        ("Can I return a product?", "Yes, within 14 days."),
        ("When do I get my money back?", "Within 14 days."),
    ]


def test_parse_document_without_heading_uses_slug():
    assert parse_document("store-hours", "Open daily.").title == "Store hours"


def test_faq_chunks_are_prefixed_with_title_and_heading():
    chunks = build_faq_chunks(7, parse_document("returns", DOC))

    assert [c.key for c in chunks] == [("faq", 7, 0), ("faq", 7, 1), ("faq", 7, 2)]
    assert chunks[1].content == "Returns and cancellations: Can I return a product?\nYes, within 14 days."
    assert chunks[0].content.startswith("Returns and cancellations\n")
    assert chunks[1].product_id is None
    assert chunks[1].metadata == {
        "slug": "returns",
        "title": "Returns and cancellations",
        "heading": "Can I return a product?",
    }


def test_load_documents_skips_readme_and_bad_names(tmp_path):
    (tmp_path / "README.md").write_text("# About")
    (tmp_path / "Delivery.md").write_text("# Delivery\n## Where?\nRomania.")
    (tmp_path / "bad name.md").write_text("# Bad")
    (tmp_path / "notes.txt").write_text("ignored")

    docs = load_documents(tmp_path)

    assert [d.slug for d in docs] == ["delivery"]
    assert load_documents(tmp_path / "missing") == []


def test_shipped_knowledge_base_is_well_formed():
    docs = load_documents(KB_DIR)
    slugs = {d.slug for d in docs}

    assert {"contact", "delivery", "ordering", "payment", "returns", "warranty"} <= slugs
    for doc in docs:
        assert doc.title and doc.sections, doc.slug
        assert all(s.heading for s in doc.sections), f"{doc.slug}: text outside a ## section"


def test_fuse_passages_ranks_chunks_and_splits_title():
    by_vector = [
        {"id": 1, "content": "Delivery: Where?\nRomania and the EU.", "metadata": '{"slug": "delivery"}'},
        {"id": 2, "content": "Returns: Can I return?\nYes.", "metadata": {"slug": "returns"}},
    ]
    by_keyword = [{"id": 2, "content": "Returns: Can I return?\nYes.", "metadata": {"slug": "returns"}}]

    passages = fuse_passages([by_vector, by_keyword], limit=5)

    assert [(p.id, p.slug, p.title, p.content) for p in passages] == [
        (2, "returns", "Returns: Can I return?", "Yes."),
        (1, "delivery", "Delivery: Where?", "Romania and the EU."),
    ]


# --- endpoints ------------------------------------------------------------------------------


class FakeHybrid:
    async def search_kb(self, query, limit):
        return [KbPassage(id=1, title="Returns: Can I return?", slug="returns", content="Yes.", score=0.03)]


class FakeIndexer:
    def __init__(self):
        self.documents = None

    async def index_kb(self, documents):
        self.documents = documents
        return KbStats(len(documents), 0, 10, 10, 0, 0)


class FakeServices:
    def __init__(self, indexer=None):
        self.searcher = type("Searcher", (), {"hybrid": FakeHybrid()})()
        self.indexer = indexer
        self.embedder = object()
        self.settings = None


def make_client(services, router):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_services] = lambda: services
    return TestClient(app)


def test_kb_search_endpoint():
    api = make_client(FakeServices(), search_api.router)

    response = api.post("/search/kb", json={"query": "return policy"}, headers=TOKEN)

    assert response.status_code == 200
    assert response.json()[0]["slug"] == "returns"


def test_kb_sync_endpoint_indexes_the_shipped_documents():
    indexer = FakeIndexer()
    api = make_client(FakeServices(indexer), index_api.router)

    response = api.post("/index/kb", headers=TOKEN)

    assert response.status_code == 200
    assert response.json()["status"] == "synced"
    assert {d.slug for d in indexer.documents} >= {"returns", "delivery"}
