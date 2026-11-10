from app.indexing.chunkers import build_product_chunks, normalize_text, split_words
from tests.factories import make_product, make_review


def by_type(chunks, source_type):
    return [c for c in chunks if c.source_type == source_type]


def test_product_with_json_specs():
    chunks = build_product_chunks(make_product(), [make_review()])

    card = by_type(chunks, "product_card")[0]
    assert card.content.startswith("iPhone 16\nCategory: Phones")
    assert "Options: Black / 128GB, Blue / 256GB, Black / 256GB" in card.content
    assert "Attributes: Color: Black, Blue; Storage: 128GB, 256GB" in card.content
    assert "Key specs: Operating System: iOS; Connectivity: 5G; Screen Size: 6.1 inch" in card.content
    assert card.metadata["average_rating"] == 4.5

    specs = by_type(chunks, "product_spec")
    assert [c.content.splitlines()[0] for c in specs] == [
        "iPhone 16 — GENERAL CHARACTERISTICS",
        "iPhone 16 — DISPLAY",
    ]
    assert [c.chunk_index for c in specs] == [0, 1]

    # The JSON itself must never be indexed as prose.
    assert by_type(chunks, "product_desc") == []
    assert all("{" not in c.content for c in chunks)


def test_no_prices_or_stock_in_chunks():
    chunks = build_product_chunks(make_product(), [make_review()])
    assert all("4999" not in c.content for c in chunks)
    assert all("price" not in c.metadata and "stock" not in c.metadata for c in chunks)


def test_reviews_are_prefixed_and_carry_rating_without_pii():
    review = by_type(build_product_chunks(make_product(), [make_review(rating=4)]), "review")[0]

    assert review.content == "Review of iPhone 16 (4/5):\nGreat battery life"
    assert review.source_id == 100 and review.product_id == 12
    assert review.metadata == {
        "product_name": "iPhone 16",
        "category": "Phones",
        "rating": 4,
        "created_at": "2026-11-01T12:00:00",
    }


def test_blank_reviews_are_skipped():
    assert by_type(build_product_chunks(make_product(), [make_review(content="  \n ")]), "review") == []


def test_prose_description_is_chunked_with_overlap():
    prose = " ".join(f"w{i}" for i in range(600))
    chunks = by_type(build_product_chunks(make_product(description=prose)), "product_desc")

    assert len(chunks) == 3
    assert all(c.content.startswith("iPhone 16 — description\n") for c in chunks)
    first, second = (c.content.split("\n", 1)[1].split() for c in chunks[:2])
    assert first[-40:] == second[:40]


def test_hash_ignores_metadata_but_tracks_content():
    a = build_product_chunks(make_product())[0]
    b = build_product_chunks(make_product(average_rating="3.0"))[0]
    c = build_product_chunks(make_product(name="iPhone 16 Pro"))[0]

    assert a.content_hash == b.content_hash
    assert a.metadata != b.metadata
    assert a.content_hash != c.content_hash


def test_split_words_and_normalize():
    assert split_words("", 10, 2) == []
    assert split_words("a b c", 10, 2) == ["a b c"]
    assert normalize_text("a\x00b   c\n\n  d ") == "ab c\nd"
