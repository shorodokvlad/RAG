from app.db import to_pgvector
from app.indexing.chunkers import build_product_chunks
from app.indexing.indexer import ExistingChunk, plan_changes, stored_hash
from tests.factories import make_product, make_review

KEY = "gemini-embedding-001:1024"


def existing_from(chunks, key=KEY):
    return [ExistingChunk(i, c.key, stored_hash(c, key)) for i, c in enumerate(chunks, start=1)]


def test_first_index_embeds_everything():
    chunks = build_product_chunks(make_product(), [make_review()])
    plan = plan_changes([], chunks, KEY)
    assert plan.to_embed == chunks and plan.to_touch == [] and plan.to_delete == []


def test_unchanged_chunks_are_touched_changed_embedded_and_missing_deleted():
    old = build_product_chunks(make_product(), [make_review(100), make_review(101, content="Too heavy")])
    existing = existing_from(old)

    # Review 101 deleted, review 100 edited, product card unchanged.
    new = build_product_chunks(make_product(), [make_review(100, content="Battery is ok")])
    plan = plan_changes(existing, new, KEY)

    assert [c.key for c in plan.to_embed] == [("review", 100, 0)]
    assert {c.source_type for c in plan.to_touch} == {"product_card", "product_spec"}
    deleted_keys = {row.key for row in existing if row.id in plan.to_delete}
    assert deleted_keys == {("review", 101, 0)}


def test_switching_embedding_model_re_embeds_everything():
    chunks = build_product_chunks(make_product(), [make_review()])
    plan = plan_changes(existing_from(chunks, "voyage-3.5:1024"), chunks, KEY)
    assert plan.to_embed == chunks and plan.to_touch == []


def test_to_pgvector_format():
    assert to_pgvector([0.5, -1.0, 1e-9]) == "[0.5,-1,1e-09]"
