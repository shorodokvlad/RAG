"""ai schema: chunks (pgvector), generated summaries, chat history, feedback, knowledge base

Revision ID: 0001
Revises:
Create Date: 2026-11-11
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1024  # voyage-3.5 with output_dimension=1024 (see app/config.py)


def upgrade() -> None:
    # On Supabase you can also enable it in Dashboard → Database → Extensions; this is then a no-op.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(f"""
        CREATE TABLE ai.chunks (
            id            BIGSERIAL PRIMARY KEY,
            source_type   TEXT NOT NULL CHECK (source_type IN
                              ('product_card', 'product_spec', 'product_desc', 'review', 'faq')),
            source_id     BIGINT NOT NULL,
            product_id    BIGINT,
            category_id   BIGINT,
            chunk_index   INT NOT NULL DEFAULT 0,
            content       TEXT NOT NULL,
            content_hash  TEXT NOT NULL,
            metadata      JSONB NOT NULL DEFAULT '{{}}',
            embedding     VECTOR({EMBEDDING_DIM}) NOT NULL,
            tsv           TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
            updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (source_type, source_id, chunk_index)
        )
    """)
    op.execute("CREATE INDEX chunks_embedding_hnsw ON ai.chunks USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX chunks_tsv_gin ON ai.chunks USING gin (tsv)")
    op.execute("CREATE INDEX chunks_product_id ON ai.chunks (product_id)")
    op.execute("CREATE INDEX chunks_category_id ON ai.chunks (category_id)")
    op.execute("CREATE INDEX chunks_source_type ON ai.chunks (source_type)")

    op.execute("""
        CREATE TABLE ai.product_overviews (
            product_id    BIGINT PRIMARY KEY,
            lang          TEXT NOT NULL DEFAULT 'ro',
            overview      JSONB NOT NULL,
            source_hash   TEXT NOT NULL,
            model         TEXT NOT NULL,
            generated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE ai.review_summaries (
            product_id    BIGINT PRIMARY KEY,
            summary       JSONB NOT NULL,
            review_count  INT NOT NULL,
            reviews_hash  TEXT NOT NULL,
            model         TEXT NOT NULL,
            generated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE ai.chat_sessions (
            id          UUID PRIMARY KEY,
            user_id     BIGINT,
            summary     TEXT,
            started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_at     TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX chat_sessions_user_id ON ai.chat_sessions (user_id)")

    op.execute("""
        CREATE TABLE ai.chat_messages (
            id          BIGSERIAL PRIMARY KEY,
            session_id  UUID NOT NULL REFERENCES ai.chat_sessions(id) ON DELETE CASCADE,
            role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
            content     JSONB NOT NULL,
            tokens_in   INT,
            tokens_out  INT,
            latency_ms  INT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX chat_messages_session ON ai.chat_messages (session_id, id)")

    op.execute("""
        CREATE TABLE ai.feedback (
            id          BIGSERIAL PRIMARY KEY,
            message_id  BIGINT REFERENCES ai.chat_messages(id) ON DELETE CASCADE,
            feature     TEXT NOT NULL CHECK (feature IN ('chat', 'overview', 'review_summary', 'search')),
            target_id   BIGINT,
            rating      SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
            comment     TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE ai.kb_documents (
            id          BIGSERIAL PRIMARY KEY,
            slug        TEXT NOT NULL UNIQUE,
            title       TEXT NOT NULL,
            body        TEXT NOT NULL,
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)


def downgrade() -> None:
    for table in (
        "kb_documents",
        "feedback",
        "chat_messages",
        "chat_sessions",
        "review_summaries",
        "product_overviews",
        "chunks",
    ):
        op.execute(f"DROP TABLE IF EXISTS ai.{table}")
    # The vector extension and the ai schema (with alembic_version) are left in place.
