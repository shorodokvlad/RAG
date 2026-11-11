"""Chat sessions and history in ai.chat_sessions / ai.chat_messages.

Only the user's messages and the assistant's final answers are kept (tool results are
re-fetched when needed, so prices are never stale). The ids of products shown in a turn
are kept too, so "compare the first two" still works a few turns later.
"""

import json
import uuid
from dataclasses import dataclass, field

import asyncpg

HISTORY_TURNS = 10


@dataclass
class Session:
    id: uuid.UUID
    user_id: int | None
    history: list[dict] = field(default_factory=list)  # OpenAI-style user/assistant messages
    is_new: bool = False


def history_message(role: str, content: dict) -> dict:
    text = content.get("text", "")
    shown = content.get("products") or []
    if role == "assistant" and shown:
        notes = ", ".join(f"#{p['id']} {p['name']}" for p in shown)
        text = f"{text}\n[Products shown, in order: {notes}]".strip()
    return {"role": role, "content": text}


class ChatMemory:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def open(self, session_id: uuid.UUID | None, user_id: int | None) -> Session:
        """Continue a session, or start a new one.

        A guest session is claimed by the user who logs in during it. A session that belongs
        to another user is never reused: the caller gets a fresh session instead.
        """
        async with self._pool.acquire() as conn:
            if session_id is not None:
                row = await conn.fetchrow("SELECT user_id FROM ai.chat_sessions WHERE id = $1", session_id)
                owner_ok = row is not None and (row["user_id"] is None or row["user_id"] == user_id)
                if owner_ok:
                    if row["user_id"] is None and user_id is not None:
                        await conn.execute(
                            "UPDATE ai.chat_sessions SET user_id = $2 WHERE id = $1 AND user_id IS NULL",
                            session_id,
                            user_id,
                        )
                    rows = await conn.fetch(
                        """
                        SELECT role, content FROM (
                            SELECT id, role, content FROM ai.chat_messages
                            WHERE session_id = $1 AND role IN ('user', 'assistant')
                            ORDER BY id DESC LIMIT $2
                        ) recent ORDER BY id
                        """,
                        session_id,
                        HISTORY_TURNS * 2,
                    )
                    history = [history_message(r["role"], _json(r["content"])) for r in rows]
                    # The model expects the history to start with a user message.
                    while history and history[0]["role"] != "user":
                        history.pop(0)
                    return Session(session_id, user_id, history)

            new_id = uuid.uuid4()
            await conn.execute("INSERT INTO ai.chat_sessions (id, user_id) VALUES ($1, $2)", new_id, user_id)
            return Session(new_id, user_id, is_new=True)

    async def save_turn(
        self,
        session: Session,
        question: str,
        answer: str,
        products: list[dict],
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
    ) -> int:
        """Store the question and the answer; returns the answer's message id (for feedback)."""
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO ai.chat_messages (session_id, role, content) VALUES ($1, 'user', $2::jsonb)",
                session.id,
                json.dumps({"text": question}, ensure_ascii=False),
            )
            message_id = await conn.fetchval(
                """
                INSERT INTO ai.chat_messages (session_id, role, content, tokens_in, tokens_out, latency_ms)
                VALUES ($1, 'assistant', $2::jsonb, $3, $4, $5)
                RETURNING id
                """,
                session.id,
                json.dumps({"text": answer, "products": products}, ensure_ascii=False),
                input_tokens,
                output_tokens,
                latency_ms,
            )
            await conn.execute("UPDATE ai.chat_sessions SET last_at = now() WHERE id = $1", session.id)
        return message_id

    async def add_feedback(
        self, session_id: uuid.UUID, message_id: int, rating: int, comment: str | None
    ) -> bool:
        """Thumbs up/down on an answer of this session; False if the message isn't in it."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                INSERT INTO ai.feedback (message_id, feature, target_id, rating, comment)
                SELECT m.id, 'chat', NULL, $3, $4 FROM ai.chat_messages m
                WHERE m.id = $2 AND m.session_id = $1 AND m.role = 'assistant'
                """,
                session_id,
                message_id,
                rating,
                comment,
            )
        return result.endswith(" 1")


def _json(value) -> dict:
    return json.loads(value) if isinstance(value, str) else (value or {})
