"""The chat agent: a tool-use loop that streams events (PLAN.md §4).

Per turn: load the session → up to MAX_TOOL_ROUNDS model calls, each streaming its text
and possibly asking for tools → run the tools (their cards/tables/buttons go straight to
the browser) → final answer → save the turn.
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.agent.events import Event
from app.agent.memory import ChatMemory, Session
from app.agent.prompts import SYSTEM_PROMPT, page_context
from app.agent.tools import ToolContext, run_tool, tool_schemas
from app.clients.llm import LLM, ChatTurn, LLMError, LLMUnavailable

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5
MAX_TOOL_CALLS_PER_ROUND = 4
# Tools whose product cards are worth remembering for "the first one", "the cheaper one".
_REMEMBERED_PRODUCT_EVENTS = {"products", "comparison", "action"}


@dataclass(frozen=True)
class ChatRequest:
    message: str
    session_id: uuid.UUID | None = None
    user_id: int | None = None
    page_product_id: int | None = None


class ChatAgent:
    def __init__(self, llm: LLM, memory: ChatMemory, services):
        self._llm = llm
        self._memory = memory
        self._services = services

    async def _system_messages(self, request: ChatRequest) -> list[dict]:
        # One system message: not every OpenAI-compatible provider accepts several.
        system = SYSTEM_PROMPT
        if request.page_product_id is not None:
            try:
                product = await self._services.spring.get_product(request.page_product_id)
            except Exception:
                log.warning("page product %s unavailable", request.page_product_id, exc_info=True)
                product = None
            context = page_context(product)
            if context:
                system += "\n\n" + context
        return [{"role": "system", "content": system}]

    async def run(self, request: ChatRequest) -> AsyncIterator[Event]:
        started = time.monotonic()
        try:
            session = await self._memory.open(request.session_id, request.user_id)
        except Exception:
            log.exception("chat session unavailable")
            yield Event("error", {"message": "The assistant is unavailable right now. Please try again."})
            return
        yield Event("session", {"session_id": str(session.id), "new": session.is_new})

        messages = await self._system_messages(request)
        messages += session.history
        messages.append({"role": "user", "content": request.message})
        ctx = ToolContext(self._services, request.user_id)

        answer: list[str] = []
        shown: dict[int, dict] = {}
        tokens_in = tokens_out = 0
        try:
            for round_number in range(MAX_TOOL_ROUNDS + 1):
                # The last round has no tools, which forces a written answer.
                tools = tool_schemas() if round_number < MAX_TOOL_ROUNDS else None
                turn: ChatTurn | None = None
                async for part in self._llm.stream_chat(messages, tools=tools):
                    if isinstance(part, ChatTurn):
                        turn = part
                    else:
                        answer.append(part)
                        yield Event("token", {"text": part})
                assert turn is not None
                tokens_in += turn.input_tokens
                tokens_out += turn.output_tokens
                if not turn.tool_calls:
                    break

                calls = turn.tool_calls[:MAX_TOOL_CALLS_PER_ROUND]
                turn.tool_calls = calls
                messages.append(turn.to_message())
                if turn.content and not turn.content.endswith(("\n", " ")):
                    answer.append("\n\n")
                    yield Event("token", {"text": "\n\n"})
                for call in calls:
                    yield Event("tool", {"name": call.name})
                    result = await run_tool(ctx, call.name, call.arguments)
                    for event in result.events:
                        if event.name == "products":
                            # A product looked up again in the same answer isn't shown twice.
                            fresh = [c for c in event.data.get("items", []) if c["id"] not in shown]
                            if not fresh:
                                continue
                            event = Event("products", {**event.data, "items": fresh})
                        if event.name in _REMEMBERED_PRODUCT_EVENTS:
                            for card in _cards(event):
                                shown.setdefault(card["id"], {"id": card["id"], "name": card["name"]})
                        yield event
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": result.text})
        except LLMUnavailable:
            log.warning("LLM unavailable during chat", exc_info=True)
            yield Event(
                "error", {"message": "The assistant is busy right now. Please try again in a minute."}
            )
            return
        except LLMError:
            log.warning("LLM error during chat", exc_info=True)
            yield Event("error", {"message": "The assistant couldn't answer this time. Please try again."})
            return

        text = "".join(answer).strip()
        if not text:
            text = "Sorry, I couldn't put an answer together. Could you rephrase the question?"
            yield Event("token", {"text": text})
        latency_ms = int((time.monotonic() - started) * 1000)
        message_id = None
        try:
            message_id = await self._memory.save_turn(
                session, request.message, text, list(shown.values()), tokens_in, tokens_out, latency_ms
            )
        except Exception:
            log.exception("saving chat turn failed")
        yield Event("done", {"message_id": message_id})


def _cards(event: Event) -> list[dict]:
    if event.name == "products":
        return event.data.get("items", [])
    if event.name == "comparison":
        return event.data.get("products", [])
    if event.name == "action":
        return [event.data["product"]] if event.data.get("product") else []
    return []


__all__ = ["ChatAgent", "ChatRequest", "Session"]
