"""Chat assistant endpoints (F1). Spring's /ai/chat gateway streams these to the browser.

POST /chat answers with Server-Sent Events: session, token, tool, products, comparison,
action, sources, then done (or error). user_id is set by Spring from the shopper's JWT;
it's trusted because only Spring has the service token.
"""

import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.events import Event
from app.agent.loop import ChatRequest
from app.agent.memory import ChatMemory
from app.security import require_service_token
from app.services import Services, get_services, require_chat_agent

router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(require_service_token)])

ServicesDep = Annotated[Services, Depends(get_services)]

MAX_MESSAGE_CHARS = 1000


class PageContext(BaseModel):
    product_id: int | None = Field(default=None, ge=1)


class ChatBody(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    session_id: uuid.UUID | None = None
    user_id: int | None = Field(default=None, ge=1)
    page: PageContext = Field(default_factory=PageContext)


class FeedbackBody(BaseModel):
    session_id: uuid.UUID
    message_id: int = Field(ge=1)
    rating: Literal[1, -1]
    comment: str | None = Field(default=None, max_length=500)


async def _sse(events: AsyncIterator[Event]) -> AsyncIterator[str]:
    async for event in events:
        yield event.to_sse()


@router.post("")
async def chat(body: ChatBody, services: ServicesDep) -> StreamingResponse:
    agent = require_chat_agent(services)
    message = body.message.strip()
    if not message:
        raise HTTPException(422, "The message is empty")
    request = ChatRequest(
        message=message,
        session_id=body.session_id,
        user_id=body.user_id,
        page_product_id=body.page.product_id,
    )
    return StreamingResponse(
        _sse(agent.run(request)),
        media_type="text/event-stream",
        # No caching or proxy buffering: tokens must reach the browser as they're written.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def feedback(body: FeedbackBody, services: ServicesDep) -> None:
    saved = await ChatMemory(services.pool).add_feedback(
        body.session_id, body.message_id, body.rating, body.comment
    )
    if not saved:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Message not found in this session")
