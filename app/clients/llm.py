"""LLM client for any OpenAI-compatible chat API, with validated Pydantic outputs.

Default: Google Gemini's free tier through its OpenAI-compatible endpoint
(https://generativelanguage.googleapis.com/v1beta/openai/). Because it only speaks the
OpenAI chat format, the same code runs on other free/cheap providers by changing
LLM_BASE_URL + LLM_MODEL (Groq, OpenRouter, a local Ollama, ...), as long as the model
supports JSON-schema structured output.
"""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import openai
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)


class LLMError(Exception):
    """The model didn't produce a usable answer."""


class LLMRefusal(LLMError):
    pass


class LLMTruncated(LLMError):
    pass


class LLMUnavailable(LLMError):
    """Quota exhausted or provider down even after retries (free tiers hit this)."""


@dataclass(frozen=True)
class LLMResult[T: BaseModel]:
    output: T
    model: str
    input_tokens: int
    output_tokens: int


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text, as the model wrote it
    # Provider fields to send back unchanged with this call. Gemini puts the "thought
    # signature" here (extra_content.google.thought_signature) and rejects a follow-up
    # request that drops it.
    extra: dict = field(default_factory=dict)

    def to_message(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
            **self.extra,
        }


@dataclass
class ChatTurn:
    """One model response, assembled from the stream."""

    content: str
    tool_calls: list[ToolCall]
    finish_reason: str | None
    input_tokens: int = 0
    output_tokens: int = 0

    def to_message(self) -> dict:
        message: dict = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            message["tool_calls"] = [call.to_message() for call in self.tool_calls]
        return message


def _map_errors(exc: Exception) -> LLMError:
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)):
        return LLMUnavailable(str(exc))
    return LLMError(str(exc))


class LLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        client: openai.AsyncOpenAI | None = None,
    ):
        self.model = model
        # The SDK retries 408/409/429/5xx with backoff; free tiers return 429 when busy.
        self._client = client or openai.AsyncOpenAI(
            api_key=api_key, base_url=base_url, max_retries=4, timeout=90.0
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def parse[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        output: type[T],
        max_tokens: int = 4000,
        temperature: float = 0.2,
    ) -> LLMResult[T]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            completion = await self._client.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=output,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except openai.LengthFinishReasonError as exc:
            raise LLMTruncated(f"output hit max_tokens={max_tokens}") from exc
        except openai.ContentFilterFinishReasonError as exc:
            raise LLMRefusal("provider content filter blocked the answer") from exc
        except ValidationError as exc:
            raise LLMError(f"model output didn't match {output.__name__}: {exc}") from exc
        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as exc:
            raise LLMUnavailable(str(exc)) from exc

        message = completion.choices[0].message
        if message.refusal:
            raise LLMRefusal(message.refusal)
        if message.parsed is None:
            raise LLMError("no structured output in the response")

        usage = completion.usage
        result = LLMResult(
            output=message.parsed,
            model=completion.model or self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
        log.info(
            "llm %s → %s: in=%s out=%s",
            output.__name__,
            result.model,
            result.input_tokens,
            result.output_tokens,
        )
        return result

    async def stream_chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        max_tokens: int = 1500,
        temperature: float = 0.3,
    ) -> AsyncIterator[str | ChatTurn]:
        """Stream one model response: yields text deltas (str) as they arrive, then the ChatTurn.

        Tool-call arguments arrive in pieces; they're assembled by index and returned in the
        final ChatTurn, never as text.
        """
        kwargs: dict = {}
        if tools:
            kwargs["tools"] = tools
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs,
            )
            text: list[str] = []
            calls: dict[int, ToolCall] = {}
            finish_reason = None
            input_tokens = output_tokens = 0
            async for chunk in stream:
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens or input_tokens
                    output_tokens = chunk.usage.completion_tokens or output_tokens
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                finish_reason = choice.finish_reason or finish_reason
                delta = choice.delta
                if delta is None:
                    continue
                if delta.content:
                    text.append(delta.content)
                    yield delta.content
                for part in delta.tool_calls or []:
                    index = part.index if part.index is not None else len(calls)
                    call = calls.setdefault(index, ToolCall(id="", name="", arguments=""))
                    if part.id:
                        call.id = part.id
                    if part.function is not None:
                        if part.function.name:
                            call.name += part.function.name
                        if part.function.arguments:
                            call.arguments += part.function.arguments
                    if part.model_extra:
                        call.extra.update(part.model_extra)
        except openai.APIError as exc:
            raise _map_errors(exc) from exc

        tool_calls = []
        for index, call in sorted(calls.items()):
            if not call.name:
                continue
            call.id = call.id or f"call_{index}"
            tool_calls.append(call)
        turn = ChatTurn("".join(text), tool_calls, finish_reason, input_tokens, output_tokens)
        log.info(
            "llm chat → %s: tools=%s finish=%s in=%s out=%s",
            self.model,
            [c.name for c in tool_calls],
            finish_reason,
            input_tokens,
            output_tokens,
        )
        yield turn
