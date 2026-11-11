"""LLM client for any OpenAI-compatible chat API, with validated Pydantic outputs.

Default: Google Gemini's free tier through its OpenAI-compatible endpoint
(https://generativelanguage.googleapis.com/v1beta/openai/). Because it only speaks the
OpenAI chat format, the same code runs on other free/cheap providers by changing
LLM_BASE_URL + LLM_MODEL (Groq, OpenRouter, a local Ollama, ...), as long as the model
supports JSON-schema structured output.
"""

import logging
from dataclasses import dataclass

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
