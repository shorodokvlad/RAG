import httpx
from tenacity import retry_if_exception, stop_after_attempt, wait_exponential


class RetryableStatusError(Exception):
    def __init__(self, response: httpx.Response):
        super().__init__(f"{response.request.method} {response.request.url} -> {response.status_code}")
        self.response = response


RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def raise_for_status(response: httpx.Response) -> None:
    """Raise RetryableStatusError for transient codes, httpx.HTTPStatusError for the rest."""
    if response.status_code in RETRYABLE_STATUS:
        raise RetryableStatusError(response)
    response.raise_for_status()


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, RetryableStatusError | httpx.TransportError)


def retry_policy(attempts: int = 4) -> dict:
    """kwargs for tenacity's AsyncRetrying: exponential backoff on transient HTTP failures."""
    return {
        "retry": retry_if_exception(_is_retryable),
        "stop": stop_after_attempt(attempts),
        "wait": wait_exponential(multiplier=1, min=1, max=20),
        "reraise": True,
    }
