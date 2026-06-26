from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx


class ProviderRequestError(RuntimeError):
    def __init__(self, *, provider: str, attempts: int, cause: Exception) -> None:
        super().__init__(f"{provider} request failed after {attempts} attempts: {cause}")
        self.provider = provider
        self.attempts = attempts
        self.cause = cause


async def request_with_retries(
    request: Callable[[], Awaitable[httpx.Response]],
    *,
    provider: str,
    max_attempts: int = 5,
    retry_delay_sec: float = 0.5,
) -> httpx.Response:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = await request()
            response.raise_for_status()
            return response
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            last_error = exc
            if attempt == max_attempts or not _is_retryable(exc):
                raise ProviderRequestError(provider=provider, attempts=attempt, cause=exc) from exc
            if retry_delay_sec > 0:
                await asyncio.sleep(retry_delay_sec)

    raise ProviderRequestError(
        provider=provider,
        attempts=max_attempts,
        cause=last_error or RuntimeError("request did not complete"),
    )


def _is_retryable(exc: httpx.HTTPStatusError | httpx.TransportError) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return exc.response.status_code == 429 or exc.response.status_code >= 500
