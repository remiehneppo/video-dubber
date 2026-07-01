from __future__ import annotations

import asyncio
import email.utils
import logging
import random
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)


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
                logger.error("%s request failed attempt=%s/%s retryable=%s error=%s", provider, attempt, max_attempts, _is_retryable(exc), exc)
                raise ProviderRequestError(provider=provider, attempts=attempt, cause=exc) from exc
            delay_sec = _retry_delay_sec(exc, attempt=attempt, base_delay_sec=retry_delay_sec)
            logger.warning("%s request failed attempt=%s/%s; retrying in %ss: %s", provider, attempt, max_attempts, delay_sec, exc)
            if delay_sec > 0:
                await asyncio.sleep(delay_sec)

    raise ProviderRequestError(
        provider=provider,
        attempts=max_attempts,
        cause=last_error or RuntimeError("request did not complete"),
    )


def _is_retryable(exc: httpx.HTTPStatusError | httpx.TransportError) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return exc.response.status_code == 429 or exc.response.status_code >= 500


def _retry_delay_sec(
    exc: httpx.HTTPStatusError | httpx.TransportError,
    *,
    attempt: int,
    base_delay_sec: float,
) -> float:
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return retry_after
    if base_delay_sec <= 0:
        return 0
    backoff = min(base_delay_sec * (2 ** max(0, attempt - 1)), 30.0)
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return min(30.0, backoff + random.uniform(0.0, backoff * 0.25))
    return backoff


def _retry_after_seconds(exc: httpx.HTTPStatusError | httpx.TransportError) -> float | None:
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    value = exc.response.headers.get("retry-after")
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, min(float(value), 120.0))
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, min((parsed - datetime.now(timezone.utc)).total_seconds(), 120.0))
