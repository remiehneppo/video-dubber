from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import BoundedSemaphore
from typing import Generic, TypeVar

from dubber.core.models import RuntimeConfig


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class TaskCompletion(Generic[T, R]):
    index: int
    item: T
    result: R | None = None
    error: BaseException | None = None


class ProviderConcurrency:
    """Process-local request limits shared by every job in a CLI invocation."""

    def __init__(self, runtime: RuntimeConfig) -> None:
        for name in ("asr_concurrency", "llm_concurrency", "tts_concurrency"):
            value = getattr(runtime, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"runtime.{name} must be an integer >= 1, got {value!r}")
        self.asr_limit = runtime.asr_concurrency
        self.llm_limit = runtime.llm_concurrency
        self.tts_limit = runtime.tts_concurrency
        self._asr = BoundedSemaphore(self.asr_limit)
        self._llm = BoundedSemaphore(self.llm_limit)
        self._tts = BoundedSemaphore(self.tts_limit)

    async def run_asr(self, operation: Callable[[], Awaitable[R]]) -> R:
        return await self._run(self._asr, operation)

    async def run_llm(self, operation: Callable[[], Awaitable[R]]) -> R:
        return await self._run(self._llm, operation)

    async def run_tts(self, operation: Callable[[], Awaitable[R]]) -> R:
        return await self._run(self._tts, operation)

    @staticmethod
    async def _run(semaphore: BoundedSemaphore, operation: Callable[[], Awaitable[R]]) -> R:
        await asyncio.to_thread(semaphore.acquire)
        try:
            return await operation()
        finally:
            semaphore.release()


def validate_threaded_provider_clients(provider_bundle: object) -> None:
    """Reject loop-bound clients that cannot cross per-unit worker event loops."""
    for name in ("asr", "llm", "tts"):
        provider = getattr(provider_bundle, name, None)
        if provider is not None and getattr(provider, "client", None) is not None:
            raise RuntimeError(
                f"{name} provider uses an injected async client, which is not supported "
                "by threaded pipeline execution; let the provider create a client per request"
            )


def run_bounded(
    items: Iterable[T],
    worker: Callable[[T], R],
    *,
    max_workers: int,
    on_completion: Callable[[TaskCompletion[T, R]], None] | None = None,
) -> list[R]:
    """Run a bounded number of tasks and persist completions on the caller thread.

    New work stops being submitted after the first observed failure. Tasks already
    in flight are allowed to finish and all of their completions are reported.
    Successful output is returned in input order.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    indexed = list(enumerate(items))
    if not indexed:
        return []

    results: dict[int, R] = {}
    first_error: BaseException | None = None
    next_index = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Future[R], tuple[int, T]] = {}

        def submit_one() -> bool:
            nonlocal next_index
            if next_index >= len(indexed):
                return False
            index, item = indexed[next_index]
            next_index += 1
            futures[executor.submit(worker, item)] = (index, item)
            return True

        for _ in range(min(max_workers, len(indexed))):
            submit_one()

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            completions: list[TaskCompletion[T, R]] = []
            for future in done:
                index, item = futures.pop(future)
                try:
                    result = future.result()
                except BaseException as exc:
                    completion = TaskCompletion[T, R](index=index, item=item, error=exc)
                    if first_error is None:
                        first_error = exc
                else:
                    results[index] = result
                    completion = TaskCompletion(index=index, item=item, result=result)
                completions.append(completion)

            for completion in sorted(completions, key=lambda value: value.index):
                if on_completion is None:
                    continue
                try:
                    on_completion(completion)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc

            if first_error is None:
                for _ in completions:
                    if not submit_one():
                        break

    if first_error is not None:
        raise first_error
    return [results[index] for index in range(len(indexed))]
