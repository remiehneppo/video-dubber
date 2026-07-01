from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from dubber.core.concurrency import (
    ProviderConcurrency,
    TaskCompletion,
    run_bounded,
    validate_threaded_provider_clients,
)
from dubber.core.models import RuntimeConfig
from dubber.pipeline.job_manager import BatchManager, BatchOptions, JobManager, RunSummary


def test_run_bounded_respects_limit_and_returns_input_order() -> None:
    lock = threading.Lock()
    in_flight = 0
    peak = 0

    def worker(value: int) -> int:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep((4 - value) * 0.005)
        with lock:
            in_flight -= 1
        return value * 10

    assert run_bounded(range(4), worker, max_workers=2) == [0, 10, 20, 30]
    assert peak == 2


def test_run_bounded_stops_scheduling_after_failure_and_reports_in_flight() -> None:
    barrier = threading.Barrier(2)
    started: list[int] = []
    completions: list[TaskCompletion[int, int]] = []

    def worker(value: int) -> int:
        started.append(value)
        if value < 2:
            barrier.wait(timeout=1)
        if value == 0:
            raise RuntimeError("boom")
        if value == 1:
            time.sleep(0.02)
        return value

    with pytest.raises(RuntimeError, match="boom"):
        run_bounded(range(5), worker, max_workers=2, on_completion=completions.append)

    assert started == [0, 1]
    assert {completion.item for completion in completions} == {0, 1}
    assert next(completion for completion in completions if completion.item == 1).result == 1


def test_provider_concurrency_is_shared_across_threads() -> None:
    limits = ProviderConcurrency(RuntimeConfig(asr_concurrency=2, llm_concurrency=1, tts_concurrency=1))
    lock = threading.Lock()
    in_flight = 0
    peak = 0

    async def request() -> int:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        with lock:
            in_flight -= 1
        return 1

    run_bounded(range(6), lambda _: asyncio.run(limits.run_asr(request)), max_workers=6)

    assert peak == 2


@pytest.mark.parametrize(
    ("method_name", "expected_peak"),
    [("run_asr", 3), ("run_llm", 2), ("run_tts", 4)],
)
def test_provider_concurrency_applies_each_configured_limit(method_name: str, expected_peak: int) -> None:
    limits = ProviderConcurrency(RuntimeConfig(asr_concurrency=3, llm_concurrency=2, tts_concurrency=4))
    lock = threading.Lock()
    in_flight = 0
    peak = 0

    async def request() -> None:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        with lock:
            in_flight -= 1

    limiter = getattr(limits, method_name)
    run_bounded(range(8), lambda _: asyncio.run(limiter(request)), max_workers=8)

    assert peak == expected_peak


def test_provider_concurrency_is_global_across_parallel_jobs_and_retries() -> None:
    limits = ProviderConcurrency(RuntimeConfig(asr_concurrency=2))
    lock = threading.Lock()
    in_flight = 0
    peak = 0

    async def request_with_retry() -> None:
        nonlocal in_flight, peak
        for _ in range(2):
            with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            with lock:
                in_flight -= 1

    def job(_: int) -> None:
        run_bounded(
            range(4),
            lambda __: asyncio.run(limits.run_asr(request_with_retry)),
            max_workers=4,
        )

    run_bounded(range(2), job, max_workers=2)

    assert peak == 2


def test_waiting_for_provider_permit_does_not_block_event_loop() -> None:
    limits = ProviderConcurrency(RuntimeConfig(asr_concurrency=1))
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    ticker_ran = False

    async def first() -> None:
        first_started.set()
        await release_first.wait()

    async def second() -> None:
        return None

    async def run() -> None:
        nonlocal ticker_ran
        first_task = asyncio.create_task(limits.run_asr(first))
        await first_started.wait()
        second_task = asyncio.create_task(limits.run_asr(second))
        await asyncio.sleep(0)
        ticker_ran = True
        release_first.set()
        await asyncio.gather(first_task, second_task)

    asyncio.run(asyncio.wait_for(run(), timeout=1))

    assert ticker_ran is True


def test_threaded_pipeline_rejects_injected_async_clients() -> None:
    bundle = SimpleNamespace(
        asr=SimpleNamespace(client=object()),
        llm=SimpleNamespace(client=None),
        tts=SimpleNamespace(client=None),
    )

    with pytest.raises(RuntimeError, match="asr provider uses an injected async client"):
        validate_threaded_provider_clients(bundle)


def test_batch_manager_shares_one_coordinator_across_job_managers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "one.mp4").write_bytes(b"one")
    (input_dir / "two.mp4").write_bytes(b"two")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "runtime:\n"
        "  max_parallel_jobs: 2\n"
        "  asr_concurrency: 2\n"
        "  llm_concurrency: 1\n"
        "  tts_concurrency: 2\n",
        encoding="utf-8",
    )
    coordinators: list[ProviderConcurrency] = []
    lock = threading.Lock()

    def fake_run(self: JobManager, options, *, stop_after=None) -> RunSummary:
        assert self.concurrency is not None
        with lock:
            coordinators.append(self.concurrency)
        return RunSummary(str(options.job_id), "completed", "", str(options.workspace_dir))

    manager = BatchManager()

    def fake_glossary(root, state, config, jobs, concurrency):
        coordinators.append(concurrency)
        path = root / "artifacts" / "glossary.locked.json"
        path.write_text('{"terms": []}', encoding="utf-8")
        return path

    def fake_complete(root, state, jobs, glossary, max_workers, *, concurrency, **kwargs):
        coordinators.append(concurrency)
        for job in jobs:
            job["status"] = "completed"

    monkeypatch.setattr(JobManager, "run", fake_run)
    monkeypatch.setattr(manager, "_build_shared_glossary", fake_glossary)
    monkeypatch.setattr(manager, "_complete_jobs", fake_complete)

    summary = manager.run(
        BatchOptions(
            input_dir=input_dir,
            workspace_dir=tmp_path / "workspace",
            config_path=config_path,
        )
    )

    assert summary.status == "completed"
    assert len(coordinators) == 4
    assert len({id(coordinator) for coordinator in coordinators}) == 1


def test_batch_job_preserves_waiting_review_status_from_job_manager(tmp_path: Path) -> None:
    manager = BatchManager()
    root = tmp_path / "batch_review"
    root.mkdir()
    job = {
        "job_id": "job_review",
        "status": "ready",
        "error": None,
        "output_video": "",
    }
    state = {"schema_version": "1.0", "batch_id": "batch_review", "jobs": [job]}

    manager._run_jobs(
        state,
        root,
        [job],
        lambda _job: RunSummary("job_review", "waiting_review", "", "workspace/job_review"),
        1,
        success_status="completed",
    )

    assert job["status"] == "waiting_review"


def test_batch_resume_rebuilds_missing_locked_glossary_from_asr_completed_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "batch_resume"
    (root / "artifacts").mkdir(parents=True)
    job_dir = root / "jobs" / "job_one" / "artifacts"
    job_dir.mkdir(parents=True)
    (job_dir / "transcript.v1.json").write_text(
        json.dumps({"schema_version": "1.0", "segments": []}),
        encoding="utf-8",
    )
    (root / "jobs" / "job_one" / "job_state.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "job_id": "job_one",
                "input_file": "input/input.mp4",
                "status": "running",
                "current_stage": "translation",
                "stages": {},
                "created_at": "2026-06-30T00:00:00+00:00",
                "updated_at": "2026-06-30T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "runtime:\n"
        "  max_parallel_jobs: 1\n"
        "  asr_concurrency: 1\n"
        "  llm_concurrency: 1\n"
        "  tts_concurrency: 1\n",
        encoding="utf-8",
    )
    (root / "batch_state.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "batch_id": "batch_resume",
                "status": "running",
                "provider_mode": "mock",
                "glossary_review": False,
                "config_path": str(config_path),
                "domain": "math",
                "jobs": [
                    {
                        "input_file": "input.mp4",
                        "input_name": "input.mp4",
                        "job_id": "job_one",
                        "status": "asr_completed",
                        "error": None,
                        "output_video": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manager = BatchManager()
    rebuilt_jobs: list[list[str]] = []

    def fake_build(root_path, state, config, jobs, concurrency):
        rebuilt_jobs.append([str(job["job_id"]) for job in jobs])
        path = root_path / "artifacts" / "glossary.locked.json"
        path.write_text('{"schema_version":"1.0","terms":[]}', encoding="utf-8")
        return path

    def fake_complete(root_path, state, jobs, glossary, max_workers, *, concurrency, **kwargs):
        assert glossary.name == "glossary.locked.json"
        for job in jobs:
            job["status"] = "completed"

    monkeypatch.setattr(manager, "_build_shared_glossary", fake_build)
    monkeypatch.setattr(manager, "_complete_jobs", fake_complete)
    monkeypatch.setattr(JobManager, "publish_shared_glossary", lambda *args, **kwargs: None)

    summary = manager.resume(workspace, "batch_resume")

    assert rebuilt_jobs == [["job_one"]]
    assert summary.status == "completed"
    assert (root / "artifacts" / "glossary.locked.json").exists()
