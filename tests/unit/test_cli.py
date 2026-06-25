from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli import main
from dubber.core.enums import StageName, StageStatus
from dubber.core.paths import WorkspacePaths
from dubber.orchestrator.checkpoint_store import CheckpointStore


def test_jobs_lists_existing_job_ids(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workspace = tmp_path / "workspace"
    paths = WorkspacePaths.create(workspace, "job_one")
    CheckpointStore.create(paths.job_state_file, job_id="job_one", input_file=Path("input/a.mp4")).save()

    exit_code = main(["jobs", "--workspace", str(workspace)])

    assert exit_code == 0
    assert "job_one" in capsys.readouterr().out


def test_status_prints_job_state_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workspace = tmp_path / "workspace"
    paths = WorkspacePaths.create(workspace, "job_status")
    store = CheckpointStore.create(paths.job_state_file, job_id="job_status", input_file=Path("input/a.mp4"))
    store.mark_stage(StageName.JOB_INIT, StageStatus.COMPLETED, artifact="input_metadata.v1.json")
    store.save()

    exit_code = main(["status", "--workspace", str(workspace), "--job", "job_status"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["job_id"] == "job_status"
    assert payload["stages"]["job_init"]["status"] == "completed"


def test_validate_returns_nonzero_for_missing_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workspace = tmp_path / "workspace"
    paths = WorkspacePaths.create(workspace, "job_validate")
    CheckpointStore.create(paths.job_state_file, job_id="job_validate", input_file=Path("input/a.mp4")).save()

    exit_code = main(["validate", "--workspace", str(workspace), "--job", "job_validate"])

    assert exit_code == 1
    assert "manifest.json missing" in capsys.readouterr().out


def test_run_reports_input_error_for_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--input", "video.mp4"]) == 1
    assert "Input video does not exist" in capsys.readouterr().out


def test_resume_reports_missing_job_state(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["resume", "--job", "job_123"]) == 1
    assert "resume failed" in capsys.readouterr().out


def test_run_openai_compatible_mode_reports_missing_provider_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "--input", "video.mp4", "--provider-mode", "openai_compatible"]) == 1
    out = capsys.readouterr().out
    assert "provider config invalid" in out or "Input video does not exist" in out
