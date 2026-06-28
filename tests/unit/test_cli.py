from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli import main
from dubber.core.enums import StageName, StageStatus
from dubber.core.io import write_json_atomic
from dubber.core.paths import WorkspacePaths
from dubber.orchestrator.checkpoint_store import CheckpointStore
from dubber.pipeline.job_manager import JobManager


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


def test_job_manager_loads_resolved_config_path_for_resume(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    paths = WorkspacePaths.create(workspace, "job_config")
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  domain: coding",
                "mixing:",
                "  original_ducking_db: -31",
                "  tts_boost_db: 11",
                "  final_loudness_normalization: false",
            ]
        ),
        encoding="utf-8",
    )
    write_json_atomic(
        paths.root / "config.resolved.json",
        {
            "provider_mode": "mock",
            "domain": "ai",
            "glossary_review": False,
            "config_path": str(config_path),
        },
    )

    config = JobManager()._load_resolved_config(paths)

    assert config.project.domain == "ai"
    assert config.mixing.original_ducking_db == -31.0
    assert config.mixing.tts_boost_db == 11.0
    assert config.mixing.final_loudness_normalization is False
