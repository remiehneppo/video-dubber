from __future__ import annotations

from pathlib import Path

import pytest

from dubber.pipeline.job_manager import JobManager


def test_plan_resume_missing_job_does_not_create_workspace_directories(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"

    with pytest.raises(FileNotFoundError):
        JobManager().plan_resume(workspace, "job_missing")

    assert not (workspace / "job_missing").exists()
