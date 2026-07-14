from __future__ import annotations

from pathlib import Path

import pytest

from cli import _review_batch_jobs
from dubber.core.enums import JobStatus
from dubber.core.io import write_json_atomic


def test_batch_review_selected_job_missing_required_artifact_fails(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    batch_root = workspace / "batch_missing_review"
    (batch_root / "jobs" / "job_waiting" / "artifacts").mkdir(parents=True)
    write_json_atomic(
        batch_root / "batch_state.json",
        {
            "schema_version": "1.0",
            "batch_id": "batch_missing_review",
            "status": JobStatus.WAITING_REVIEW.value,
            "jobs": [
                {
                    "job_id": "job_waiting",
                    "input_name": "lesson.mp4",
                    "status": JobStatus.WAITING_REVIEW.value,
                    "error": None,
                }
            ],
        },
    )

    with pytest.raises(FileNotFoundError, match="review.required.json missing"):
        _review_batch_jobs(workspace, "batch_missing_review", ["job_waiting"])
