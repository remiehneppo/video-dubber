from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from dubber.core.enums import JobStatus, StageName, StageStatus
from dubber.core.io import read_json, write_json_atomic
from dubber.core.models import JobState, StageProgress, utc_now_iso


class CheckpointStore:
    def __init__(self, state_path: Path, state: JobState) -> None:
        self.state_path = state_path
        self.state = state

    @classmethod
    def create(cls, state_path: Path, *, job_id: str, input_file: Path) -> CheckpointStore:
        return cls(state_path=state_path, state=JobState.create(job_id=job_id, input_file=input_file))

    @classmethod
    def load(cls, state_path: Path) -> CheckpointStore:
        return cls(state_path=state_path, state=JobState.from_dict(read_json(state_path)))

    def mark_stage(
        self,
        stage: StageName,
        status: StageStatus,
        *,
        artifact: str | None = None,
        done: int | None = None,
        total: int | None = None,
        error: str | None = None,
    ) -> None:
        stages = dict(self.state.stages)
        stages[stage] = StageProgress(status=status, artifact=artifact, done=done, total=total, error=error)
        self.state = replace(
            self.state,
            current_stage=stage,
            stages=stages,
            last_error=error,
            updated_at=utc_now_iso(),
        )

    def mark_job(self, status: JobStatus, *, error: str | None = None) -> None:
        self.state = replace(
            self.state,
            status=status,
            last_error=error,
            updated_at=utc_now_iso(),
        )

    def save(self) -> None:
        write_json_atomic(self.state_path, self.state.to_dict())
