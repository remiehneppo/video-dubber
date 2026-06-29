from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dubber.core.enums import StageStatus
from dubber.core.io import read_json, write_json_atomic
from dubber.orchestrator.artifact_manifest import sha256_file


@dataclass(frozen=True)
class SegmentCheckpoint:
    segment_id: str
    status: StageStatus
    artifact: str | None = None
    sha256: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "segment_id": self.segment_id,
            "status": self.status.value,
        }
        if self.artifact is not None:
            data["artifact"] = self.artifact
        if self.sha256 is not None:
            data["sha256"] = self.sha256
        if self.error is not None:
            data["error"] = self.error
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SegmentCheckpoint:
        return cls(
            segment_id=str(data["segment_id"]),
            status=StageStatus(str(data["status"])),
            artifact=str(data["artifact"]) if data.get("artifact") is not None else None,
            sha256=str(data["sha256"]) if data.get("sha256") is not None else None,
            error=str(data["error"]) if data.get("error") is not None else None,
        )


class SegmentCheckpointStore:
    def __init__(self, path: Path, stage: str, segments: dict[str, SegmentCheckpoint]) -> None:
        self.path = path
        self.stage = stage
        self.segments = segments

    @classmethod
    def create(cls, path: Path, *, stage: str, segment_ids: list[str]) -> SegmentCheckpointStore:
        return cls(
            path=path,
            stage=stage,
            segments={
                segment_id: SegmentCheckpoint(segment_id=segment_id, status=StageStatus.PENDING)
                for segment_id in segment_ids
            },
        )

    @classmethod
    def load_or_create(cls, path: Path, *, stage: str, segment_ids: list[str]) -> SegmentCheckpointStore:
        """Load compatible progress and add/remove units to match the current input."""
        if not path.exists():
            return cls.create(path, stage=stage, segment_ids=segment_ids)
        existing = cls.load(path)
        if existing.stage != stage:
            return cls.create(path, stage=stage, segment_ids=segment_ids)
        return cls(
            path=path,
            stage=stage,
            segments={
                segment_id: existing.segments.get(
                    segment_id,
                    SegmentCheckpoint(segment_id=segment_id, status=StageStatus.PENDING),
                )
                for segment_id in segment_ids
            },
        )

    def invalidate_missing_artifacts(self, root: Path) -> None:
        for segment_id, checkpoint in list(self.segments.items()):
            if checkpoint.status != StageStatus.COMPLETED:
                continue
            if checkpoint.artifact is None:
                self.mark(segment_id, StageStatus.PENDING)
                continue
            path = root / checkpoint.artifact
            if (
                not path.exists()
                or not path.is_file()
                or (checkpoint.sha256 is not None and sha256_file(path) != checkpoint.sha256)
            ):
                self.mark(segment_id, StageStatus.PENDING)

    @classmethod
    def load(cls, path: Path) -> SegmentCheckpointStore:
        data = read_json(path)
        return cls(
            path=path,
            stage=str(data["stage"]),
            segments={
                str(item["segment_id"]): SegmentCheckpoint.from_dict(item)
                for item in data.get("segments", [])
            },
        )

    @property
    def done_count(self) -> int:
        return sum(1 for segment in self.segments.values() if segment.status == StageStatus.COMPLETED)

    @property
    def total_count(self) -> int:
        return len(self.segments)

    def incomplete_segment_ids(self) -> list[str]:
        return [
            segment.segment_id
            for segment in self.segments.values()
            if segment.status != StageStatus.COMPLETED
        ]

    def mark(
        self,
        segment_id: str,
        status: StageStatus,
        *,
        artifact: str | None = None,
        error: str | None = None,
    ) -> None:
        if segment_id not in self.segments:
            raise KeyError(f"Unknown segment_id: {segment_id}")
        checksum = None
        if artifact is not None:
            artifact_path = self.path.parent.parent / artifact
            if artifact_path.is_file():
                checksum = sha256_file(artifact_path)
        self.segments[segment_id] = SegmentCheckpoint(
            segment_id=segment_id,
            status=status,
            artifact=artifact,
            sha256=checksum,
            error=error,
        )

    def save(self) -> None:
        write_json_atomic(
            self.path,
            {
                "schema_version": "1.0",
                "stage": self.stage,
                "done": self.done_count,
                "total": self.total_count,
                "segments": [segment.to_dict() for segment in self.segments.values()],
            },
        )
