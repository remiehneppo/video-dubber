from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dubber.core.enums import StageStatus
from dubber.core.io import read_json, write_json_atomic


@dataclass(frozen=True)
class SegmentCheckpoint:
    segment_id: str
    status: StageStatus
    artifact: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "segment_id": self.segment_id,
            "status": self.status.value,
        }
        if self.artifact is not None:
            data["artifact"] = self.artifact
        if self.error is not None:
            data["error"] = self.error
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SegmentCheckpoint:
        return cls(
            segment_id=str(data["segment_id"]),
            status=StageStatus(str(data["status"])),
            artifact=str(data["artifact"]) if data.get("artifact") is not None else None,
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
        self.segments[segment_id] = SegmentCheckpoint(
            segment_id=segment_id,
            status=status,
            artifact=artifact,
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
