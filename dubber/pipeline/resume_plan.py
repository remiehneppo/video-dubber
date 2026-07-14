from __future__ import annotations

from dataclasses import asdict, dataclass

from dubber.core.enums import StageName


@dataclass(frozen=True)
class ResumePlan:
    job_id: str
    start_stage: StageName | None
    from_stage: StageName | None
    no_cache: bool
    preserve_checkpoints: bool
    already_completed: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["start_stage"] = self.start_stage.value if self.start_stage is not None else None
        payload["from_stage"] = self.from_stage.value if self.from_stage is not None else None
        return payload
