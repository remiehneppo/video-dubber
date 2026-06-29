from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentDurationPlan:
    segment_id: str
    action: str
    stretch_ratio: float
    overflow_ms: int
    warnings: list[str]


def plan_segment_duration(
    segment_id: str,
    *,
    orig_duration_ms: int,
    tts_duration_ms: int,
    speedup_soft_limit: float = 1.2,
    speedup_hard_limit: float = 1.3,
    rephrase_already_attempted: bool = False,
    max_overflow_ms: int = 0,
) -> SegmentDurationPlan:
    if orig_duration_ms <= 0:
        raise ValueError("orig_duration_ms must be positive")
    if tts_duration_ms <= 0:
        raise ValueError("tts_duration_ms must be positive")

    ratio = round(tts_duration_ms / orig_duration_ms, 4)
    if ratio < 1.0:
        return SegmentDurationPlan(
            segment_id=segment_id,
            action="pad_silence",
            stretch_ratio=1.0,
            overflow_ms=0,
            warnings=["tts_duration_shorter_than_source"],
        )
    if ratio == 1.0:
        return SegmentDurationPlan(
            segment_id=segment_id,
            action="pad_silence",
            stretch_ratio=1.0,
            overflow_ms=0,
            warnings=[],
        )
    overflow_ms = tts_duration_ms - orig_duration_ms
    if overflow_ms <= max_overflow_ms:
        return SegmentDurationPlan(
            segment_id=segment_id,
            action="overflow",
            stretch_ratio=1.0,
            overflow_ms=overflow_ms,
            warnings=["tts_duration_overflow_into_source_silence"] if overflow_ms > 0 else [],
        )

    if ratio <= speedup_soft_limit:
        return SegmentDurationPlan(
            segment_id=segment_id,
            action="time_stretch",
            stretch_ratio=round(ratio, 2),
            overflow_ms=0,
            warnings=[],
        )
    if ratio <= speedup_hard_limit:
        return SegmentDurationPlan(
            segment_id=segment_id,
            action="time_stretch",
            stretch_ratio=round(ratio, 2),
            overflow_ms=0,
            warnings=["tts_duration_near_hard_limit"],
        )

    if rephrase_already_attempted and overflow_ms <= max_overflow_ms:
        return SegmentDurationPlan(
            segment_id=segment_id,
            action="overflow",
            stretch_ratio=1.0,
            overflow_ms=overflow_ms,
            warnings=["tts_duration_overflow"],
        )
    return SegmentDurationPlan(
        segment_id=segment_id,
        action="rephrase",
        stretch_ratio=1.0,
        overflow_ms=0,
        warnings=["tts_duration_exceeds_hard_limit"],
    )
