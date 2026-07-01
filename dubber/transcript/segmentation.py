from __future__ import annotations

from typing import Any

from dubber.asr.timestamps import NormalizedASRTimestamps, TimestampUnit
from dubber.core.models import TranscriptSegmentationConfig

_SENTENCE_ENDINGS = (".", "?", "!")


def build_transcript_segments(
    chunks: list[dict[str, Any]],
    config: TranscriptSegmentationConfig,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for chunk in chunks:
        timestamps = chunk["timestamps"]
        if not isinstance(timestamps, NormalizedASRTimestamps):
            raise TypeError("chunk timestamps must be NormalizedASRTimestamps")
        output.extend(_segments_for_chunk(chunk, timestamps, config, start_index=len(output) + 1))
    return output


def _segments_for_chunk(
    chunk: dict[str, Any],
    timestamps: NormalizedASRTimestamps,
    config: TranscriptSegmentationConfig,
    *,
    start_index: int,
) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    current: list[TimestampUnit] = []
    for index, unit in enumerate(timestamps.units):
        current.append(unit)
        next_unit = timestamps.units[index + 1] if index + 1 < len(timestamps.units) else None
        if _should_split(current, unit, next_unit, config):
            segments.append(_build_segment(chunk, timestamps, current, start_index + len(segments)))
            current = []
    if current:
        if segments and _duration(current) < config.target_min_segment_ms:
            segments[-1] = _merge_segment_units(segments[-1], current)
        else:
            segments.append(_build_segment(chunk, timestamps, current, start_index + len(segments)))
    return segments


def _should_split(
    current: list[TimestampUnit],
    unit: TimestampUnit,
    next_unit: TimestampUnit | None,
    config: TranscriptSegmentationConfig,
) -> bool:
    duration_ms = _duration(current)
    if next_unit is None:
        return True
    pause_ms = max(0, next_unit.start_ms - unit.end_ms)
    if duration_ms >= config.max_segment_ms:
        return True
    if duration_ms < config.target_min_segment_ms:
        return False
    if config.prefer_punctuation_split and unit.text.rstrip().endswith(_SENTENCE_ENDINGS):
        return True
    if duration_ms >= config.preferred_max_segment_ms and pause_ms >= config.min_pause_split_ms:
        return True
    if pause_ms >= config.min_pause_split_ms and duration_ms >= config.target_min_segment_ms:
        return True
    return False


def _build_segment(
    chunk: dict[str, Any],
    timestamps: NormalizedASRTimestamps,
    units: list[TimestampUnit],
    index: int,
) -> dict[str, object]:
    vad_risk_flags = [str(flag) for flag in chunk.get("vad_risk_flags", [])]
    risk_flags = list(dict.fromkeys([*timestamps.risk_flags, *vad_risk_flags]))
    segment = {
        "segment_id": f"seg_{index:06d}",
        "start_ms": units[0].start_ms,
        "end_ms": units[-1].end_ms,
        "duration_ms": units[-1].end_ms - units[0].start_ms,
        "source_text": _join_text(units),
        "confidence": 1.0,
        "timestamp_source": timestamps.source,
        "timestamp_quality": timestamps.quality,
        "risk_flags": risk_flags,
        "asr_warnings": risk_flags,
        "raw_response_path": str(chunk.get("raw_response_path", "")),
        "source_chunk_id": str(chunk.get("chunk_id", "")),
        "vad_split_reason": str(chunk.get("vad_split_reason", "")),
        "vad_risk_flags": vad_risk_flags,
        "silence_before_ms": int(chunk.get("silence_before_ms", 0)),
        "silence_after_ms": int(chunk.get("silence_after_ms", 0)),
    }
    if timestamps.source == "word":
        segment["words"] = [_timestamp_unit_to_dict(unit) for unit in units]
    return segment


def _merge_segment_units(segment: dict[str, object], units: list[TimestampUnit]) -> dict[str, object]:
    merged = dict(segment)
    merged["end_ms"] = units[-1].end_ms
    merged["duration_ms"] = int(merged["end_ms"]) - int(merged["start_ms"])
    merged["source_text"] = f"{segment['source_text']} {_join_text(units)}".strip()
    if "words" in merged:
        merged["words"] = [*list(merged["words"]), *[_timestamp_unit_to_dict(unit) for unit in units]]
    risk_flags = list(merged.get("risk_flags", []))
    if "merged_short_tail" not in risk_flags:
        risk_flags.append("merged_short_tail")
    merged["risk_flags"] = risk_flags
    merged["asr_warnings"] = risk_flags
    return merged


def _duration(units: list[TimestampUnit]) -> int:
    if not units:
        return 0
    return units[-1].end_ms - units[0].start_ms


def _join_text(units: list[TimestampUnit]) -> str:
    return " ".join(unit.text.strip() for unit in units if unit.text.strip()).strip()


def _timestamp_unit_to_dict(unit: TimestampUnit) -> dict[str, object]:
    return {"text": unit.text.strip(), "start_ms": unit.start_ms, "end_ms": unit.end_ms}
