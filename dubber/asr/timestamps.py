from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class MissingASRTimestampsError(ValueError):
    pass


@dataclass(frozen=True)
class TimestampUnit:
    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class NormalizedASRTimestamps:
    source: str
    quality: str
    units: list[TimestampUnit]
    risk_flags: list[str]


def normalize_asr_timestamps(
    raw: dict[str, Any],
    *,
    chunk_start_ms: int,
    require_timestamps: bool,
    allow_chunk_text_fallback: bool,
    chunk_end_ms: int | None = None,
) -> NormalizedASRTimestamps:
    word_units = _word_units(raw.get("words"), chunk_start_ms=chunk_start_ms)
    if word_units:
        return NormalizedASRTimestamps(source="word", quality="word", units=word_units, risk_flags=[])

    segment_units = _segment_units(raw.get("segments"), chunk_start_ms=chunk_start_ms)
    if segment_units:
        return NormalizedASRTimestamps(source="segment", quality="segment", units=segment_units, risk_flags=["word_timestamps_missing"])

    text = str(raw.get("text", "")).strip()
    if allow_chunk_text_fallback and text and chunk_end_ms is not None:
        return NormalizedASRTimestamps(
            source="chunk",
            quality="chunk_fallback",
            units=[TimestampUnit(text=text, start_ms=chunk_start_ms, end_ms=chunk_end_ms)],
            risk_flags=["timestamps_missing", "chunk_text_fallback"],
        )

    if require_timestamps:
        raise MissingASRTimestampsError("ASR response did not include timestamps")

    return NormalizedASRTimestamps(source="none", quality="none", units=[], risk_flags=["timestamps_missing"])


def _word_units(words: Any, *, chunk_start_ms: int) -> list[TimestampUnit]:
    if not isinstance(words, list):
        return []
    units: list[TimestampUnit] = []
    for word in words:
        if not isinstance(word, dict):
            continue
        text = str(word.get("word") or word.get("text") or "").strip()
        if not text:
            continue
        start = _seconds_to_absolute_ms(word.get("start"), chunk_start_ms)
        end = _seconds_to_absolute_ms(word.get("end"), chunk_start_ms)
        if start is None or end is None or end <= start:
            continue
        units.append(TimestampUnit(text=text, start_ms=start, end_ms=end))
    return units


def _segment_units(segments: Any, *, chunk_start_ms: int) -> list[TimestampUnit]:
    if not isinstance(segments, list):
        return []
    units: list[TimestampUnit] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = _seconds_to_absolute_ms(segment.get("start"), chunk_start_ms)
        end = _seconds_to_absolute_ms(segment.get("end"), chunk_start_ms)
        if start is None or end is None or end <= start:
            continue
        units.append(TimestampUnit(text=text, start_ms=start, end_ms=end))
    return units


def _seconds_to_absolute_ms(value: Any, chunk_start_ms: int) -> int | None:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return chunk_start_ms + int(seconds * 1000)
