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


@dataclass(frozen=True)
class _WordTimestampNormalization:
    units: list[TimestampUnit]
    complete: bool
    repaired: bool
    has_textual_entries: bool


def normalize_asr_timestamps(
    raw: dict[str, Any],
    *,
    chunk_start_ms: int,
    require_timestamps: bool,
    allow_chunk_text_fallback: bool,
    require_word_timestamps: bool = False,
    chunk_end_ms: int | None = None,
) -> NormalizedASRTimestamps:
    raw_words = raw.get("words")
    word_result = _normalize_word_units(raw_words, chunk_start_ms=chunk_start_ms)
    if word_result.units:
        if require_word_timestamps and not word_result.complete:
            raise MissingASRTimestampsError("ASR word-level timestamps are incomplete or out of order")
        risk_flags = ["word_timestamp_repaired"] if word_result.repaired else []
        return NormalizedASRTimestamps(source="word", quality="word", units=word_result.units, risk_flags=risk_flags)

    if require_word_timestamps:
        if word_result.has_textual_entries:
            raise MissingASRTimestampsError("ASR word-level timestamps are incomplete or out of order")
        raise MissingASRTimestampsError("ASR response did not include word-level timestamps")

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


def _normalize_word_units(words: Any, *, chunk_start_ms: int) -> _WordTimestampNormalization:
    if not isinstance(words, list):
        return _WordTimestampNormalization([], complete=False, repaired=False, has_textual_entries=False)

    entries: list[tuple[str, int | None, int | None]] = []
    for word in words:
        if not isinstance(word, dict):
            continue
        text = str(word.get("word") or word.get("text") or "").strip()
        if not text:
            continue
        entries.append(
            (
                text,
                _seconds_to_absolute_ms(word.get("start"), chunk_start_ms),
                _seconds_to_absolute_ms(word.get("end"), chunk_start_ms),
            )
        )

    if not entries:
        return _WordTimestampNormalization([], complete=False, repaired=False, has_textual_entries=False)
    if any(start is None or end is None for _, start, end in entries):
        return _WordTimestampNormalization([], complete=False, repaired=False, has_textual_entries=True)

    units: list[TimestampUnit] = []
    repaired = False
    cursor: int | None = None
    for index, (text, raw_start, raw_end) in enumerate(entries):
        start = int(raw_start)
        end = int(raw_end)
        original_start = start
        original_end = end
        if cursor is not None and start < cursor:
            start = cursor
        if end <= start:
            future_boundary = _future_word_timestamp_boundary(entries, index, start)
            end = min(start + 20, future_boundary) if future_boundary is not None and future_boundary > start else start + 20
        if end <= start:
            end = start + 20
        if start != original_start or end != original_end:
            repaired = True
        units.append(TimestampUnit(text=text, start_ms=start, end_ms=end))
        cursor = end

    complete = len(entries) == len(units) and all(
        current.start_ms >= previous.start_ms and current.end_ms >= previous.end_ms
        for previous, current in zip(units, units[1:])
    )
    return _WordTimestampNormalization(units, complete=complete, repaired=repaired, has_textual_entries=True)


def _future_word_timestamp_boundary(
    entries: list[tuple[str, int | None, int | None]],
    index: int,
    current_start_ms: int,
) -> int | None:
    for _, future_start, future_end in entries[index + 1:]:
        for value in (future_start, future_end):
            if value is not None and value > current_start_ms:
                return int(value)
    return None


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
