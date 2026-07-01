from __future__ import annotations

from typing import Any


def build_speech_timeline(
    cues: list[dict[str, Any]],
    *,
    source_segments: list[dict[str, Any]] | None = None,
    total_duration_ms: int | None = None,
    guard_ms: int = 100,
    source_pause_threshold_ms: int = 250,
) -> dict[str, object]:
    """Build a voice-first timeline with explicit speech and silence intervals.

    This artifact is intentionally derived from the cue artifact for this slice:
    cue generation remains the compatibility boundary, while TTS scheduling can
    consume explicit silence gaps in the next slice.
    """

    ordered = sorted(cues, key=lambda cue: (int(cue["start_ms"]), int(cue["end_ms"]), str(cue["cue_id"])))
    speech_intervals = [_speech_interval(cue, index) for index, cue in enumerate(ordered)]
    silence_intervals = _silence_intervals(speech_intervals, total_duration_ms=total_duration_ms, guard_ms=guard_ms)
    source_speech_intervals, source_timing_quality = _source_speech_intervals(
        source_segments or [],
        ordered,
        pause_threshold_ms=source_pause_threshold_ms,
    )
    if not source_speech_intervals:
        source_speech_intervals = [
            {
                "interval_id": f"source_speech_{index:06d}",
                "start_ms": int(interval["start_ms"]),
                "end_ms": int(interval["end_ms"]),
                "duration_ms": int(interval["duration_ms"]),
                "word_count": 0,
                "segment_ids": list(interval.get("parent_segment_ids", [])),
                "cue_ids": [str(interval["cue_id"])],
            }
            for index, interval in enumerate(speech_intervals)
        ]
        source_timing_quality = "cue_fallback"
    return {
        "schema_version": "1.0",
        "source": "dubbing_cues.v1",
        "guard_ms": guard_ms,
        "total_duration_ms": total_duration_ms,
        "speech_intervals": speech_intervals,
        "silence_intervals": silence_intervals,
        "source_timing_quality": source_timing_quality,
        "source_pause_threshold_ms": source_pause_threshold_ms,
        "source_speech_intervals": source_speech_intervals,
        "source_silence_intervals": _source_silence_intervals(
            source_speech_intervals,
            total_duration_ms=total_duration_ms,
        ),
    }


def _source_speech_intervals(
    source_segments: list[dict[str, Any]],
    cues: list[dict[str, Any]],
    *,
    pause_threshold_ms: int,
) -> tuple[list[dict[str, object]], str]:
    words: list[dict[str, object]] = []
    for segment in source_segments:
        segment_id = str(segment.get("segment_id", ""))
        for word in segment.get("words", []):
            if not isinstance(word, dict):
                continue
            start_ms = int(word.get("start_ms", 0))
            end_ms = int(word.get("end_ms", 0))
            if end_ms <= start_ms:
                continue
            words.append({"start_ms": start_ms, "end_ms": end_ms, "segment_id": segment_id})
    if not words:
        return [], "unavailable"
    words.sort(key=lambda word: (int(word["start_ms"]), int(word["end_ms"])))
    groups: list[list[dict[str, object]]] = []
    current = [words[0]]
    for word in words[1:]:
        gap_ms = int(word["start_ms"]) - int(current[-1]["end_ms"])
        if gap_ms < max(0, pause_threshold_ms):
            current.append(word)
        else:
            groups.append(current)
            current = [word]
    groups.append(current)

    intervals: list[dict[str, object]] = []
    for index, group in enumerate(groups):
        start_ms = int(group[0]["start_ms"])
        end_ms = max(int(word["end_ms"]) for word in group)
        cue_ids = [
            str(cue["cue_id"])
            for cue in cues
            if int(cue["end_ms"]) > start_ms and int(cue["start_ms"]) < end_ms
        ]
        intervals.append({
            "interval_id": f"source_speech_{index:06d}",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
            "word_count": len(group),
            "segment_ids": list(dict.fromkeys(str(word["segment_id"]) for word in group)),
            "cue_ids": cue_ids,
        })
    return intervals, "word"


def _source_silence_intervals(
    speech_intervals: list[dict[str, object]],
    *,
    total_duration_ms: int | None,
) -> list[dict[str, object]]:
    if not speech_intervals:
        return []
    gaps: list[dict[str, object]] = []
    boundaries: list[tuple[int, int, str]] = []
    first_start = int(speech_intervals[0]["start_ms"])
    if first_start > 0:
        boundaries.append((0, first_start, "before_first"))
    for current, following in zip(speech_intervals, speech_intervals[1:]):
        start_ms = int(current["end_ms"])
        end_ms = int(following["start_ms"])
        if end_ms > start_ms:
            boundaries.append((start_ms, end_ms, "between_speech"))
    if total_duration_ms is not None:
        last_end = int(speech_intervals[-1]["end_ms"])
        if total_duration_ms > last_end:
            boundaries.append((last_end, total_duration_ms, "after_last"))
    for index, (start_ms, end_ms, position) in enumerate(boundaries):
        gaps.append({
            "interval_id": f"source_silence_{index:06d}",
            "position": position,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
        })
    return gaps


def _speech_interval(cue: dict[str, Any], index: int) -> dict[str, object]:
    start_ms = int(cue["start_ms"])
    end_ms = int(cue["end_ms"])
    return {
        "index": index,
        "cue_id": str(cue["cue_id"]),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": max(0, end_ms - start_ms),
        "source_text": str(cue.get("source_text", "")),
        "display_text": str(cue.get("display_text") or cue.get("translated_text") or ""),
        "spoken_text": str(cue.get("spoken_text") or cue.get("translated_text") or ""),
        "parent_segment_ids": list(cue.get("parent_segment_ids", [])),
        "protected_spans": list(cue.get("protected_spans", [])),
        "risk_flags": list(cue.get("risk_flags", [])),
    }


def _silence_intervals(
    speech_intervals: list[dict[str, object]],
    *,
    total_duration_ms: int | None,
    guard_ms: int,
) -> list[dict[str, object]]:
    if not speech_intervals:
        if total_duration_ms is None or total_duration_ms <= 0:
            return []
        return [
            {
                "gap_id": "gap_000000",
                "position": "full_track",
                "start_ms": 0,
                "end_ms": total_duration_ms,
                "duration_ms": total_duration_ms,
                "preceding_cue_id": None,
                "following_cue_id": None,
                "guard_ms": guard_ms,
                "usable_overflow_ms": _usable_overflow_ms(total_duration_ms, guard_ms),
                "preserve_pause_ms": _preserve_pause_ms(total_duration_ms),
            }
        ]

    gaps: list[dict[str, object]] = []
    first_start = int(speech_intervals[0]["start_ms"])
    if first_start > 0:
        gaps.append(_gap("gap_000000", "before_first", 0, first_start, None, str(speech_intervals[0]["cue_id"]), guard_ms))
    for index, current in enumerate(speech_intervals[:-1], start=1):
        following = speech_intervals[index]
        start_ms = int(current["end_ms"])
        end_ms = int(following["start_ms"])
        if end_ms <= start_ms:
            continue
        gaps.append(
            _gap(
                f"gap_{index:06d}",
                "between_speech",
                start_ms,
                end_ms,
                str(current["cue_id"]),
                str(following["cue_id"]),
                guard_ms,
            )
        )
    if total_duration_ms is not None:
        last_end = int(speech_intervals[-1]["end_ms"])
        if total_duration_ms > last_end:
            gaps.append(
                _gap(
                    f"gap_{len(gaps):06d}",
                    "after_last",
                    last_end,
                    total_duration_ms,
                    str(speech_intervals[-1]["cue_id"]),
                    None,
                    guard_ms,
                )
            )
    return gaps


def _gap(
    gap_id: str,
    position: str,
    start_ms: int,
    end_ms: int,
    preceding_cue_id: str | None,
    following_cue_id: str | None,
    guard_ms: int,
) -> dict[str, object]:
    duration_ms = max(0, end_ms - start_ms)
    preserve_pause_ms = _preserve_pause_ms(duration_ms)
    return {
        "gap_id": gap_id,
        "position": position,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": duration_ms,
        "preceding_cue_id": preceding_cue_id,
        "following_cue_id": following_cue_id,
        "guard_ms": guard_ms,
        "preserve_pause_ms": preserve_pause_ms,
        "usable_overflow_ms": max(0, duration_ms - max(0, guard_ms) - preserve_pause_ms),
    }


def _preserve_pause_ms(duration_ms: int) -> int:
    if duration_ms < 250:
        return duration_ms
    if duration_ms < 600:
        return max(150, int(duration_ms * 0.60))
    if duration_ms < 1200:
        return max(250, int(duration_ms * 0.45))
    return max(350, int(duration_ms * 0.30))


def _usable_overflow_ms(duration_ms: int, guard_ms: int) -> int:
    return max(0, duration_ms - max(0, guard_ms) - _preserve_pause_ms(duration_ms))
