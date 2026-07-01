from __future__ import annotations

import hashlib
import math
import re
from typing import Any

_PUNCTUATION = (".", "?", "!", ";", ":", "。", "？", "！")
_FORMULA_TOKENS = {
    "d", "r", "x", "y", "t", "u", "v", "n", "pi", "π",
    "dr", "dx", "dy", "dt", "over", "squared", "cubed",
    "/", "*", "×", "+", "-", "=", "^2", "^3",
}
_FORMULA_MARKERS = {
    "d", "pi", "π", "dr", "dx", "dy", "dt", "over",
    "squared", "cubed", "/", "*", "×", "+", "-", "=", "^2", "^3",
}


def build_dubbing_cues(
    segments: list[dict[str, Any]],
    *,
    target_duration_ms: int = 4000,
    min_duration_ms: int = 1500,
    max_duration_ms: int = 6000,
) -> list[dict[str, object]]:
    if not 0 < min_duration_ms <= target_duration_ms <= max_duration_ms:
        raise ValueError("cue durations must satisfy 0 < min <= target <= max")
    units = _timeline_units(segments, max_duration_ms=max_duration_ms)
    groups: list[list[dict[str, object]]] = []
    cursor = 0
    while cursor < len(units):
        remaining_duration = int(units[-1]["end_ms"]) - int(units[cursor]["start_ms"])
        if remaining_duration <= max_duration_ms:
            groups.append(units[cursor:])
            break
        candidates: list[tuple[tuple[int, int], int]] = []
        first_safe_before_max: int | None = None
        for index in range(cursor, len(units)):
            duration = int(units[index]["end_ms"]) - int(units[cursor]["start_ms"])
            if duration > max_duration_ms:
                break
            safe = _is_safe_boundary(units, index)
            if safe and first_safe_before_max is None:
                first_safe_before_max = index
            if duration < min_duration_ms:
                continue
            pause_ms = (
                int(units[index + 1]["start_ms"]) - int(units[index]["end_ms"])
                if index + 1 < len(units)
                else 0
            )
            natural = str(units[index]["text"]).rstrip().endswith(_PUNCTUATION) or pause_ms >= 400
            if safe:
                candidates.append(((0 if natural else 1, abs(duration - target_duration_ms)), index))
        chosen = (
            min(candidates)[1]
            if candidates
            else first_safe_before_max
            if first_safe_before_max is not None
            else _next_safe_boundary(units, cursor, min_duration_ms)
        )
        groups.append(units[cursor:chosen + 1])
        cursor = chosen + 1

    groups = _merge_short_groups(groups, min_duration_ms=min_duration_ms, max_duration_ms=max_duration_ms)

    return [_cue_from_group(group, max_duration_ms=max_duration_ms) for group in groups if group]


def _timeline_units(segments: list[dict[str, Any]], *, max_duration_ms: int) -> list[dict[str, object]]:
    units: list[dict[str, object]] = []
    for segment in segments:
        parent = str(segment["segment_id"])
        words = segment.get("words")
        if isinstance(words, list) and words:
            valid_words = [word for word in words if isinstance(word, dict)]
            raw_words = str(segment.get("source_text_raw", "")).split()
            aligned_raw_words = raw_words if len(raw_words) == len(valid_words) else []
            for word_index, word in enumerate(valid_words):
                if not isinstance(word, dict):
                    continue
                text = str(word.get("text") or word.get("word") or "").strip()
                raw_text = str(
                    word.get("raw_text")
                    or (aligned_raw_words[word_index] if aligned_raw_words else text)
                ).strip()
                start_ms = int(word.get("start_ms", 0))
                end_ms = int(word.get("end_ms", 0))
                if text and end_ms > start_ms:
                    units.append({
                        "text": text,
                        "raw_text": raw_text,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "parent": parent,
                    })
            continue
        start_ms = int(segment["start_ms"])
        end_ms = int(segment["end_ms"])
        text_words = str(segment.get("source_text", "")).split()
        raw_words = str(segment.get("source_text_raw", segment.get("source_text", ""))).split()
        if len(raw_words) != len(text_words):
            raw_words = text_words
        part_count = max(1, math.ceil((end_ms - start_ms) / max_duration_ms))
        for part in range(part_count):
            unit_start = start_ms + ((end_ms - start_ms) * part // part_count)
            unit_end = start_ms + ((end_ms - start_ms) * (part + 1) // part_count)
            word_start = len(text_words) * part // part_count
            word_end = len(text_words) * (part + 1) // part_count
            units.append({
                "text": " ".join(text_words[word_start:word_end]),
                "raw_text": " ".join(raw_words[word_start:word_end]),
                "start_ms": unit_start,
                "end_ms": unit_end,
                "parent": parent,
            })
    return sorted(units, key=lambda unit: (int(unit["start_ms"]), int(unit["end_ms"])))


def _group_duration(group: list[dict[str, object]]) -> int:
    return int(group[-1]["end_ms"]) - int(group[0]["start_ms"])


def _merge_short_groups(
    groups: list[list[dict[str, object]]],
    *,
    min_duration_ms: int,
    max_duration_ms: int,
) -> list[list[dict[str, object]]]:
    merged = [list(group) for group in groups if group]
    index = 0
    while index < len(merged):
        if _group_duration(merged[index]) >= min_duration_ms:
            index += 1
            continue

        next_merge = (
            index + 1 < len(merged)
            and _groups_can_merge(merged[index], merged[index + 1], max_duration_ms=max_duration_ms)
        )
        previous_merge = (
            index > 0
            and _groups_can_merge(merged[index - 1], merged[index], max_duration_ms=max_duration_ms)
        )
        if next_merge and (
            not previous_merge
            or _merge_score(merged[index], merged[index + 1], min_duration_ms)
            <= _merge_score(merged[index - 1], merged[index], min_duration_ms)
        ):
            merged[index:index + 2] = [[*merged[index], *merged[index + 1]]]
            continue
        if previous_merge:
            merged[index - 1:index + 1] = [[*merged[index - 1], *merged[index]]]
            index = max(0, index - 1)
            continue
        index += 1
    return merged


def _groups_can_merge(
    left: list[dict[str, object]],
    right: list[dict[str, object]],
    *,
    max_duration_ms: int,
) -> bool:
    if not left or not right:
        return False
    combined_duration = int(right[-1]["end_ms"]) - int(left[0]["start_ms"])
    if combined_duration > max_duration_ms:
        return False
    return _is_safe_group_boundary(left, right)


def _merge_score(left: list[dict[str, object]], right: list[dict[str, object]], min_duration_ms: int) -> int:
    duration = int(right[-1]["end_ms"]) - int(left[0]["start_ms"])
    return abs(duration - min_duration_ms)


def _is_safe_group_boundary(left: list[dict[str, object]], right: list[dict[str, object]]) -> bool:
    return _is_safe_boundary([*left, *right], len(left) - 1)


def _is_safe_boundary(units: list[dict[str, object]], index: int) -> bool:
    if index + 1 >= len(units):
        return True
    tokens = [_normalized_token(str(unit.get("text", ""))) for unit in units]
    if not (_is_formula_token(tokens[index]) and _is_formula_token(tokens[index + 1])):
        return True
    left = index
    while left > 0 and _is_formula_token(tokens[left - 1]):
        left -= 1
    right = index + 1
    while right + 1 < len(tokens) and _is_formula_token(tokens[right + 1]):
        right += 1
    run = tokens[left:right + 1]
    return not any(token in _FORMULA_MARKERS or _is_number_token(token) for token in run)


def _next_safe_boundary(units: list[dict[str, object]], cursor: int, min_duration_ms: int) -> int:
    for index in range(cursor, len(units)):
        duration = int(units[index]["end_ms"]) - int(units[cursor]["start_ms"])
        if duration >= min_duration_ms and _is_safe_boundary(units, index):
            return index
    return len(units) - 1


def _is_formula_token(token: str) -> bool:
    return token in _FORMULA_TOKENS or _is_number_token(token)


def _is_number_token(token: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:[./]\d+)?", token))


def _normalized_token(text: str) -> str:
    token = text.strip().lower()
    token = token.strip("\"'“”‘’()[]{}.,!?;:")
    token = token.replace("π", "pi")
    return token


def _cue_from_group(group: list[dict[str, object]], *, max_duration_ms: int) -> dict[str, object]:
    start_ms = int(group[0]["start_ms"])
    end_ms = int(group[-1]["end_ms"])
    parents = list(dict.fromkeys(str(unit["parent"]) for unit in group))
    source_text = " ".join(str(unit["text"]).strip() for unit in group if str(unit["text"]).strip()).strip()
    source_text_raw = " ".join(
        str(unit.get("raw_text", unit["text"])).strip()
        for unit in group
        if str(unit.get("raw_text", unit["text"])).strip()
    ).strip()
    identity = f"{start_ms}:{end_ms}:{'|'.join(parents)}:{source_text}".encode("utf-8")
    duration_ms = end_ms - start_ms
    return {
        "cue_id": f"cue_{hashlib.sha256(identity).hexdigest()[:12]}",
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": duration_ms,
        "source_text": source_text,
        "source_text_raw": source_text_raw,
        "translated_text": "",
        "parent_segment_ids": parents,
        "risk_flags": (
            ["cue_duration_exceeds_max_for_safe_boundary"]
            if duration_ms > max_duration_ms
            else []
        ),
    }
