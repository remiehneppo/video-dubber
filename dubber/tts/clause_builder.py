from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TTSClause:
    segment_id: str
    parent_segment_id: str
    start_ms: int
    end_ms: int
    duration_ms: int
    source_text: str
    translated_text: str

    def to_segment(self) -> dict[str, object]:
        return {
            "segment_id": self.segment_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class TTSWorkItem:
    segment: dict[str, Any]
    translated_text: str
    parent_segment_ids: tuple[str, ...]


def build_tts_work_items(
    segments: list[dict[str, Any]],
    translations_by_id: dict[str, dict[str, Any]],
    *,
    min_segment_ms: int = 1500,
    max_merge_gap_ms: int = 700,
) -> list[TTSWorkItem]:
    items: list[TTSWorkItem] = []
    index = 0
    while index < len(segments):
        segment = dict(segments[index])
        parent_ids = [str(segment["segment_id"])]
        translated_parts = [str(translations_by_id.get(parent_ids[0], {}).get("vi_text", ""))]
        while index + 1 < len(segments) and _should_merge_short_fragment(
            segment,
            segments[index + 1],
            min_segment_ms=min_segment_ms,
            max_merge_gap_ms=max_merge_gap_ms,
        ):
            index += 1
            next_segment = segments[index]
            next_id = str(next_segment["segment_id"])
            parent_ids.append(next_id)
            translated_parts.append(str(translations_by_id.get(next_id, {}).get("vi_text", "")))
            segment = _merge_segments(segment, next_segment, parent_ids)
        items.append(
            TTSWorkItem(
                segment=segment,
                translated_text=" ".join(part.strip() for part in translated_parts if part.strip()),
                parent_segment_ids=tuple(parent_ids),
            )
        )
        index += 1
    return items


def build_tts_clauses(
    segment: dict[str, Any],
    translated_text: str,
    *,
    min_pause_ms: int = 700,
) -> list[TTSClause]:
    parent_id = str(segment["segment_id"])
    fallback = [_fallback_clause(segment, translated_text)]
    if segment.get("timestamp_source") != "word":
        return fallback

    words = _valid_words(segment.get("words"))
    if len(words) < 2:
        return fallback
    source_groups = _split_words_at_pauses(words, min_pause_ms=max(0, min_pause_ms))
    if len(source_groups) < 2:
        return fallback

    translated_clauses = _split_sentences(translated_text)
    if len(translated_clauses) != len(source_groups):
        return fallback

    clauses: list[TTSClause] = []
    for index, (source_words, clause_text) in enumerate(zip(source_groups, translated_clauses, strict=True), start=1):
        start_ms = int(source_words[0]["start_ms"])
        end_ms = int(source_words[-1]["end_ms"])
        clauses.append(
            TTSClause(
                segment_id=f"{parent_id}__clause_{index:03d}",
                parent_segment_id=parent_id,
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=end_ms - start_ms,
                source_text=" ".join(str(word["text"]).strip() for word in source_words).strip(),
                translated_text=clause_text,
            )
        )
    return clauses


def _fallback_clause(segment: dict[str, Any], translated_text: str) -> TTSClause:
    start_ms = int(segment["start_ms"])
    end_ms = int(segment["end_ms"])
    return TTSClause(
        segment_id=str(segment["segment_id"]),
        parent_segment_id=str(segment["segment_id"]),
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=int(segment.get("duration_ms", end_ms - start_ms)),
        source_text=str(segment.get("source_text", "")),
        translated_text=translated_text,
    )


def _valid_words(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    words: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            start_ms = int(item["start_ms"])
            end_ms = int(item["end_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(item.get("text", "")).strip()
        if text and end_ms > start_ms:
            words.append({"text": text, "start_ms": start_ms, "end_ms": end_ms})
    return words


def _split_words_at_pauses(words: list[dict[str, object]], min_pause_ms: int) -> list[list[dict[str, object]]]:
    groups: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for index, word in enumerate(words):
        current.append(word)
        next_word = words[index + 1] if index + 1 < len(words) else None
        if next_word is None or int(next_word["start_ms"]) - int(word["end_ms"]) >= min_pause_ms:
            groups.append(current)
            current = []
    return groups


def _split_sentences(text: str) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return [""]
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def _should_merge_short_fragment(
    current: dict[str, Any],
    following: dict[str, Any],
    *,
    min_segment_ms: int,
    max_merge_gap_ms: int,
) -> bool:
    if int(current.get("duration_ms", 0)) >= min_segment_ms:
        return False
    if str(current.get("source_text", "")).rstrip().endswith((".", "?", "!")):
        return False
    gap_ms = int(following["start_ms"]) - int(current["end_ms"])
    return 0 <= gap_ms <= max_merge_gap_ms


def _merge_segments(
    current: dict[str, Any],
    following: dict[str, Any],
    parent_ids: list[str],
) -> dict[str, Any]:
    merged = dict(current)
    merged["segment_id"] = "__merged__".join(parent_ids)
    merged["end_ms"] = int(following["end_ms"])
    merged["duration_ms"] = int(merged["end_ms"]) - int(merged["start_ms"])
    merged["source_text"] = f"{current.get('source_text', '')} {following.get('source_text', '')}".strip()
    if current.get("timestamp_source") == "word" and following.get("timestamp_source") == "word":
        merged["words"] = [*list(current.get("words", [])), *list(following.get("words", []))]
        merged["timestamp_source"] = "word"
    return merged
