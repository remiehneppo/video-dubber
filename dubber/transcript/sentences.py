from __future__ import annotations

import re
from typing import Any


_TERMINAL_PUNCTUATION = (".", "?", "!", ";", ":", "。", "？", "！")
_DANGLING_LEFT = {
    "a", "an", "the", "and", "or", "but", "of", "to", "for", "with", "by",
    "as", "that", "which", "who", "whose", "what", "when", "where", "how",
    "will", "would", "could", "should", "can", "may", "might", "must",
    "is", "are", "was", "were", "be", "been", "being",
}
_DANGLING_RIGHT = {
    "of", "to", "for", "with", "by", "as", "than", "from",
}


def build_source_sentences(
    segments: list[dict[str, Any]],
    *,
    natural_pause_ms: int = 700,
) -> list[dict[str, object]]:
    words = _timeline_words(segments)
    if not words:
        return []

    groups: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for index, word in enumerate(words):
        current.append(word)
        text = str(word["text"]).rstrip()
        if text.endswith(_TERMINAL_PUNCTUATION):
            groups.append(current)
            current = []
            continue
        if index + 1 >= len(words):
            continue
        pause_ms = int(words[index + 1]["start_ms"]) - int(word["end_ms"])
        if pause_ms >= natural_pause_ms and _is_safe_pause_boundary(words, index):
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    return [_sentence_from_group(group, index) for index, group in enumerate(groups, start=1)]


def _timeline_words(segments: list[dict[str, Any]]) -> list[dict[str, object]]:
    words: list[dict[str, object]] = []
    for segment in segments:
        segment_id = str(segment.get("segment_id", ""))
        source_chunk_ids = _source_chunk_ids(segment)
        risk_flags = list(segment.get("risk_flags", []))
        raw_words = segment.get("words")
        if not isinstance(raw_words, list):
            continue
        valid_words = [word for word in raw_words if isinstance(word, dict)]
        source_raw_tokens = str(segment.get("source_text_raw", "")).split()
        aligned_raw_tokens = source_raw_tokens if len(source_raw_tokens) == len(valid_words) else []
        for word_index, raw_word in enumerate(valid_words):
            if not isinstance(raw_word, dict):
                continue
            text = str(raw_word.get("text") or raw_word.get("word") or "").strip()
            start_ms = int(raw_word.get("start_ms", 0))
            end_ms = int(raw_word.get("end_ms", 0))
            if not text or end_ms <= start_ms:
                continue
            words.append(
                {
                    **raw_word,
                    "text": text,
                    "raw_text": str(
                        raw_word.get("raw_text")
                        or (aligned_raw_tokens[word_index] if aligned_raw_tokens else text)
                    ),
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "parent_segment_id": segment_id,
                    "source_chunk_ids": source_chunk_ids,
                    "risk_flags": risk_flags,
                }
            )
    return sorted(words, key=lambda word: (int(word["start_ms"]), int(word["end_ms"])))


def _is_safe_pause_boundary(words: list[dict[str, object]], index: int) -> bool:
    left = _normalized_token(str(words[index]["text"]))
    right = _normalized_token(str(words[index + 1]["text"]))
    return left not in _DANGLING_LEFT and right not in _DANGLING_RIGHT


def _normalized_token(text: str) -> str:
    return re.sub(r"^\W+|\W+$", "", text.casefold())


def _sentence_from_group(group: list[dict[str, object]], index: int) -> dict[str, object]:
    parents = list(dict.fromkeys(str(word["parent_segment_id"]) for word in group))
    source_chunk_ids = list(dict.fromkeys(
        str(source_chunk_id)
        for word in group
        for source_chunk_id in list(word.get("source_chunk_ids", []))
    ))
    risk_flags = list(dict.fromkeys(
        flag
        for word in group
        for flag in list(word.get("risk_flags", []))
    ))
    words = []
    for source in group:
        word = dict(source)
        word.pop("parent_segment_id", None)
        word.pop("risk_flags", None)
        words.append(word)
    return {
        "segment_id": f"sentence_{index:06d}",
        "start_ms": int(group[0]["start_ms"]),
        "end_ms": int(group[-1]["end_ms"]),
        "duration_ms": int(group[-1]["end_ms"]) - int(group[0]["start_ms"]),
        "source_text": _join_words(group, "text"),
        "source_text_raw": _join_words(group, "raw_text"),
        "words": words,
        "parent_segment_ids": parents,
        "source_chunk_ids": source_chunk_ids,
        "risk_flags": risk_flags,
        "boundary_reason": (
            "sentence_punctuation"
            if str(group[-1]["text"]).rstrip().endswith(_TERMINAL_PUNCTUATION)
            else "natural_pause_or_end"
        ),
    }


def _source_chunk_ids(segment: dict[str, Any]) -> list[str]:
    raw_ids = segment.get("source_chunk_ids")
    if isinstance(raw_ids, list):
        return list(dict.fromkeys(str(item) for item in raw_ids if str(item)))
    raw_id = str(segment.get("source_chunk_id", ""))
    return [raw_id] if raw_id else []


def _join_words(words: list[dict[str, object]], key: str) -> str:
    text = " ".join(str(word.get(key, word.get("text", ""))).strip() for word in words).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", text)
