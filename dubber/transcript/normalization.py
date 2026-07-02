from __future__ import annotations

import re
from typing import Any

from dubber.domain.profiles import DomainProfile, detect_protected_spans


_CALCULUS_CONTEXT_TERMS = (
    "integral",
    "derivative",
    "differential",
    "radius",
    "radii",
    "ring",
    "thickness",
    "area",
    "2 pi",
    "pi r",
    "π",
    "with respect to",
)


def normalize_transcript_segments(
    segments: list[dict[str, Any]],
    profile: DomainProfile,
    *,
    context_radius: int = 1,
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    raw_texts = [str(segment.get("source_text", "")) for segment in segments]
    for index, segment in enumerate(segments):
        start = max(0, index - context_radius)
        end = min(len(segments), index + context_radius + 1)
        context_text = " ".join(raw_texts[start:end])
        normalized.append(_normalize_segment(segment, profile, context_text=context_text))
    return _collapse_repeated_asr_loops_across_segments(normalized, profile)


def build_word_timeline(segments: list[dict[str, object]]) -> list[dict[str, object]]:
    words: list[dict[str, object]] = []
    for segment in segments:
        segment_id = str(segment.get("segment_id", ""))
        source_chunk_id = str(segment.get("source_chunk_id", ""))
        for word in _copy_words(segment.get("words")):
            item = dict(word)
            item["segment_id"] = segment_id
            item["source_chunk_id"] = source_chunk_id
            words.append(item)
    return sorted(
        words,
        key=lambda word: (
            int(word.get("start_ms", 0)),
            int(word.get("end_ms", 0)),
            str(word.get("segment_id", "")),
        ),
    )


def find_source_normalization_candidates(
    segments: list[dict[str, object]],
    profile: DomainProfile,
) -> list[dict[str, object]]:
    if profile.profile_id != "calculus":
        return []
    candidates: list[dict[str, object]] = []
    for segment in segments:
        source_text = str(segment.get("source_text_raw", segment.get("source_text", "")))
        for match_index, match in enumerate(re.finditer(r"(?<![A-Za-zÀ-ỹ])doctor(?![A-Za-zÀ-ỹ])", source_text, re.IGNORECASE), start=1):
            already_normalized = any(
                isinstance(edit, dict)
                and str(edit.get("rule_id", "")) == "calculus.doctor_to_dr"
                and int(edit.get("start_char", -1)) == match.start()
                for edit in list(segment.get("normalization_edits", []))
            )
            if already_normalized:
                continue
            segment_id = str(segment.get("segment_id", ""))
            candidates.append(
                {
                    "candidate_id": f"{segment_id}:doctor:{match_index}",
                    "segment_id": segment_id,
                    "original": match.group(0),
                    "start_char": match.start(),
                    "end_char": match.end(),
                    "source_text": source_text,
                    "suggested_normalized": "dr",
                    "reason": "Potential ASR confusion between doctor and calculus differential dr; LLM may suggest but must not apply automatically.",
                }
            )
    return candidates


def attach_source_normalization_suggestions(
    segments: list[dict[str, object]],
    suggestions: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not suggestions:
        return segments
    suggestions_by_segment: dict[str, list[dict[str, object]]] = {}
    for suggestion in suggestions:
        if not isinstance(suggestion, dict):
            continue
        segment_id = str(suggestion.get("segment_id", "")).strip()
        if not segment_id:
            continue
        suggestions_by_segment.setdefault(segment_id, []).append(
            {
                "candidate_id": str(suggestion.get("candidate_id", "")),
                "original": str(suggestion.get("original", "")),
                "suggested_normalized": str(suggestion.get("suggested_normalized", "")),
                "confidence": float(suggestion.get("confidence", 0.0) or 0.0),
                "reason": str(suggestion.get("reason", "")),
                "rule_id": "llm.source_normalization_suggestion",
            }
        )
    normalized: list[dict[str, object]] = []
    for segment in segments:
        item = dict(segment)
        segment_suggestions = suggestions_by_segment.get(str(item.get("segment_id", "")), [])
        if segment_suggestions:
            existing_suggestions = item.get("normalization_suggestions", [])
            if not isinstance(existing_suggestions, list):
                existing_suggestions = []
            item["normalization_suggestions"] = [
                *existing_suggestions,
                *segment_suggestions,
            ]
            item["risk_flags"] = list(dict.fromkeys([
                *list(item.get("risk_flags", [])),
                "source_normalization_llm_suggestion",
            ]))
        normalized.append(item)
    return normalized


def _normalize_segment(segment: dict[str, Any], profile: DomainProfile, *, context_text: str) -> dict[str, object]:
    raw_text = str(segment.get("source_text", ""))
    normalized_text = raw_text
    edits: list[dict[str, object]] = []
    confidence = 1.0
    risk_flags = list(segment.get("risk_flags", []))
    words = _copy_words(segment.get("words"))

    if profile.profile_id == "calculus" and _looks_like_calculus_context(context_text):
        normalized_text, doctor_edits = _replace_doctor_differentials(normalized_text)
        if doctor_edits:
            edits.extend(doctor_edits)
            words = _normalize_doctor_words(words)
            confidence = min(confidence, 0.92)
            risk_flags.append("source_normalized_calculus_notation")

    protected_spans = [span.to_dict() for span in detect_protected_spans(normalized_text, profile)]
    result = dict(segment)
    result["source_text_raw"] = raw_text
    result["source_text_normalized"] = normalized_text
    result["source_text"] = normalized_text
    result["normalization_edits"] = edits
    result["normalization_suggestions"] = list(segment.get("normalization_suggestions", []))
    result["normalization_confidence"] = confidence
    result["protected_spans"] = protected_spans
    result["risk_flags"] = list(dict.fromkeys(risk_flags))
    if words:
        result["words"] = words
    return result


def _looks_like_calculus_context(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in _CALCULUS_CONTEXT_TERMS)


def _replace_doctor_differentials(text: str) -> tuple[str, list[dict[str, object]]]:
    edits: list[dict[str, object]] = []

    def replace(match: re.Match[str]) -> str:
        edits.append(
            {
                "rule_id": "calculus.doctor_to_dr",
                "original": match.group(0),
                "normalized": "dr",
                "start_char": match.start(),
                "end_char": match.end(),
                "confidence": 0.92,
                "reason": "ASR commonly mishears calculus differential dr as doctor in radius/thickness/integral context.",
            }
        )
        return "dr"

    normalized = re.sub(r"(?<![A-Za-zÀ-ỹ])doctor(?![A-Za-zÀ-ỹ])", replace, text, flags=re.IGNORECASE)
    return normalized, edits


def _copy_words(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    copied: list[dict[str, object]] = []
    for word in value:
        if not isinstance(word, dict):
            continue
        item = dict(word)
        item.setdefault("raw_text", str(item.get("text") or item.get("word") or ""))
        copied.append(item)
    return copied


def _normalize_doctor_words(words: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for word in words:
        item = dict(word)
        text = str(item.get("text") or item.get("word") or "")
        replaced = re.sub(r"(?i)(?<![A-Za-zÀ-ỹ])doctor(?![A-Za-zÀ-ỹ])", "dr", text)
        if "text" in item:
            item["text"] = replaced
        if "word" in item:
            item["word"] = replaced
        if "text" not in item and "word" not in item:
            item["text"] = replaced
        normalized.append(item)
    return normalized


def _collapse_repeated_asr_loops(
    words: list[dict[str, object]],
    *,
    min_phrase_words: int = 8,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Remove immediately repeated long word sequences while retaining timestamps."""
    collapsed = list(words)
    edits: list[dict[str, object]] = []
    while True:
        content = [
            (index, token)
            for index, word in enumerate(collapsed)
            if (token := _comparison_token(str(word.get("text") or word.get("word") or "")))
        ]
        match: tuple[int, int, int] | None = None
        for phrase_length in range(min_phrase_words, len(content) // 2 + 1):
            for start in range(0, len(content) - (2 * phrase_length) + 1):
                phrase = [token for _, token in content[start:start + phrase_length]]
                next_phrase = [
                    token
                    for _, token in content[start + phrase_length:start + (2 * phrase_length)]
                ]
                if phrase != next_phrase:
                    continue
                copies = 2
                while start + ((copies + 1) * phrase_length) <= len(content):
                    candidate = [
                        token
                        for _, token in content[
                            start + (copies * phrase_length):start + ((copies + 1) * phrase_length)
                        ]
                    ]
                    if candidate != phrase:
                        break
                    copies += 1
                match = (start, phrase_length, copies)
                break
            if match is not None:
                break
        if match is None:
            break

        start, phrase_length, copies = match
        first_end = content[start + phrase_length - 1][0]
        remove_start = content[start + phrase_length][0]
        remove_end = content[start + (copies * phrase_length) - 1][0]
        while remove_end + 1 < len(collapsed) and not _comparison_token(
            str(collapsed[remove_end + 1].get("text") or collapsed[remove_end + 1].get("word") or "")
        ):
            remove_end += 1
        original = _join_words(collapsed[remove_start:remove_end + 1])
        retained = _join_words(collapsed[content[start][0]:first_end + 1])
        retained_segment_id = str(collapsed[content[start][0]].get("_segment_id", ""))
        removed_segment_ids = list(dict.fromkeys(
            str(word.get("_segment_id", ""))
            for word in collapsed[remove_start:remove_end + 1]
            if str(word.get("_segment_id", ""))
        ))
        prefix = _join_words(collapsed[:remove_start])
        start_char = len(prefix) + (1 if prefix else 0)
        del collapsed[remove_start:remove_end + 1]
        edits.append(
            {
                "rule_id": "generic.repeated_asr_loop",
                "original": original,
                "normalized": retained,
                "start_char": start_char,
                "end_char": start_char + len(original),
                "confidence": 0.88,
                "reason": f"Removed {copies - 1} immediately repeated ASR copies of a {phrase_length}-word phrase.",
                "retained_segment_id": retained_segment_id,
                "removed_segment_ids": removed_segment_ids,
            }
        )
    return collapsed, edits


def _collapse_repeated_asr_loops_across_segments(
    segments: list[dict[str, object]],
    profile: DomainProfile,
) -> list[dict[str, object]]:
    tagged_words: list[dict[str, object]] = []
    original_counts: dict[str, int] = {}
    for segment in segments:
        segment_id = str(segment.get("segment_id", ""))
        words = _copy_words(segment.get("words"))
        original_counts[segment_id] = len(words)
        for word in words:
            tagged = dict(word)
            tagged["_segment_id"] = segment_id
            tagged_words.append(tagged)

    collapsed_words, edits = _collapse_repeated_asr_loops(tagged_words)
    if not edits:
        return segments

    words_by_segment: dict[str, list[dict[str, object]]] = {}
    for word in collapsed_words:
        segment_id = str(word.get("_segment_id", ""))
        clean = dict(word)
        clean.pop("_segment_id", None)
        words_by_segment.setdefault(segment_id, []).append(clean)

    normalized: list[dict[str, object]] = []
    for segment in segments:
        segment_id = str(segment.get("segment_id", ""))
        words = words_by_segment.get(segment_id, [])
        if original_counts.get(segment_id, 0) > 0 and not words:
            continue
        item = dict(segment)
        segment_edits = [
            dict(edit)
            for edit in edits
            if segment_id == str(edit.get("retained_segment_id", ""))
        ]
        if words:
            item["words"] = words
            item["source_text"] = _join_words(words)
            item["source_text_normalized"] = item["source_text"]
            item["start_ms"] = int(words[0].get("start_ms", item.get("start_ms", 0)))
            item["end_ms"] = int(words[-1].get("end_ms", item.get("end_ms", 0)))
            item["duration_ms"] = int(item["end_ms"]) - int(item["start_ms"])
            item["protected_spans"] = [
                span.to_dict()
                for span in detect_protected_spans(str(item["source_text"]), profile)
            ]
        if segment_edits:
            item["normalization_edits"] = [
                *list(item.get("normalization_edits", [])),
                *segment_edits,
            ]
            item["normalization_confidence"] = min(
                float(item.get("normalization_confidence", 1.0)),
                0.88,
            )
            item["risk_flags"] = list(dict.fromkeys([
                *list(item.get("risk_flags", [])),
                "source_normalized_repetitive_asr_loop",
            ]))
        normalized.append(item)
    return normalized


def _comparison_token(text: str) -> str:
    return re.sub(r"[^\w]+", "", text.casefold(), flags=re.UNICODE)


def _join_words(words: list[dict[str, object]]) -> str:
    text = " ".join(
        str(word.get("text") or word.get("word") or "").strip()
        for word in words
        if str(word.get("text") or word.get("word") or "").strip()
    )
    return re.sub(r"\s+([,.;:!?])", r"\1", text).strip()
