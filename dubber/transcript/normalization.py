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
    return normalized


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
