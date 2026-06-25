from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompressionResult:
    segment_id: str
    vi_text: str
    warnings: list[str]


FILLER_PHRASES = [
    "Bây giờ ",
    "bây giờ ",
    "chúng ta hãy cùng nhau ",
    "hãy cùng nhau ",
    "thật chi tiết ",
    "trong ví dụ này",
    "một cách ",
    "rất ",
]


def compress_segment_translation(
    segment: Mapping[str, Any],
    glossary_terms: list[Mapping[str, Any]],
    *,
    max_length_ratio: float = 1.15,
) -> CompressionResult:
    segment_id = str(segment["segment_id"])
    source_text = str(segment.get("source_text", "")).strip()
    original_text = str(segment.get("vi_text", "")).strip()
    max_chars = max(1, int(max(1, len(source_text)) * max_length_ratio))
    warnings: list[str] = []

    if len(original_text) <= max_chars and _locked_terms_present(source_text, original_text, glossary_terms):
        return CompressionResult(segment_id=segment_id, vi_text=original_text, warnings=[])

    compressed = original_text
    for phrase in FILLER_PHRASES:
        compressed = compressed.replace(phrase, "")
    compressed = " ".join(compressed.split()).strip(" ,.")

    if len(compressed) > max_chars:
        compressed = _trim_to_limit_preserving_terms(compressed, max_chars, source_text, glossary_terms)

    compressed, inserted = _ensure_locked_terms(source_text, compressed, glossary_terms)
    if compressed != original_text:
        warnings.append("compressed_for_length")
    if inserted:
        warnings.append("locked_glossary_reinserted")

    return CompressionResult(segment_id=segment_id, vi_text=compressed, warnings=warnings)


def _locked_terms_present(
    source_text: str,
    vi_text: str,
    glossary_terms: list[Mapping[str, Any]],
) -> bool:
    source_lower = source_text.lower()
    vi_lower = vi_text.lower()
    for term in glossary_terms:
        if not term.get("locked", False):
            continue
        original = str(term.get("original", "")).strip().lower()
        vietnamese = str(term.get("vietnamese", "")).strip().lower()
        if original and vietnamese and original in source_lower and vietnamese not in vi_lower:
            return False
    return True


def _ensure_locked_terms(
    source_text: str,
    vi_text: str,
    glossary_terms: list[Mapping[str, Any]],
) -> tuple[str, bool]:
    inserted = False
    result = vi_text
    source_lower = source_text.lower()
    result_lower = result.lower()
    for term in glossary_terms:
        if not term.get("locked", False):
            continue
        original = str(term.get("original", "")).strip()
        vietnamese = str(term.get("vietnamese", "")).strip()
        if original and vietnamese and original.lower() in source_lower and vietnamese.lower() not in result_lower:
            result = f"{vietnamese}: {result}" if result else vietnamese
            result_lower = result.lower()
            inserted = True
    return result, inserted


def _trim_to_limit_preserving_terms(
    text: str,
    max_chars: int,
    source_text: str,
    glossary_terms: list[Mapping[str, Any]],
) -> str:
    protected_terms = [
        str(term.get("vietnamese", "")).strip()
        for term in glossary_terms
        if term.get("locked", False)
        and str(term.get("original", "")).strip().lower() in source_text.lower()
        and str(term.get("vietnamese", "")).strip()
    ]
    if any(term.lower() in text.lower() for term in protected_terms):
        for term in protected_terms:
            index = text.lower().find(term.lower())
            if index >= 0:
                window_start = max(0, index - max_chars // 3)
                window = text[window_start : window_start + max_chars]
                return window.strip(" ,.")

    words = text.split()
    result_words: list[str] = []
    for word in words:
        candidate = " ".join([*result_words, word])
        if len(candidate) > max_chars:
            break
        result_words.append(word)
    return " ".join(result_words).strip(" ,.")
