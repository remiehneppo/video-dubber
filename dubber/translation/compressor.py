from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dubber.translation.glossary import term_applies


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

    used_terms = [str(term) for term in segment.get("used_terms", [])]
    if len(original_text) <= max_chars and _locked_terms_present(source_text, original_text, glossary_terms, used_terms=used_terms):
        return CompressionResult(segment_id=segment_id, vi_text=original_text, warnings=[])

    compressed = original_text
    for phrase in FILLER_PHRASES:
        compressed = compressed.replace(phrase, "")
    compressed = " ".join(compressed.split()).strip()

    locked_terms_missing = not _locked_terms_present(source_text, compressed, glossary_terms, used_terms=used_terms)
    if compressed != original_text:
        warnings.append("compressed_for_length")
    if len(compressed) > max_chars:
        warnings.append("length_compression_required")
    if locked_terms_missing:
        warnings.append("locked_glossary_missing")

    return CompressionResult(segment_id=segment_id, vi_text=compressed, warnings=warnings)


def _locked_terms_present(
    source_text: str,
    vi_text: str,
    glossary_terms: list[Mapping[str, Any]],
    *,
    used_terms: list[str] | None = None,
) -> bool:
    vi_lower = vi_text.lower()
    for term in glossary_terms:
        if not term.get("locked", False):
            continue
        original = str(term.get("original", "")).strip().lower()
        vietnamese = str(term.get("vietnamese", "")).strip().lower()
        if original and vietnamese and term_applies(source_text, original, used_terms=used_terms or []) and vietnamese not in vi_lower:
            return False
    return True
