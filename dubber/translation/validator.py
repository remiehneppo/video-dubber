from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dubber.domain.profiles import protected_translation_errors
from dubber.translation.glossary import term_applies


class TranslationValidationError(ValueError):
    pass


@dataclass(frozen=True)
class TranslationValidationReport:
    warnings: list[dict[str, Any]]

    @property
    def warning_count(self) -> int:
        return len(self.warnings)


def validate_translations(
    source_segments: list[Mapping[str, Any]],
    translated_segments: list[Mapping[str, Any]],
    glossary_terms: list[Mapping[str, Any]],
    *,
    max_length_ratio: float = 1.15,
    protected_spans_by_segment: Mapping[str, list[Mapping[str, Any]]] | None = None,
) -> TranslationValidationReport:
    source_ids = [str(segment["segment_id"]) for segment in source_segments]
    translated_ids = [str(segment["segment_id"]) for segment in translated_segments]
    if source_ids != translated_ids:
        raise TranslationValidationError("translated segment ids do not match source segment ids")

    warnings: list[dict[str, Any]] = []
    source_by_id = {str(segment["segment_id"]): segment for segment in source_segments}
    for translated in translated_segments:
        segment_id = str(translated["segment_id"])
        vi_text = str(translated.get("vi_text", "")).strip()
        source_text = str(source_by_id[segment_id].get("source_text", ""))
        if not source_text.strip():
            if vi_text:
                warnings.append(
                    {
                        "segment_id": segment_id,
                        "warning": "empty_source_segment_translated",
                        "length_ratio": 0.0,
                        "max_length_ratio": max_length_ratio,
                    }
                )
            continue
        if not vi_text:
            raise TranslationValidationError(f"vi_text is empty for {segment_id}")

        ratio = _text_length_ratio(source_text, vi_text)
        if ratio > max_length_ratio:
            warnings.append(
                {
                    "segment_id": segment_id,
                    "warning": "length_ratio_exceeded",
                    "length_ratio": ratio,
                    "max_length_ratio": max_length_ratio,
                }
            )

        protected_errors = protected_translation_errors(
            source_text,
            vi_text,
            list((protected_spans_by_segment or {}).get(segment_id, [])),
        )
        if protected_errors:
            raise TranslationValidationError(f"protected span violation for {segment_id}: {'; '.join(protected_errors)}")

        missing_terms = _missing_locked_glossary_terms(
            segment_id,
            source_text,
            vi_text,
            glossary_terms,
            used_terms=[str(term) for term in translated.get("used_terms", [])],
        )
        for original in missing_terms:
            warnings.append({
                "segment_id": segment_id,
                "warning": "locked_glossary_term_missing",
                "original": original,
            })

    return TranslationValidationReport(warnings=warnings)


def _text_length_ratio(source_text: str, vi_text: str) -> float:
    source_len = max(1, len(source_text.strip()))
    return len(vi_text.strip()) / source_len


def _missing_locked_glossary_terms(
    segment_id: str,
    source_text: str,
    vi_text: str,
    glossary_terms: list[Mapping[str, Any]],
    *,
    used_terms: list[str],
) -> list[str]:
    vi_lower = vi_text.lower()
    missing: list[str] = []
    for term in glossary_terms:
        if not bool(term.get("locked", False)):
            continue
        original = str(term.get("original", "")).strip()
        vietnamese = str(term.get("vietnamese", "")).strip()
        if not original or not vietnamese:
            continue
        if term_applies(source_text, original, used_terms=used_terms) and vietnamese.lower() not in vi_lower:
            missing.append(original)
    return missing
