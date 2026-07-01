from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


def term_applies(source_text: str, original: str, *, used_terms: Iterable[str] = ()) -> bool:
    """Return whether a glossary original occurs as a complete token/phrase."""
    normalized = original.strip()
    if not normalized:
        return False
    if len(normalized) == 1:
        return normalized.casefold() in {str(term).strip().casefold() for term in used_terms}
    pattern = rf"(?<!\w){re.escape(normalized)}(?!\w)"
    return re.search(pattern, source_text, flags=re.IGNORECASE | re.UNICODE) is not None


def normalize_glossary_terms(terms: Iterable[Mapping[str, Any]]) -> list[dict[str, object]]:
    """Deduplicate by original term, retaining the highest-confidence definition."""
    merged: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for position, term in enumerate(terms, start=1):
        original = str(term.get("original", "")).strip()
        if not original:
            continue
        key = original.casefold()
        confidence = float(term.get("confidence", 1.0))
        segments = sorted({str(item) for item in term.get("source_segments", [])})
        candidate = {
            "term_id": str(term.get("term_id", f"term_{position:04d}")),
            "original": original,
            "vietnamese": str(term.get("vietnamese", "")).strip(),
            "category": str(term.get("category", "term")),
            "confidence": confidence,
            "locked": bool(term.get("locked", True)),
            "source_segments": segments,
            "notes": str(term.get("notes", "")),
            "protected": bool(term.get("protected", False)),
            "spoken": str(term.get("spoken", "")).strip(),
            "display": str(term.get("display", "")).strip(),
            "forbidden": list(dict.fromkeys(str(item) for item in term.get("forbidden", []))),
        }
        current = merged.get(key)
        if current is None:
            merged[key] = candidate
            order.append(key)
            continue
        all_segments = sorted({*map(str, current["source_segments"]), *segments})
        current_protected = bool(current.get("protected", False))
        candidate_protected = bool(candidate.get("protected", False))
        candidate_wins = (
            candidate_protected and not current_protected
        ) or (
            candidate_protected == current_protected
            and confidence > float(current["confidence"])
        )
        if candidate_wins:
            candidate["source_segments"] = all_segments
            candidate["locked"] = bool(current["locked"]) or bool(candidate["locked"])
            _merge_domain_metadata(candidate, current)
            merged[key] = candidate
        else:
            current["source_segments"] = all_segments
            current["locked"] = bool(current["locked"]) or bool(candidate["locked"])
            _merge_domain_metadata(current, candidate)
    return [merged[key] for key in order]


def _merge_domain_metadata(target: dict[str, object], other: Mapping[str, Any]) -> None:
    target_protected = bool(target.get("protected", False))
    other_protected = bool(other.get("protected", False))
    if other_protected and not target_protected:
        target["spoken"] = str(other.get("spoken", "")).strip()
        target["display"] = str(other.get("display", "")).strip()
    target["protected"] = target_protected or other_protected
    target["forbidden"] = list(dict.fromkeys([
        *(str(item) for item in target.get("forbidden", [])),
        *(str(item) for item in other.get("forbidden", [])),
    ]))
