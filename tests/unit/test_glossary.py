from __future__ import annotations

from dubber.translation.glossary import normalize_glossary_terms, term_applies


def test_glossary_deduplicates_by_original_and_keeps_highest_confidence_translation() -> None:
    terms = normalize_glossary_terms(
        [
            {"term_id": "one", "original": "Calculus", "vietnamese": "phép tính", "confidence": 0.4, "source_segments": ["seg_1"]},
            {"term_id": "two", "original": " calculus ", "vietnamese": "giải tích", "confidence": 0.9, "source_segments": ["seg_2", "seg_1"]},
        ]
    )

    assert len(terms) == 1
    assert terms[0]["vietnamese"] == "giải tích"
    assert terms[0]["confidence"] == 0.9
    assert terms[0]["source_segments"] == ["seg_1", "seg_2"]


def test_term_matching_uses_word_boundaries_and_requires_used_terms_for_one_character() -> None:
    assert term_applies("calculus is useful", "calculus") is True
    assert term_applies("recalculus", "calculus") is False
    assert term_applies("a calculus lesson", "a") is False
    assert term_applies("a calculus lesson", "a", used_terms=["a"]) is True


def test_glossary_normalization_preserves_locked_domain_metadata() -> None:
    terms = normalize_glossary_terms([
        {
            "term_id": "protected_dr",
            "original": "dr",
            "vietnamese": "d r",
            "confidence": 1.0,
            "locked": True,
            "protected": True,
            "spoken": "d r",
            "display": "dr",
            "forbidden": ["doctor", "bác sĩ", "tiến sĩ"],
            "source_segments": ["seg_1"],
        }
    ])

    assert terms[0]["protected"] is True
    assert terms[0]["spoken"] == "d r"
    assert terms[0]["display"] == "dr"
    assert terms[0]["forbidden"] == ["doctor", "bác sĩ", "tiến sĩ"]


def test_protected_domain_term_cannot_be_overridden_by_llm_confidence() -> None:
    terms = normalize_glossary_terms([
        {
            "term_id": "protected_dr",
            "original": "dr",
            "vietnamese": "d r",
            "confidence": 1.0,
            "locked": True,
            "protected": True,
            "spoken": "d r",
            "display": "dr",
            "forbidden": ["doctor"],
            "source_segments": ["seg_1"],
        },
        {
            "term_id": "llm_dr",
            "original": "dr",
            "vietnamese": "bác sĩ",
            "confidence": 99.0,
            "locked": True,
            "source_segments": ["seg_1"],
        },
    ])

    assert terms[0]["term_id"] == "protected_dr"
    assert terms[0]["vietnamese"] == "d r"
