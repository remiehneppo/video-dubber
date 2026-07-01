from __future__ import annotations

from dubber.domain.profiles import load_domain_profile
from dubber.transcript.cues import build_dubbing_cues
from dubber.transcript.normalization import normalize_transcript_segments


def test_calculus_source_normalization_repairs_doctor_as_dr_with_provenance() -> None:
    profile = load_domain_profile("mathematics", explicit_profile="calculus")
    segments = [
        {
            "segment_id": "seg_1",
            "source_text": "Each ring has thickness doctor, a tiny change in radius.",
            "start_ms": 0,
            "end_ms": 2000,
            "duration_ms": 2000,
            "words": [
                {"text": "Each", "start_ms": 0, "end_ms": 200},
                {"text": "doctor,", "start_ms": 900, "end_ms": 1200},
            ],
        }
    ]

    normalized = normalize_transcript_segments(segments, profile)

    segment = normalized[0]
    assert segment["source_text_raw"] == "Each ring has thickness doctor, a tiny change in radius."
    assert segment["source_text"] == "Each ring has thickness dr, a tiny change in radius."
    assert segment["source_text_normalized"] == segment["source_text"]
    assert segment["normalization_edits"][0]["rule_id"] == "calculus.doctor_to_dr"
    assert segment["normalization_confidence"] == 0.92
    assert "source_normalized_calculus_notation" in segment["risk_flags"]
    assert segment["words"][1]["text"] == "dr,"
    assert segment["words"][1]["raw_text"] == "doctor,"
    assert segment["protected_spans"][0]["canonical"] == "dr"

    cue = build_dubbing_cues(
        normalized,
        target_duration_ms=1200,
        min_duration_ms=500,
        max_duration_ms=2000,
    )[0]
    assert cue["source_text"] == "Each dr,"
    assert cue["source_text_raw"] == "Each doctor,"


def test_source_normalization_does_not_rewrite_doctor_without_calculus_context() -> None:
    profile = load_domain_profile("general")
    normalized = normalize_transcript_segments(
        [{"segment_id": "seg_1", "source_text": "The doctor explains the story."}],
        profile,
    )

    assert normalized[0]["source_text"] == "The doctor explains the story."
    assert normalized[0]["normalization_edits"] == []
