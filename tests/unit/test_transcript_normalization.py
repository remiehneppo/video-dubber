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


def test_source_normalization_collapses_repeated_asr_loop_before_cue_planning() -> None:
    profile = load_domain_profile("AI, Coding, data science, Education", explicit_profile="generic")
    phrase = "which now has had a chance to be influenced by all the other vectors in this sequence"
    words = []
    for index, token in enumerate((phrase + ", " + phrase + ", " + phrase + ".").replace(",", " ,").replace(".", " .").split()):
        words.append({"text": token, "start_ms": index * 220, "end_ms": index * 220 + 160})

    normalized = normalize_transcript_segments([
        {
            "segment_id": "seg_loop",
            "source_text": " ".join(word["text"] for word in words),
            "start_ms": 0,
            "end_ms": words[-1]["end_ms"],
            "duration_ms": words[-1]["end_ms"],
            "words": words,
        }
    ], profile)

    segment = normalized[0]
    assert segment["source_text"].lower().count("chance to be influenced") == 1
    assert "source_normalized_repetitive_asr_loop" in segment["risk_flags"]
    assert segment["normalization_edits"][0]["rule_id"] == "generic.repeated_asr_loop"
    assert max(int(word["end_ms"]) for word in segment["words"]) < 5000

    cues = build_dubbing_cues(normalized, target_duration_ms=4000, min_duration_ms=1500, max_duration_ms=6000)
    joined = " ".join(str(cue["source_text"]) for cue in cues).lower()
    assert joined.count("chance to be influenced") == 1


def test_source_normalization_collapses_repeated_asr_loop_across_segments() -> None:
    profile = load_domain_profile("AI", explicit_profile="generic")
    phrase = "which now has had a chance to be influenced by all the other vectors in this sequence"
    texts = [f"At the end, {phrase}, {phrase}, {phrase},", f"{phrase}, {phrase},"]
    cursor = 0
    segments = []
    for segment_index, text in enumerate(texts, start=1):
        words = []
        for token in text.replace(",", " ,").split():
            words.append({"text": token, "start_ms": cursor, "end_ms": cursor + 160})
            cursor += 220
        segments.append(
            {
                "segment_id": f"seg_{segment_index:06d}",
                "source_text": text,
                "start_ms": words[0]["start_ms"],
                "end_ms": words[-1]["end_ms"],
                "duration_ms": words[-1]["end_ms"] - words[0]["start_ms"],
                "words": words,
            }
        )

    normalized = normalize_transcript_segments(segments, profile)

    joined = " ".join(str(segment["source_text"]) for segment in normalized).lower()
    assert joined.count("chance to be influenced") == 1
    assert [segment["segment_id"] for segment in normalized] == ["seg_000001"]
    assert normalized[0]["end_ms"] < segments[1]["start_ms"]
    assert normalized[0]["normalization_edits"][0]["removed_segment_ids"] == [
        "seg_000001",
        "seg_000002",
    ]
