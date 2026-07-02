from __future__ import annotations

from dubber.transcript.cues import build_dubbing_cues
from dubber.transcript.sentences import build_source_sentences


def test_source_sentences_merge_incomplete_clause_across_asr_segments() -> None:
    segments = [
        {
            "segment_id": "seg_000001",
            "source_text": "Changing those parameters will",
            "risk_flags": ["word_timestamp_repaired"],
            "words": [
                {"text": "Changing", "start_ms": 0, "end_ms": 300},
                {"text": "those", "start_ms": 320, "end_ms": 500},
                {"text": "parameters", "start_ms": 520, "end_ms": 900},
                {"text": "will", "start_ms": 920, "end_ms": 1100},
            ],
        },
        {
            "segment_id": "seg_000002",
            "source_text": "change the probabilities.",
            "risk_flags": [],
            "words": [
                {"text": "change", "start_ms": 1120, "end_ms": 1400},
                {"text": "the", "start_ms": 1420, "end_ms": 1540},
                {"text": "probabilities.", "start_ms": 1560, "end_ms": 2100},
            ],
        },
    ]

    sentences = build_source_sentences(segments)

    assert len(sentences) == 1
    assert sentences[0]["source_text"] == "Changing those parameters will change the probabilities."
    assert sentences[0]["parent_segment_ids"] == ["seg_000001", "seg_000002"]
    assert sentences[0]["start_ms"] == 0
    assert sentences[0]["end_ms"] == 2100
    cues = build_dubbing_cues(sentences, target_duration_ms=2000, min_duration_ms=1000, max_duration_ms=3000)
    assert cues[0]["parent_segment_ids"] == ["seg_000001", "seg_000002"]


def test_source_sentences_keep_incomplete_phrase_across_long_pause() -> None:
    segments = [
        {
            "segment_id": "seg_000001",
            "words": [
                {"text": "all", "start_ms": 0, "end_ms": 200},
                {"text": "of", "start_ms": 220, "end_ms": 400},
                {"text": "the", "start_ms": 1500, "end_ms": 1700},
                {"text": "parameters.", "start_ms": 1720, "end_ms": 2200},
            ],
        }
    ]

    sentences = build_source_sentences(segments, natural_pause_ms=700)

    assert [sentence["source_text"] for sentence in sentences] == ["all of the parameters."]
