from __future__ import annotations

from dubber.transcript.timeline import build_speech_timeline
from dubber.pipeline.stages import _timeline_overflow_by_cue


def test_speech_timeline_records_speech_and_silence_gaps() -> None:
    timeline = build_speech_timeline(
        [
            {
                "cue_id": "cue_a",
                "start_ms": 1000,
                "end_ms": 3000,
                "source_text": "source a",
                "display_text": "hiển thị a",
                "spoken_text": "đọc a",
                "parent_segment_ids": ["seg_1"],
            },
            {
                "cue_id": "cue_b",
                "start_ms": 3800,
                "end_ms": 5000,
                "source_text": "source b",
                "translated_text": "fallback b",
                "parent_segment_ids": ["seg_2"],
            },
        ],
        total_duration_ms=6500,
        guard_ms=100,
    )

    assert timeline["schema_version"] == "1.0"
    assert [item["cue_id"] for item in timeline["speech_intervals"]] == ["cue_a", "cue_b"]
    assert timeline["speech_intervals"][0]["display_text"] == "hiển thị a"
    assert timeline["speech_intervals"][1]["spoken_text"] == "fallback b"
    assert timeline["silence_intervals"] == [
        {
            "gap_id": "gap_000000",
            "position": "before_first",
            "start_ms": 0,
            "end_ms": 1000,
            "duration_ms": 1000,
            "preceding_cue_id": None,
            "following_cue_id": "cue_a",
            "guard_ms": 100,
            "preserve_pause_ms": 450,
            "usable_overflow_ms": 450,
        },
        {
            "gap_id": "gap_000001",
            "position": "between_speech",
            "start_ms": 3000,
            "end_ms": 3800,
            "duration_ms": 800,
            "preceding_cue_id": "cue_a",
            "following_cue_id": "cue_b",
            "guard_ms": 100,
            "preserve_pause_ms": 360,
            "usable_overflow_ms": 340,
        },
        {
            "gap_id": "gap_000002",
            "position": "after_last",
            "start_ms": 5000,
            "end_ms": 6500,
            "duration_ms": 1500,
            "preceding_cue_id": "cue_b",
            "following_cue_id": None,
            "guard_ms": 100,
            "preserve_pause_ms": 450,
            "usable_overflow_ms": 950,
        },
    ]


def test_speech_timeline_preserves_short_pause_as_not_usable_for_overflow() -> None:
    timeline = build_speech_timeline(
        [
            {"cue_id": "cue_a", "start_ms": 0, "end_ms": 1000},
            {"cue_id": "cue_b", "start_ms": 1200, "end_ms": 2000},
        ],
        guard_ms=100,
    )

    gap = timeline["silence_intervals"][0]
    assert gap["duration_ms"] == 200
    assert gap["preserve_pause_ms"] == 200
    assert gap["usable_overflow_ms"] == 0


def test_tts_overflow_budget_uses_only_gap_after_preceding_cue() -> None:
    timeline = build_speech_timeline(
        [
            {"cue_id": "cue_a", "start_ms": 1000, "end_ms": 2000},
            {"cue_id": "cue_b", "start_ms": 3000, "end_ms": 4000},
        ],
        total_duration_ms=5000,
        guard_ms=100,
    )

    overflow_by_cue = _timeline_overflow_by_cue(timeline)

    assert "cue_a" in overflow_by_cue
    assert "cue_b" in overflow_by_cue
    assert overflow_by_cue["cue_a"] == 450
    assert overflow_by_cue["cue_b"] == 450
    assert None not in overflow_by_cue


def test_speech_timeline_exposes_natural_source_pauses_from_word_timestamps() -> None:
    timeline = build_speech_timeline(
        [
            {
                "cue_id": "cue_1",
                "start_ms": 0,
                "end_ms": 1800,
                "source_text": "one two three",
                "translated_text": "một hai ba",
                "parent_segment_ids": ["seg_1"],
            }
        ],
        source_segments=[
            {
                "segment_id": "seg_1",
                "words": [
                    {"text": "one", "start_ms": 0, "end_ms": 300},
                    {"text": "two", "start_ms": 400, "end_ms": 700},
                    {"text": "three", "start_ms": 1500, "end_ms": 1800},
                ],
            }
        ],
        total_duration_ms=2500,
        source_pause_threshold_ms=250,
    )

    assert [
        (interval["start_ms"], interval["end_ms"])
        for interval in timeline["source_speech_intervals"]
    ] == [(0, 700), (1500, 1800)]
    assert [
        (interval["start_ms"], interval["end_ms"])
        for interval in timeline["source_silence_intervals"]
    ] == [(700, 1500), (1800, 2500)]
    assert timeline["source_pause_threshold_ms"] == 250
