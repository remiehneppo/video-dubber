from __future__ import annotations

from dubber.asr.timestamps import NormalizedASRTimestamps, TimestampUnit
from dubber.core.models import TranscriptSegmentationConfig
from dubber.transcript.segmentation import build_transcript_segments


def test_build_transcript_segments_splits_on_punctuation_after_target_minimum() -> None:
    timestamps = NormalizedASRTimestamps(
        source="word",
        quality="word",
        units=[
            TimestampUnit("This", 0, 500),
            TimestampUnit("is", 600, 900),
            TimestampUnit("first.", 1000, 2500),
            TimestampUnit("This", 3500, 3900),
            TimestampUnit("is", 4000, 4300),
            TimestampUnit("second.", 4400, 6200),
        ],
        risk_flags=[],
    )

    segments = build_transcript_segments(
        [
            {
                "chunk_id": "seg_000001",
                "raw_response_path": "raw/asr/seg_000001.json",
                "timestamps": timestamps,
            }
        ],
        TranscriptSegmentationConfig(
            target_min_segment_ms=2000,
            preferred_max_segment_ms=5000,
            max_segment_ms=8000,
            min_pause_split_ms=600,
            prefer_punctuation_split=True,
        ),
    )

    assert [segment["source_text"] for segment in segments] == ["This is first.", "This is second."]
    assert [segment["segment_id"] for segment in segments] == ["seg_000001", "seg_000002"]
    assert segments[0]["timestamp_source"] == "word"
    assert segments[0]["timestamp_quality"] == "word"


def test_build_transcript_segments_splits_long_segment_on_pause() -> None:
    timestamps = NormalizedASRTimestamps(
        source="word",
        quality="word",
        units=[
            TimestampUnit("alpha", 0, 1000),
            TimestampUnit("beta", 1100, 2000),
            TimestampUnit("gamma", 3600, 4500),
            TimestampUnit("delta", 4600, 5500),
        ],
        risk_flags=[],
    )

    segments = build_transcript_segments(
        [{"chunk_id": "seg_000001", "raw_response_path": "raw.json", "timestamps": timestamps}],
        TranscriptSegmentationConfig(
            target_min_segment_ms=1500,
            preferred_max_segment_ms=3000,
            max_segment_ms=8000,
            min_pause_split_ms=1000,
            prefer_punctuation_split=True,
        ),
    )

    assert [segment["source_text"] for segment in segments] == ["alpha beta", "gamma delta"]


def test_build_transcript_segments_marks_segment_timestamp_fallback() -> None:
    timestamps = NormalizedASRTimestamps(
        source="segment",
        quality="segment",
        units=[TimestampUnit("A full ASR segment.", 1000, 4000)],
        risk_flags=["word_timestamps_missing"],
    )

    segments = build_transcript_segments(
        [{"chunk_id": "seg_000001", "raw_response_path": "raw.json", "timestamps": timestamps}],
        TranscriptSegmentationConfig(),
    )

    assert len(segments) == 1
    assert segments[0]["timestamp_source"] == "segment"
    assert "word_timestamps_missing" in segments[0]["risk_flags"]
