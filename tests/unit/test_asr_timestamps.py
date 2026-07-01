from __future__ import annotations

import pytest

from dubber.asr.timestamps import MissingASRTimestampsError, normalize_asr_timestamps


def test_normalize_asr_timestamps_prefers_words() -> None:
    result = normalize_asr_timestamps(
        {
            "text": "Hello world.",
            "words": [
                {"word": "Hello", "start": 1.0, "end": 1.4},
                {"word": "world.", "start": 1.5, "end": 2.0},
            ],
        },
        chunk_start_ms=10_000,
        require_timestamps=True,
        require_word_timestamps=True,
        allow_chunk_text_fallback=False,
    )

    assert result.source == "word"
    assert result.quality == "word"
    assert [(unit.text, unit.start_ms, unit.end_ms) for unit in result.units] == [
        ("Hello", 11_000, 11_400),
        ("world.", 11_500, 12_000),
    ]


def test_normalize_asr_timestamps_falls_back_to_segments_when_word_timestamps_are_not_required() -> None:
    result = normalize_asr_timestamps(
        {
            "text": "Hello world.",
            "segments": [
                {"text": "Hello world.", "start": 1.0, "end": 2.0},
            ],
        },
        chunk_start_ms=10_000,
        require_timestamps=True,
        require_word_timestamps=False,
        allow_chunk_text_fallback=False,
    )

    assert result.source == "segment"
    assert result.quality == "segment"
    assert [(unit.text, unit.start_ms, unit.end_ms) for unit in result.units] == [
        ("Hello world.", 11_000, 12_000),
    ]


def test_normalize_asr_timestamps_raises_for_segment_only_when_word_timestamps_required() -> None:
    with pytest.raises(MissingASRTimestampsError, match="word-level timestamps"):
        normalize_asr_timestamps(
            {
                "text": "Hello world.",
                "segments": [{"text": "Hello world.", "start": 1.0, "end": 2.0}],
            },
            chunk_start_ms=10_000,
            require_timestamps=True,
            require_word_timestamps=True,
            allow_chunk_text_fallback=False,
        )


def test_normalize_asr_timestamps_rejects_partial_word_timestamp_payload() -> None:
    with pytest.raises(MissingASRTimestampsError, match="incomplete or out of order"):
        normalize_asr_timestamps(
            {
                "text": "Hello broken world.",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.4},
                    {"word": "broken", "start": 0.5},
                    {"word": "world.", "start": 0.8, "end": 1.2},
                ],
            },
            chunk_start_ms=0,
            require_timestamps=True,
            require_word_timestamps=True,
            allow_chunk_text_fallback=False,
        )


def test_normalize_asr_timestamps_repairs_zero_duration_word_timestamp_payload() -> None:
    result = normalize_asr_timestamps(
        {
            "text": "In fact, a nice way.",
            "words": [
                {"word": "In", "start": 0.0, "end": 0.18},
                {"word": "fact,", "start": 0.18, "end": 0.18},
                {"word": "a", "start": 0.18, "end": 0.36},
                {"word": "nice", "start": 0.36, "end": 0.60},
            ],
        },
        chunk_start_ms=10_000,
        require_timestamps=True,
        require_word_timestamps=True,
        allow_chunk_text_fallback=False,
    )

    assert result.source == "word"
    assert result.quality == "word"
    assert result.risk_flags == ["word_timestamp_repaired"]
    assert [unit.text for unit in result.units] == ["In", "fact,", "a", "nice"]
    assert all(
        current.start_ms >= previous.start_ms and current.end_ms >= previous.end_ms
        for previous, current in zip(result.units, result.units[1:])
    )
    assert all(unit.end_ms > unit.start_ms for unit in result.units)


def test_normalize_asr_timestamps_raises_when_required_timestamps_missing() -> None:
    with pytest.raises(MissingASRTimestampsError, match="ASR response did not include timestamps"):
        normalize_asr_timestamps(
            {"text": "Hello world."},
            chunk_start_ms=10_000,
            require_timestamps=True,
            require_word_timestamps=False,
            allow_chunk_text_fallback=False,
        )
