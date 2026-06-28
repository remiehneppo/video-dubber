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
        allow_chunk_text_fallback=False,
    )

    assert result.source == "word"
    assert result.quality == "word"
    assert [(unit.text, unit.start_ms, unit.end_ms) for unit in result.units] == [
        ("Hello", 11_000, 11_400),
        ("world.", 11_500, 12_000),
    ]


def test_normalize_asr_timestamps_falls_back_to_segments() -> None:
    result = normalize_asr_timestamps(
        {
            "text": "Hello world.",
            "segments": [
                {"text": "Hello world.", "start": 1.0, "end": 2.0},
            ],
        },
        chunk_start_ms=10_000,
        require_timestamps=True,
        allow_chunk_text_fallback=False,
    )

    assert result.source == "segment"
    assert result.quality == "segment"
    assert [(unit.text, unit.start_ms, unit.end_ms) for unit in result.units] == [
        ("Hello world.", 11_000, 12_000),
    ]


def test_normalize_asr_timestamps_raises_when_required_timestamps_missing() -> None:
    with pytest.raises(MissingASRTimestampsError, match="ASR response did not include timestamps"):
        normalize_asr_timestamps(
            {"text": "Hello world."},
            chunk_start_ms=10_000,
            require_timestamps=True,
            allow_chunk_text_fallback=False,
        )
