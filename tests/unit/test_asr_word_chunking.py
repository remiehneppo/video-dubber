from __future__ import annotations

from dubber.asr.timestamps import TimestampUnit
from dubber.asr.word_chunking import build_word_timestamp_chunks
from dubber.core.models import ASRChunkingConfig


def test_build_word_timestamp_chunks_splits_at_initial_silence_gap() -> None:
    chunks = build_word_timestamp_chunks(
        [
            TimestampUnit("alpha", 0, 1000),
            TimestampUnit("beta", 6000, 7000),
        ],
        audio_duration_ms=9000,
        config=ASRChunkingConfig(enabled=True),
    )

    assert [(chunk.word_start_ms, chunk.word_end_ms) for chunk in chunks] == [(0, 1000), (6000, 7000)]
    assert [chunk.split_threshold_ms for chunk in chunks] == [5000, 5000]
    assert [unit.text for unit in chunks[0].units] == ["alpha"]
    assert [unit.text for unit in chunks[1].units] == ["beta"]


def test_build_word_timestamp_chunks_lowers_threshold_for_oversized_chunks() -> None:
    chunks = build_word_timestamp_chunks(
        [
            TimestampUnit("alpha", 0, 1000),
            TimestampUnit("beta", 4600, 5000),
            TimestampUnit("gamma", 5200, 9000),
        ],
        audio_duration_ms=10000,
        config=ASRChunkingConfig(
            enabled=True,
            max_chunk_duration_ms=6000,
            initial_silence_ms=5000,
            min_silence_ms=3000,
            silence_step_ms=500,
        ),
    )

    assert [[unit.text for unit in chunk.units] for chunk in chunks] == [["alpha"], ["beta", "gamma"]]
    assert [chunk.split_threshold_ms for chunk in chunks] == [3500, 3500]


def test_build_word_timestamp_chunks_appends_trailing_silence_after_last_word() -> None:
    chunks = build_word_timestamp_chunks(
        [TimestampUnit("alpha", 1000, 2000)],
        audio_duration_ms=4500,
        config=ASRChunkingConfig(enabled=True),
    )

    assert chunks[0].start_ms == 1000
    assert chunks[0].end_ms == 4500
    assert chunks[0].word_end_ms == 2000
    assert chunks[0].trailing_silence_ms == 2500


def test_build_word_timestamp_chunks_caps_trailing_silence() -> None:
    chunks = build_word_timestamp_chunks(
        [TimestampUnit("alpha", 1000, 2000)],
        audio_duration_ms=10000,
        config=ASRChunkingConfig(enabled=True, trailing_silence_cap_ms=3000),
    )

    assert chunks[0].end_ms == 5000
    assert chunks[0].trailing_silence_ms == 3000


def test_build_word_timestamp_chunks_hard_splits_when_no_silence_can_satisfy_max() -> None:
    chunks = build_word_timestamp_chunks(
        [
            TimestampUnit("alpha", 0, 3000),
            TimestampUnit("beta", 3100, 6000),
            TimestampUnit("gamma", 6100, 9000),
        ],
        audio_duration_ms=9000,
        config=ASRChunkingConfig(
            enabled=True,
            max_chunk_duration_ms=5000,
            initial_silence_ms=5000,
            min_silence_ms=500,
            silence_step_ms=500,
            trailing_silence_cap_ms=1,
        ),
    )

    assert [[unit.text for unit in chunk.units] for chunk in chunks] == [["alpha"], ["beta"], ["gamma"]]
    assert all("hard_split" in chunk.risk_flags for chunk in chunks)


def test_build_word_timestamp_chunks_marks_only_forced_split_groups_as_hard_split() -> None:
    chunks = build_word_timestamp_chunks(
        [
            TimestampUnit("clean", 0, 1000),
            TimestampUnit("alpha", 7000, 10000),
            TimestampUnit("beta", 10100, 13000),
            TimestampUnit("gamma", 13100, 16000),
        ],
        audio_duration_ms=16000,
        config=ASRChunkingConfig(
            enabled=True,
            max_chunk_duration_ms=5000,
            initial_silence_ms=5000,
            min_silence_ms=500,
            silence_step_ms=500,
            trailing_silence_cap_ms=1,
        ),
    )

    assert [[unit.text for unit in chunk.units] for chunk in chunks] == [["clean"], ["alpha"], ["beta"], ["gamma"]]
    assert chunks[0].risk_flags == []
    assert all("hard_split" in chunk.risk_flags for chunk in chunks[1:])


def test_build_word_timestamp_chunks_preserves_word_order_and_membership() -> None:
    words = [
        TimestampUnit("one", 0, 500),
        TimestampUnit("two", 700, 1000),
        TimestampUnit("three", 6500, 7000),
        TimestampUnit("four", 7200, 7600),
    ]

    chunks = build_word_timestamp_chunks(
        words,
        audio_duration_ms=8000,
        config=ASRChunkingConfig(enabled=True),
    )

    flattened = [unit for chunk in chunks for unit in chunk.units]
    assert flattened == words
