from __future__ import annotations

import math
import wave
from pathlib import Path

from dubber.audio.vad import VadConfig, detect_segments


def test_detect_segments_finds_two_speech_islands(tmp_path: Path) -> None:
    wav_path = tmp_path / "two_islands.wav"
    _write_wav(
        wav_path,
        [
            ("silence", 200),
            ("tone", 500),
            ("silence", 700),
            ("tone", 600),
            ("silence", 200),
        ],
    )

    segments = detect_segments(
        wav_path,
        VadConfig(
            frame_ms=50,
            threshold_ratio=0.2,
            min_duration_ms=200,
            max_duration_ms=2000,
            silence_merge_threshold_ms=200,
        ),
    )

    assert [segment.segment_id for segment in segments] == ["seg_000001", "seg_000002"]
    assert segments[0].start_ms == 200
    assert segments[0].end_ms == 700
    assert segments[1].start_ms == 1400
    assert segments[1].end_ms == 2000
    assert all(segment.split_reason == "vad_energy" for segment in segments)


def test_detect_segments_merges_short_silence_gap(tmp_path: Path) -> None:
    wav_path = tmp_path / "merged.wav"
    _write_wav(
        wav_path,
        [
            ("tone", 300),
            ("silence", 100),
            ("tone", 300),
        ],
    )

    segments = detect_segments(
        wav_path,
        VadConfig(
            frame_ms=50,
            threshold_ratio=0.2,
            min_duration_ms=100,
            max_duration_ms=2000,
            silence_merge_threshold_ms=150,
        ),
    )

    assert len(segments) == 1
    assert segments[0].start_ms == 0
    assert segments[0].end_ms == 700


def test_detect_segments_applies_context_padding_and_merges_segments(tmp_path: Path) -> None:
    wav_path = tmp_path / "padded.wav"
    _write_wav(
        wav_path,
        [
            ("tone", 200),
            ("silence", 200),
            ("tone", 200),
        ],
    )

    segments = detect_segments(
        wav_path,
        VadConfig(
            frame_ms=50,
            threshold_ratio=0.2,
            min_duration_ms=100,
            max_duration_ms=2000,
            silence_merge_threshold_ms=100,
            context_padding_ms=150,
        ),
    )

    assert len(segments) == 1
    assert segments[0].start_ms == 0
    assert segments[0].end_ms == 600


def test_detect_segments_splits_long_interval(tmp_path: Path) -> None:
    wav_path = tmp_path / "long.wav"
    _write_wav(wav_path, [("tone", 1200)])

    segments = detect_segments(
        wav_path,
        VadConfig(
            frame_ms=100,
            threshold_ratio=0.2,
            min_duration_ms=100,
            max_duration_ms=500,
            silence_merge_threshold_ms=100,
        ),
    )

    assert [(segment.start_ms, segment.end_ms) for segment in segments] == [
        (0, 500),
        (500, 1000),
        (1000, 1200),
    ]
    assert segments[0].split_reason == "vad_soft_split"


def test_detect_segments_falls_back_to_full_audio_when_no_speech(tmp_path: Path) -> None:
    wav_path = tmp_path / "silence.wav"
    _write_wav(wav_path, [("silence", 600)])

    segments = detect_segments(wav_path, VadConfig(frame_ms=100, threshold_ratio=0.2))

    assert len(segments) == 1
    assert segments[0].start_ms == 0
    assert segments[0].end_ms == 600
    assert segments[0].risk_flags == ["no_speech_detected"]


def test_asr_context_chunks_merge_until_target_minimum(tmp_path: Path) -> None:
    wav_path = tmp_path / "target_min_chunks.wav"
    _write_wav(
        wav_path,
        [
            ("tone", 1000),
            ("silence", 3000),
            ("tone", 1000),
            ("silence", 3000),
            ("tone", 1000),
        ],
    )

    segments = detect_segments(
        wav_path,
        VadConfig(
            mode="asr_context_chunks",
            frame_ms=100,
            threshold_ratio=0.2,
            min_speech_duration_ms=500,
            target_min_chunk_ms=5000,
            preferred_max_chunk_ms=9000,
            hard_max_chunk_ms=12000,
            max_duration_ms=12000,
            silence_merge_threshold_ms=500,
            context_padding_ms=0,
        ),
    )

    assert [(segment.start_ms, segment.end_ms) for segment in segments] == [(0, 5000), (8000, 9000)]
    assert segments[0].split_reason == "vad_context_merge"


def test_asr_context_chunks_split_at_silence_before_hard_max(tmp_path: Path) -> None:
    wav_path = tmp_path / "silence_split.wav"
    _write_wav(
        wav_path,
        [
            ("tone", 4000),
            ("silence", 1000),
            ("tone", 4000),
            ("silence", 1000),
            ("tone", 4000),
        ],
    )

    segments = detect_segments(
        wav_path,
        VadConfig(
            mode="asr_context_chunks",
            frame_ms=100,
            threshold_ratio=0.2,
            min_speech_duration_ms=500,
            target_min_chunk_ms=4000,
            preferred_max_chunk_ms=7000,
            hard_max_chunk_ms=9500,
            max_duration_ms=9500,
            silence_merge_threshold_ms=5000,
            context_padding_ms=0,
        ),
    )

    assert [(segment.start_ms, segment.end_ms) for segment in segments] == [(0, 5000), (5000, 10000), (10000, 14000)]
    assert segments[0].split_reason == "vad_silence_split"
    assert "hard_split" not in segments[0].risk_flags


def _write_wav(path: Path, chunks: list[tuple[str, int]], sample_rate: int = 1000) -> None:
    frames = bytearray()
    for kind, duration_ms in chunks:
        sample_count = int(sample_rate * duration_ms / 1000)
        for index in range(sample_count):
            if kind == "tone":
                value = int(10000 * math.sin(2 * math.pi * 10 * index / sample_rate))
            else:
                value = 0
            frames.extend(value.to_bytes(2, "little", signed=True))

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
