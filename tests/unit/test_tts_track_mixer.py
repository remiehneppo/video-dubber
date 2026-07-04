from __future__ import annotations

import wave
from pathlib import Path

import pytest

from dubber.core.paths import WorkspacePaths
from dubber.providers.ffmpeg import FFmpegAdapter
from dubber.tts.mock import synthesize_tone_wav
from dubber.tts.track_mixer import assemble_commentary_track


def test_assemble_commentary_track_delays_and_mixes_segments(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    ffmpeg = FFmpegAdapter()
    first = paths.tts_dir / "seg_000001.wav"
    second = paths.tts_dir / "seg_000002.wav"
    synthesize_tone_wav(first, 120)
    synthesize_tone_wav(second, 180)

    output = paths.tts_dir / "mix.wav"
    assemble_commentary_track(
        paths=paths,
        ffmpeg=ffmpeg,
        tts_segments=[
            {"segment_id": "seg_000001", "audio_path": "tts/seg_000001.wav", "target_start_ms": 0},
            {"segment_id": "seg_000002", "audio_path": "tts/seg_000002.wav", "target_start_ms": 300},
        ],
        output_audio=output,
    )

    assert output.exists()
    with wave.open(str(output), "rb") as wav:
        assert wav.getframerate() == 44100
        duration_sec = wav.getnframes() / wav.getframerate()
    assert 0.45 <= duration_sec <= 0.55


def test_assemble_commentary_track_fits_overlap_within_max_speedup(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    ffmpeg = FFmpegAdapter()
    first = paths.tts_dir / "seg_000001.wav"
    second = paths.tts_dir / "seg_000002.wav"
    synthesize_tone_wav(first, 120)
    synthesize_tone_wav(second, 120)

    output = paths.tts_dir / "mix.wav"
    assemble_commentary_track(
        paths=paths,
        ffmpeg=ffmpeg,
        tts_segments=[
            {"segment_id": "seg_000001", "audio_path": "tts/seg_000001.wav", "target_start_ms": 0},
            {"segment_id": "seg_000002", "audio_path": "tts/seg_000002.wav", "target_start_ms": 100},
        ],
        output_audio=output,
        max_speedup_ratio=1.3,
    )

    assert (paths.tts_dir / "seg_000001.fit.wav").exists()
    with wave.open(str(output), "rb") as wav:
        duration_sec = wav.getnframes() / wav.getframerate()
    assert 0.20 <= duration_sec <= 0.25


def test_assemble_commentary_track_rejects_overlap_above_max_speedup(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    ffmpeg = FFmpegAdapter()
    first = paths.tts_dir / "seg_000001.wav"
    second = paths.tts_dir / "seg_000002.wav"
    synthesize_tone_wav(first, 120)
    synthesize_tone_wav(second, 120)

    output = paths.tts_dir / "mix.wav"
    with pytest.raises(ValueError, match="seg_000001.*tts_segment_exceeds_max_speedup"):
        assemble_commentary_track(
            paths=paths,
            ffmpeg=ffmpeg,
            tts_segments=[
                {"segment_id": "seg_000001", "audio_path": "tts/seg_000001.wav", "target_start_ms": 0},
                {"segment_id": "seg_000002", "audio_path": "tts/seg_000002.wav", "target_start_ms": 50},
            ],
            output_audio=output,
            max_speedup_ratio=1.3,
        )

    assert not output.exists()


def test_ffmpeg_assemble_commentary_track_disables_amix_normalization(tmp_path: Path, monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(command, check, stdout, stderr):
        seen["command"] = command

    monkeypatch.setattr("dubber.providers.ffmpeg.subprocess.run", fake_run)

    FFmpegAdapter().assemble_commentary_track(
        [(tmp_path / "first.wav", 0), (tmp_path / "second.wav", 100)],
        tmp_path / "mix.wav",
    )

    command = seen["command"]
    assert isinstance(command, list)
    filter_graph = command[command.index("-filter_complex") + 1]
    assert "amix=inputs=2:duration=longest:dropout_transition=0:normalize=0" in filter_graph


class RecordingFFmpeg:
    def __init__(self) -> None:
        self.normalized: list[tuple[Path, Path]] = []
        self.assembled: list[tuple[Path, int]] = []

    def duration_ms(self, audio_path: Path) -> int:
        return 100

    def normalize_loudness(self, input_audio: Path, output_audio: Path) -> None:
        self.normalized.append((input_audio, output_audio))
        synthesize_tone_wav(output_audio, 100)

    def assemble_commentary_track(self, segments: list[tuple[Path, int]], output_audio: Path) -> None:
        self.assembled = list(segments)
        synthesize_tone_wav(output_audio, 250)


def test_assemble_commentary_track_normalizes_each_tts_segment_before_mix(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_test")
    ffmpeg = RecordingFFmpeg()
    first = paths.tts_dir / "seg_000001.wav"
    second = paths.tts_dir / "seg_000002.wav"
    synthesize_tone_wav(first, 100)
    synthesize_tone_wav(second, 100)

    output = paths.tts_dir / "mix.wav"
    assemble_commentary_track(
        paths=paths,
        ffmpeg=ffmpeg,  # type: ignore[arg-type]
        tts_segments=[
            {"segment_id": "seg_000001", "audio_path": "tts/seg_000001.wav", "target_start_ms": 0},
            {"segment_id": "seg_000002", "audio_path": "tts/seg_000002.wav", "target_start_ms": 150},
        ],
        output_audio=output,
    )

    assert ffmpeg.normalized == [
        (first, paths.tts_dir / "seg_000001.loudnorm.wav"),
        (second, paths.tts_dir / "seg_000002.loudnorm.wav"),
    ]
    assert ffmpeg.assembled == [
        (paths.tts_dir / "seg_000001.loudnorm.wav", 0),
        (paths.tts_dir / "seg_000002.loudnorm.wav", 150),
    ]


def test_ffmpeg_normalize_loudness_uses_loudnorm_filter(tmp_path: Path, monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(command, check, stdout, stderr):
        seen["command"] = command

    monkeypatch.setattr("dubber.providers.ffmpeg.subprocess.run", fake_run)

    FFmpegAdapter().normalize_loudness(tmp_path / "input.wav", tmp_path / "output.wav")

    command = seen["command"]
    assert isinstance(command, list)
    assert command[command.index("-af") + 1] == "loudnorm=I=-16:TP=-1.5:LRA=11"
    assert "-ar" in command
    assert str(tmp_path / "output.wav") == command[-1]
