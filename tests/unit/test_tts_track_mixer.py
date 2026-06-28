from __future__ import annotations

import wave
from pathlib import Path

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


def test_assemble_commentary_track_trims_overlap_before_next_segment(tmp_path: Path) -> None:
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
            {"segment_id": "seg_000002", "audio_path": "tts/seg_000002.wav", "target_start_ms": 50},
        ],
        output_audio=output,
    )

    with wave.open(str(output), "rb") as wav:
        duration_sec = wav.getnframes() / wav.getframerate()
    assert 0.14 <= duration_sec <= 0.19


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
