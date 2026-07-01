from __future__ import annotations

import math
import wave
from pathlib import Path

from dubber.audio.vad import VadSegment
from dubber.core.models import DubberConfig, VadConfig
from dubber.core.paths import WorkspacePaths
from dubber.orchestrator.artifact_manifest import ArtifactManifest
from dubber.orchestrator.checkpoint_store import CheckpointStore
from dubber.pipeline.stage_context import StageContext
from dubber.pipeline.stages import run_vad


def test_run_vad_uses_configured_max_duration(tmp_path: Path) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_vad")
    _write_wav(paths.audio_dir / "vocals.wav", duration_ms=1200)
    store = CheckpointStore.create(paths.job_state_file, job_id="job_vad", input_file=Path("input/video.mp4"))
    manifest = ArtifactManifest.create("job_vad", paths.manifest_file)
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=None,
        config=DubberConfig(
            vad=VadConfig(
                frame_ms=100,
                threshold_ratio=0.2,
                min_duration_ms=100,
                    max_duration_ms=500,
                    silence_merge_threshold_ms=100,
                    context_padding_ms=50,
                    soft_split_allowed=True,
                )
        ),
    )

    run_vad(ctx)

    segments = ctx.artifact_json("segments.v1.json")["segments"]
    assert [(segment["start_ms"], segment["end_ms"]) for segment in segments] == [
        (0, 500),
        (500, 1000),
        (1000, 1200),
    ]



def test_run_vad_passes_silero_config_to_audio_backend(tmp_path: Path, monkeypatch) -> None:
    paths = WorkspacePaths.create(tmp_path, "job_vad_silero")
    _write_wav(paths.audio_dir / "vocals.wav", duration_ms=1200)
    store = CheckpointStore.create(paths.job_state_file, job_id="job_vad_silero", input_file=Path("input/video.mp4"))
    manifest = ArtifactManifest.create("job_vad_silero", paths.manifest_file)
    seen = {}

    def fake_detect_segments(wav_path: Path, config) -> list[VadSegment]:
        seen["wav_path"] = wav_path
        seen["config"] = config
        return [
            VadSegment(
                segment_id="seg_000001",
                start_ms=0,
                end_ms=1000,
                duration_ms=1000,
                silence_before_ms=0,
                silence_after_ms=0,
                speech_probability=1.0,
                split_reason="vad_silero",
            )
        ]

    monkeypatch.setattr("dubber.pipeline.stages.detect_segments", fake_detect_segments)
    ctx = StageContext(
        paths=paths,
        store=store,
        manifest=manifest,
        ffmpeg=None,
        config=DubberConfig(
            vad=VadConfig(
                mode="silero_vad",
                silero_model_path=Path("models/custom.onnx"),
                silero_threshold=0.6,
                min_silence_duration_ms=450,
                speech_padding_ms=150,
                max_vad_chunk_ms=25000,
                merge_gap_ms=200,
            )
        ),
    )

    run_vad(ctx)

    config = seen["config"]
    assert config.mode == "silero_vad"
    assert config.silero_model_path == Path("models/custom.onnx")
    assert config.silero_threshold == 0.6
    assert config.min_silence_duration_ms == 450
    assert config.speech_padding_ms == 150
    assert config.max_vad_chunk_ms == 25000
    assert config.merge_gap_ms == 200


def _write_wav(path: Path, *, duration_ms: int, sample_rate: int = 1000) -> None:
    frames = bytearray()
    sample_count = int(sample_rate * duration_ms / 1000)
    for index in range(sample_count):
        value = int(10000 * math.sin(2 * math.pi * 10 * index / sample_rate))
        frames.extend(value.to_bytes(2, "little", signed=True))

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
