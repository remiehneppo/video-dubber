from __future__ import annotations

import shutil
from pathlib import Path

from dubber.audio.vad import VadConfig, detect_segments
from dubber.core.enums import StageName, StageStatus
from dubber.core.io import write_json_atomic
from dubber.orchestrator.stage_artifacts import StageArtifacts
from dubber.pipeline.stage_context import StageContext


def run_job_init(ctx: StageContext, copied_input: Path) -> None:
    ctx.store.mark_stage(StageName.JOB_INIT, StageStatus.RUNNING)
    ctx.store.save()
    metadata_path = ctx.paths.input_dir / "input_metadata.v1.json"
    write_json_atomic(metadata_path, ctx.ffmpeg.probe(copied_input))
    StageArtifacts(ctx.paths, ctx.store, ctx.manifest).publish_file(
        stage=StageName.JOB_INIT,
        name="input_metadata",
        path=metadata_path,
    )


def run_audio_extract(ctx: StageContext, copied_input: Path) -> int:
    ctx.store.mark_stage(StageName.AUDIO_EXTRACT, StageStatus.RUNNING)
    ctx.store.save()
    original_wav = ctx.paths.audio_dir / "original.wav"
    ctx.ffmpeg.extract_audio(copied_input, original_wav)
    shutil.copy2(original_wav, ctx.paths.audio_dir / "vocals.wav")
    duration_ms = ctx.ffmpeg.duration_ms(copied_input)
    StageArtifacts(ctx.paths, ctx.store, ctx.manifest).publish_json(
        stage=StageName.AUDIO_EXTRACT,
        name="audio_analysis",
        filename="audio_analysis.v1.json",
        payload={
            "schema_version": "1.0",
            "audio_duration_ms": duration_ms,
            "sample_rate": 44100,
            "channels": 1,
            "source_separation_used": False,
            "source_separation_reason": "mock_vertical_slice",
        },
    )
    return duration_ms


def run_vad(ctx: StageContext) -> None:
    ctx.store.mark_stage(StageName.VAD, StageStatus.RUNNING)
    ctx.store.save()
    segments = detect_segments(
        ctx.paths.audio_dir / "vocals.wav",
        VadConfig(
            min_duration_ms=300,
            max_duration_ms=25_000,
            silence_merge_threshold_ms=400,
        ),
    )
    StageArtifacts(ctx.paths, ctx.store, ctx.manifest).publish_json(
        stage=StageName.VAD,
        name="segments",
        filename="segments.v1.json",
        payload={
            "schema_version": "1.0",
            "job_id": ctx.paths.root.name,
            "source_audio": "audio/vocals.wav",
            "segments": [segment.to_dict() for segment in segments],
        },
    )
