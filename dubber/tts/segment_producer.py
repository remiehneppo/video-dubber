from __future__ import annotations

from pathlib import Path
from typing import Any

from dubber.core.paths import WorkspacePaths
from dubber.providers.factory import ProviderBundle
from dubber.providers.ffmpeg import FFmpegAdapter
from dubber.tts.aligner import apply_time_stretch
from dubber.tts.duration_planner import plan_segment_duration
from dubber.tts.mock import synthesize_tone_wav


def produce_mock_tts_segment(*, paths: WorkspacePaths, segment: dict[str, Any]) -> dict[str, Any]:
    segment_id = str(segment["segment_id"])
    orig_ms = int(segment["duration_ms"])
    raw_audio_path = paths.tts_dir / f"{segment_id}.raw.wav"
    tts_duration_ms = max(100, int(orig_ms * 1.1))
    synthesize_tone_wav(raw_audio_path, tts_duration_ms)
    return _align_segment(
        paths=paths,
        segment=segment,
        raw_audio_path=raw_audio_path,
        tts_duration_ms=tts_duration_ms,
        provider_metadata={},
    )


async def produce_provider_tts_segment(
    *,
    paths: WorkspacePaths,
    segment: dict[str, Any],
    text: str,
    provider_bundle: ProviderBundle,
    ffmpeg: FFmpegAdapter,
    voice: str = "default",
) -> dict[str, Any]:
    segment_id = str(segment["segment_id"])
    raw_audio_path = paths.tts_dir / f"{segment_id}.raw.wav"
    tts_voice = voice if voice != "default" else getattr(provider_bundle.tts, "voice", voice)
    tts_result = await provider_bundle.tts.synthesize(text, voice=tts_voice, output_path=raw_audio_path)
    tts_duration_ms = tts_result.duration_ms or ffmpeg.duration_ms(tts_result.audio_path)
    return _align_segment(
        paths=paths,
        segment=segment,
        raw_audio_path=raw_audio_path,
        tts_duration_ms=tts_duration_ms,
        provider_metadata=tts_result.provider_metadata,
    )


def _align_segment(
    *,
    paths: WorkspacePaths,
    segment: dict[str, Any],
    raw_audio_path: Path,
    tts_duration_ms: int,
    provider_metadata: dict[str, Any],
) -> dict[str, Any]:
    segment_id = str(segment["segment_id"])
    orig_ms = int(segment["duration_ms"])
    aligned_audio_path = paths.tts_dir / f"{segment_id}.wav"
    timing_plan = plan_segment_duration(
        segment_id,
        orig_duration_ms=orig_ms,
        tts_duration_ms=tts_duration_ms,
    )
    apply_time_stretch(
        raw_audio_path,
        aligned_audio_path,
        timing_plan.stretch_ratio if timing_plan.action == "time_stretch" else 1.0,
    )
    return {
        "segment_id": segment["segment_id"],
        "target_start_ms": int(segment["start_ms"]) + 500,
        "target_end_ms": int(segment["end_ms"]),
        "original_start_ms": int(segment["start_ms"]),
        "original_end_ms": int(segment["end_ms"]),
        "commentary_delay_ms": 500,
        "orig_duration_ms": orig_ms,
        "tts_duration_ms": tts_duration_ms,
        "alignment_action": timing_plan.action,
        "stretch_ratio": timing_plan.stretch_ratio,
        "overflow_ms": timing_plan.overflow_ms,
        "raw_audio_path": paths.to_relative(raw_audio_path),
        "audio_path": paths.to_relative(aligned_audio_path),
        "warnings": timing_plan.warnings,
        "provider_metadata": provider_metadata,
    }
