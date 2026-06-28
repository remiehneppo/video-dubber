from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dubber.core.paths import WorkspacePaths
from dubber.providers.ffmpeg import FFmpegAdapter
from dubber.tts.aligner import apply_time_stretch

logger = logging.getLogger(__name__)


def assemble_commentary_track(
    *,
    paths: WorkspacePaths,
    ffmpeg: FFmpegAdapter,
    tts_segments: list[dict[str, Any]],
    output_audio: Path,
) -> None:
    if not tts_segments:
        raise ValueError("tts_segments must not be empty")

    ordered_segments = sorted(
        tts_segments,
        key=lambda segment: (int(segment["target_start_ms"]), str(segment.get("segment_id", segment["audio_path"]))),
    )
    inputs: list[tuple[Path, int]] = []
    for index, segment in enumerate(ordered_segments):
        audio_path = paths.resolve_relative(segment["audio_path"])
        start_ms = int(segment["target_start_ms"])
        audio_duration_ms = ffmpeg.duration_ms(audio_path)
        next_start_ms = (
            int(ordered_segments[index + 1]["target_start_ms"]) if index + 1 < len(ordered_segments) else None
        )
        mixed_audio_path = audio_path
        if next_start_ms is not None:
            slot_ms = max(1, next_start_ms - start_ms)
            if audio_duration_ms > slot_ms:
                stretch_ratio = audio_duration_ms / slot_ms
                mixed_audio_path = paths.tts_dir / f"{segment['segment_id']}.fit.wav"
                apply_time_stretch(audio_path, mixed_audio_path, stretch_ratio)
                logger.warning(
                    "stage tts commentary time_stretched segment=%s audio_duration_ms=%s slot_ms=%s stretch_ratio=%.3f start_ms=%s next_start_ms=%s",
                    segment["segment_id"],
                    audio_duration_ms,
                    slot_ms,
                    stretch_ratio,
                    start_ms,
                    next_start_ms,
                )
        inputs.append((mixed_audio_path, start_ms))
    ffmpeg.assemble_commentary_track(inputs, output_audio)
