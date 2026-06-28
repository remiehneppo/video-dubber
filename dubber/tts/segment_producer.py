from __future__ import annotations

from pathlib import Path
from typing import Any

from dubber.core.paths import WorkspacePaths
from dubber.providers.factory import ProviderBundle
from dubber.providers.ffmpeg import FFmpegAdapter
from dubber.tts.aligner import apply_time_stretch
from dubber.tts.audio_quality import TTSQualityReport, analyze_tts_wav
from dubber.tts.duration_planner import plan_segment_duration
from dubber.tts.mock import synthesize_silence_wav, synthesize_tone_wav
from dubber.tts.rephrase import rephrase_tts_text


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
    quality_retry_attempts: int = 3,
    rephrase_attempts: int = 2,
    max_speedup_ratio: float = 1.3,
    min_rms: float = 500,
    silence_rms_threshold: int = 120,
    max_internal_silence_ms: int = 2500,
    clipping_peak_threshold: int = 32760,
    max_clipped_sample_ratio: float = 0.001,
) -> dict[str, Any]:
    segment_id = str(segment["segment_id"])
    raw_audio_path = paths.tts_dir / f"{segment_id}.raw.wav"
    if not text.strip():
        tts_duration_ms = int(segment["duration_ms"])
        synthesize_silence_wav(raw_audio_path, tts_duration_ms)
        return _align_segment(
            paths=paths,
            segment=segment,
            raw_audio_path=raw_audio_path,
            tts_duration_ms=tts_duration_ms,
            provider_metadata={"content_type": "audio/wav", "generated_silence": True},
            synthesis_attempts=0,
            rephrase_attempts=0,
            final_text="",
            source_text=text,
            max_speedup_ratio=max_speedup_ratio,
        )

    tts_voice = voice if voice != "default" else getattr(provider_bundle.tts, "voice", voice)
    source_text = text
    current_text = text
    synthesis_attempts = 0
    rephrase_count = 0
    quality_attempts: list[dict[str, object]] = []
    provider_metadata: dict[str, Any] = {}
    last_report: TTSQualityReport | None = None
    orig_ms = int(segment["duration_ms"])

    while True:
        for _ in range(max(1, quality_retry_attempts)):
            synthesis_attempts += 1
            tts_result = await provider_bundle.tts.synthesize(current_text, voice=tts_voice, output_path=raw_audio_path)
            provider_metadata = tts_result.provider_metadata
            tts_duration_ms = tts_result.duration_ms or ffmpeg.duration_ms(tts_result.audio_path)
            last_report = analyze_tts_wav(
                tts_result.audio_path,
                min_rms=min_rms,
                silence_rms_threshold=silence_rms_threshold,
                max_internal_silence_ms=max_internal_silence_ms,
                clipping_peak_threshold=clipping_peak_threshold,
                max_clipped_sample_ratio=max_clipped_sample_ratio,
            )
            tts_duration_ms = last_report.duration_ms or tts_duration_ms
            quality_attempts.append(last_report.to_dict())
            if last_report.ok:
                break
        else:
            raise _tts_quality_error(
                segment_id,
                text=current_text,
                synthesis_attempts=synthesis_attempts,
                rephrase_attempts=rephrase_count,
                report=last_report,
            )

        assert last_report is not None
        speedup_ratio = tts_duration_ms / orig_ms
        if speedup_ratio <= max_speedup_ratio:
            return _align_segment(
                paths=paths,
                segment=segment,
                raw_audio_path=raw_audio_path,
                tts_duration_ms=tts_duration_ms,
                provider_metadata=provider_metadata,
                quality_report=last_report,
                quality_attempts=quality_attempts,
                synthesis_attempts=synthesis_attempts,
                rephrase_attempts=rephrase_count,
                final_text=current_text,
                source_text=source_text,
                max_speedup_ratio=max_speedup_ratio,
            )

        if rephrase_count >= rephrase_attempts:
            raise ValueError(
                f"{segment_id}: tts_duration_exceeds_max_speedup "
                f"duration_ms={tts_duration_ms} target_ms={orig_ms} ratio={speedup_ratio:.3f} "
                f"max_speedup_ratio={max_speedup_ratio:.3f} rms={last_report.rms} "
                f"max_internal_silence_ms={last_report.max_internal_silence_ms} "
                f"clipped_sample_ratio={last_report.clipped_sample_ratio} "
                f"source_text_chars={len(source_text)} final_text_chars={len(current_text)}"
            )
        rephrase_count += 1
        current_text = await rephrase_tts_text(
            provider_bundle.llm,
            text=current_text,
            target_duration_ms=orig_ms,
            current_duration_ms=tts_duration_ms,
            segment_id=segment_id,
        )


def _align_segment(
    *,
    paths: WorkspacePaths,
    segment: dict[str, Any],
    raw_audio_path: Path,
    tts_duration_ms: int,
    provider_metadata: dict[str, Any],
    quality_report: TTSQualityReport | None = None,
    quality_attempts: list[dict[str, object]] | None = None,
    synthesis_attempts: int = 1,
    rephrase_attempts: int = 0,
    final_text: str | None = None,
    source_text: str | None = None,
    max_speedup_ratio: float = 1.3,
) -> dict[str, Any]:
    segment_id = str(segment["segment_id"])
    orig_ms = int(segment["duration_ms"])
    aligned_audio_path = paths.tts_dir / f"{segment_id}.wav"
    timing_plan = plan_segment_duration(
        segment_id,
        orig_duration_ms=orig_ms,
        tts_duration_ms=tts_duration_ms,
        speedup_hard_limit=max_speedup_ratio,
    )
    apply_time_stretch(
        raw_audio_path,
        aligned_audio_path,
        timing_plan.stretch_ratio if timing_plan.action == "time_stretch" else 1.0,
    )
    row = {
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
        "synthesis_attempts": synthesis_attempts,
        "rephrase_attempts": rephrase_attempts,
    }
    if quality_report is not None:
        row["quality_report"] = quality_report.to_dict()
    if quality_attempts is not None:
        row["quality_attempts"] = quality_attempts
    if final_text is not None:
        row["final_text_chars"] = len(final_text)
    if source_text is not None:
        row["source_text_chars"] = len(source_text)
    return row


def _tts_quality_error(
    segment_id: str,
    *,
    text: str,
    synthesis_attempts: int,
    rephrase_attempts: int,
    report: TTSQualityReport | None,
) -> ValueError:
    if report is None:
        return ValueError(f"{segment_id}: tts_audio_quality_failed no_report text_chars={len(text)}")
    return ValueError(
        f"{segment_id}: tts_audio_quality_failed warnings={','.join(report.warnings)} "
        f"duration_ms={report.duration_ms} rms={report.rms} peak={report.peak} "
        f"max_internal_silence_ms={report.max_internal_silence_ms} "
        f"clipped_sample_ratio={report.clipped_sample_ratio} "
        f"synthesis_attempts={synthesis_attempts} rephrase_attempts={rephrase_attempts} "
        f"text_chars={len(text)}"
    )
