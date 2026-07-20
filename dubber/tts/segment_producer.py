from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

from dubber.core.io import write_json_atomic
from dubber.core.concurrency import ProviderConcurrency
from dubber.core.paths import WorkspacePaths
from dubber.domain.profiles import protected_translation_errors
from dubber.providers.factory import ProviderBundle
from dubber.providers.ffmpeg import FFmpegAdapter
from dubber.tts.aligner import apply_time_stretch
from dubber.tts.audio_quality import TTSQualityReport, analyze_tts_wav, compact_excessive_internal_silence, trim_edge_silence
from dubber.tts.clause_builder import build_tts_clauses
from dubber.tts.duration_planner import plan_segment_duration
from dubber.tts.mock import synthesize_silence_wav, synthesize_tone_wav
from dubber.tts.rephrase import rephrase_tts_text
from dubber.tts.semantic import SemanticMetrics, compare_tts_transcript


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


async def produce_provider_tts_rows(
    *,
    paths: WorkspacePaths,
    segment: dict[str, Any],
    text: str,
    provider_bundle: ProviderBundle,
    ffmpeg: FFmpegAdapter,
    next_segment_start_ms: int | None,
    clause_pause_threshold_ms: int = 700,
    **tts_options: Any,
) -> list[dict[str, Any]]:
    clauses = build_tts_clauses(segment, text, min_pause_ms=clause_pause_threshold_ms)
    rows: list[dict[str, Any]] = []
    for index, clause in enumerate(clauses):
        is_last_clause = index == len(clauses) - 1
        clause_next_start_ms = (
            next_segment_start_ms
            if is_last_clause
            else clauses[index + 1].start_ms + int(tts_options.get("start_delay_ms", 0))
        )
        row = await produce_provider_tts_segment(
            paths=paths,
            segment=clause.to_segment(),
            text=clause.translated_text,
            provider_bundle=provider_bundle,
            ffmpeg=ffmpeg,
            next_segment_start_ms=clause_next_start_ms,
            **tts_options,
        )
        row["parent_segment_id"] = clause.parent_segment_id
        row["clause_index"] = index + 1
        row["clause_count"] = len(clauses)
        row["source_clause_text"] = clause.source_text
        rows.append(row)
    return rows


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
    max_edge_silence_ms: int = 1200,
    max_internal_silence_ms: int = 2500,
    clipping_peak_threshold: int = 32760,
    max_clipped_sample_ratio: float = 0.001,
    next_segment_start_ms: int | None = None,
    max_overflow_ms: int = 6_000,
    overflow_reserve_ms: int = 120,
    start_delay_ms: int = 0,
    retained_edge_silence_ms: int = 100,
    semantic_validation_enabled: bool = True,
    semantic_max_cer: float = 0.25,
    semantic_min_token_recall: float = 0.85,
    semantic_retry_attempts: int = 3,
    concurrency: ProviderConcurrency | None = None,
) -> dict[str, Any]:
    segment_id = str(segment["segment_id"])
    raw_audio_path = paths.tts_dir / f"{segment_id}.raw.wav"
    overflow_budget_ms = _overflow_budget_ms(
        segment,
        next_segment_start_ms,
        max_overflow_ms=max_overflow_ms,
        overflow_reserve_ms=overflow_reserve_ms,
        start_delay_ms=start_delay_ms,
    )
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
            max_overflow_ms=overflow_budget_ms,
            start_delay_ms=start_delay_ms,
            semantic_metrics=compare_tts_transcript("", ""),
        )

    tts_voice = voice if voice != "default" else getattr(provider_bundle.tts, "voice", voice)
    source_text = text
    current_text = text
    synthesis_attempts = 0
    rephrase_count = 0
    quality_attempts: list[dict[str, object]] = []
    provider_metadata: dict[str, Any] = {}
    last_report: TTSQualityReport | None = None
    semantic_metrics: SemanticMetrics | None = None
    waveform_failures = 0
    semantic_failures = 0
    orig_ms = int(segment["duration_ms"])

    while True:
        accepted = False
        attempt_budget = max(1, quality_retry_attempts) + max(1, semantic_retry_attempts) - 1
        for _ in range(attempt_budget):
            synthesis_attempts += 1
            attempt_path = paths.tts_dir / f"{segment_id}.attempt_{synthesis_attempts:03d}.wav"
            if concurrency is None:
                tts_result = await provider_bundle.tts.synthesize(
                    current_text, voice=tts_voice, output_path=attempt_path
                )
            else:
                tts_result = await concurrency.run_tts(
                    lambda: provider_bundle.tts.synthesize(
                        current_text, voice=tts_voice, output_path=attempt_path
                    )
                )
            provider_metadata = tts_result.provider_metadata
            tts_duration_ms = tts_result.duration_ms or ffmpeg.duration_ms(tts_result.audio_path)
            last_report = analyze_tts_wav(
                tts_result.audio_path,
                min_rms=min_rms,
                silence_rms_threshold=silence_rms_threshold,
                max_edge_silence_ms=max_edge_silence_ms,
                max_internal_silence_ms=max_internal_silence_ms,
                clipping_peak_threshold=clipping_peak_threshold,
                max_clipped_sample_ratio=max_clipped_sample_ratio,
            )
            tts_duration_ms = last_report.duration_ms or tts_duration_ms
            quality_attempts.append(last_report.to_dict())
            quality_attempts[-1].update({
                "attempt": synthesis_attempts,
                "audio_path": paths.to_relative(tts_result.audio_path),
                "provider_metadata": provider_metadata,
                "duration_ms": tts_duration_ms,
            })
            if any(warning in last_report.warnings for warning in ("tts_audio_leading_silence", "tts_audio_trailing_silence")):
                duration_before_trim_ms = tts_duration_ms
                if trim_edge_silence(
                    tts_result.audio_path,
                    silence_rms_threshold=silence_rms_threshold,
                    retained_edge_silence_ms=retained_edge_silence_ms,
                ):
                    last_report = analyze_tts_wav(
                        tts_result.audio_path,
                        min_rms=min_rms,
                        silence_rms_threshold=silence_rms_threshold,
                        max_edge_silence_ms=max_edge_silence_ms,
                        max_internal_silence_ms=max_internal_silence_ms,
                        clipping_peak_threshold=clipping_peak_threshold,
                        max_clipped_sample_ratio=max_clipped_sample_ratio,
                    )
                    tts_duration_ms = last_report.duration_ms or tts_duration_ms
                    quality_attempts.append(last_report.to_dict())
                    provider_metadata = {
                        **provider_metadata,
                        "edge_silence_trimmed": True,
                        "duration_before_edge_trim_ms": duration_before_trim_ms,
                        "duration_after_edge_trim_ms": tts_duration_ms,
                    }
            if "tts_audio_internal_silence" in last_report.warnings:
                duration_before_compaction_ms = tts_duration_ms
                if compact_excessive_internal_silence(
                    tts_result.audio_path,
                    silence_rms_threshold=silence_rms_threshold,
                    max_internal_silence_ms=max_internal_silence_ms,
                ):
                    last_report = analyze_tts_wav(
                        tts_result.audio_path,
                        min_rms=min_rms,
                        silence_rms_threshold=silence_rms_threshold,
                        max_edge_silence_ms=max_edge_silence_ms,
                        max_internal_silence_ms=max_internal_silence_ms,
                        clipping_peak_threshold=clipping_peak_threshold,
                        max_clipped_sample_ratio=max_clipped_sample_ratio,
                    )
                    tts_duration_ms = last_report.duration_ms or tts_duration_ms
                    quality_attempts.append(last_report.to_dict())
                    provider_metadata = {
                        **provider_metadata,
                        "internal_silence_compacted": True,
                        "duration_before_compaction_ms": duration_before_compaction_ms,
                        "duration_after_compaction_ms": tts_duration_ms,
                    }
            quality_attempts[-1].update({
                "attempt": synthesis_attempts,
                "audio_path": paths.to_relative(tts_result.audio_path),
                "provider_metadata": provider_metadata,
                "duration_ms": tts_duration_ms,
            })
            _write_quality_trace(paths, segment_id, current_text, quality_attempts, status="running")
            if not last_report.ok:
                waveform_failures += 1
                if waveform_failures >= max(1, quality_retry_attempts):
                    break
                continue
            if semantic_validation_enabled and provider_bundle.asr is not None:
                if concurrency is None:
                    asr_result = await provider_bundle.asr.transcribe(tts_result.audio_path, language="vi")
                else:
                    asr_result = await concurrency.run_asr(
                        lambda: provider_bundle.asr.transcribe(tts_result.audio_path, language="vi")
                    )
                semantic_metrics = compare_tts_transcript(current_text, asr_result.text)
                quality_attempts[-1].update({
                    "asr_transcript": asr_result.text,
                    "cer": semantic_metrics.cer,
                    "token_recall": semantic_metrics.token_recall,
                    "semantic_ok": semantic_metrics.ok(
                        max_cer=semantic_max_cer,
                        min_token_recall=semantic_min_token_recall,
                    ),
                })
                _write_quality_trace(paths, segment_id, current_text, quality_attempts, status="running")
                if not quality_attempts[-1]["semantic_ok"]:
                    semantic_failures += 1
                    if semantic_failures >= max(1, semantic_retry_attempts):
                        break
                    continue
            else:
                semantic_metrics = compare_tts_transcript(current_text, current_text)
                quality_attempts[-1].update({
                    "asr_transcript": current_text,
                    "cer": 0.0,
                    "token_recall": 1.0,
                    "semantic_ok": True,
                    "semantic_skipped": True,
                })
                _write_quality_trace(paths, segment_id, current_text, quality_attempts, status="running")
            tts_result.audio_path.replace(raw_audio_path)
            _write_quality_trace(paths, segment_id, current_text, quality_attempts, status="completed")
            accepted = True
            break
        if not accepted:
            if last_report is not None and last_report.ok and semantic_metrics is not None:
                error = (
                    f"{segment_id}: tts_semantic_quality_failed "
                    f"cer={semantic_metrics.cer} token_recall={semantic_metrics.token_recall} "
                    f"attempts={semantic_failures}"
                )
                _write_quality_trace(paths, segment_id, current_text, quality_attempts, status="failed", final_error=error)
                raise ValueError(error)
            error = _tts_quality_error(
                segment_id,
                text=current_text,
                synthesis_attempts=synthesis_attempts,
                rephrase_attempts=rephrase_count,
                report=last_report,
            )
            _write_quality_trace(paths, segment_id, current_text, quality_attempts, status="failed", final_error=str(error))
            raise error

        assert last_report is not None
        available_ms = orig_ms + overflow_budget_ms
        required_speedup_ratio = max(1.0, tts_duration_ms / available_ms)
        if required_speedup_ratio <= max_speedup_ratio:
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
                rephrase_already_attempted=rephrase_count > 0,
                max_overflow_ms=overflow_budget_ms,
                start_delay_ms=start_delay_ms,
                semantic_metrics=semantic_metrics,
            )

        if rephrase_count >= rephrase_attempts:
            error = (
                f"{segment_id}: tts_duration_exceeds_max_speedup "
                f"duration_ms={tts_duration_ms} source_window_ms={orig_ms} "
                f"available_overflow_ms={overflow_budget_ms} target_window_ms={available_ms} "
                f"required_speedup_ratio={required_speedup_ratio:.3f} "
                f"max_speedup_ratio={max_speedup_ratio:.3f} rms={last_report.rms} "
                f"max_internal_silence_ms={last_report.max_internal_silence_ms} "
                f"clipped_sample_ratio={last_report.clipped_sample_ratio} "
                f"source_text_chars={len(source_text)} final_text_chars={len(current_text)}"
            )
            _write_quality_trace(paths, segment_id, current_text, quality_attempts, status="failed", final_error=error)
            raise ValueError(error)
        required_compression_ratio = min(
            1.0,
            (available_ms * max_speedup_ratio) / max(1, tts_duration_ms),
        )
        max_chars = max(1, ceil(len(current_text) * required_compression_ratio))
        while True:
            rephrase_count += 1

            async def rephrase() -> str:
                return await rephrase_tts_text(
                    provider_bundle.llm,
                    text=current_text,
                    target_duration_ms=available_ms,
                    current_duration_ms=tts_duration_ms,
                    segment_id=segment_id,
                    protected_spans=[
                        dict(span)
                        for span in segment.get("protected_spans", [])
                        if isinstance(span, dict)
                    ],
                    max_chars=max_chars,
                    required_compression_ratio=round(required_compression_ratio, 4),
                )

            try:
                rephrased_text = await (rephrase() if concurrency is None else concurrency.run_llm(rephrase))
            except ValueError as exc:
                if "tts_rephrase_empty" not in str(exc):
                    raise
                if rephrase_count >= rephrase_attempts:
                    error = f"{segment_id}: tts_rephrase_empty attempts={rephrase_count}"
                    _write_quality_trace(
                        paths,
                        segment_id,
                        current_text,
                        quality_attempts,
                        status="failed",
                        final_error=error,
                    )
                    raise ValueError(error) from exc
                continue
            protected_errors = protected_translation_errors(
                str(segment.get("source_text", source_text)),
                rephrased_text,
                [span for span in segment.get("protected_spans", []) if isinstance(span, dict)],
            )
            if protected_errors:
                error = (
                    f"{segment_id}: tts_rephrase_protected_span_violation "
                    + "; ".join(protected_errors)
                )
                _write_quality_trace(
                    paths,
                    segment_id,
                    current_text,
                    quality_attempts,
                    status="failed",
                    final_error=error,
                )
                raise ValueError(error)
            if len(rephrased_text) <= max_chars:
                current_text = rephrased_text
                break
            if rephrase_count >= rephrase_attempts:
                error = (
                    f"{segment_id}: tts_rephrase_exceeds_char_limit "
                    f"max_chars={max_chars} actual_chars={len(rephrased_text)} "
                    f"attempts={rephrase_count}"
                )
                _write_quality_trace(
                    paths,
                    segment_id,
                    rephrased_text,
                    quality_attempts,
                    status="failed",
                    final_error=error,
                )
                raise ValueError(error)
        waveform_failures = 0
        semantic_failures = 0
        semantic_metrics = None


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
    rephrase_already_attempted: bool = False,
    max_overflow_ms: int = 0,
    start_delay_ms: int = 0,
    semantic_metrics: SemanticMetrics | None = None,
) -> dict[str, Any]:
    segment_id = str(segment["segment_id"])
    orig_ms = int(segment["duration_ms"])
    aligned_audio_path = paths.tts_dir / f"{segment_id}.wav"
    timing_plan = plan_segment_duration(
        segment_id,
        orig_duration_ms=orig_ms,
        tts_duration_ms=tts_duration_ms,
        speedup_hard_limit=max_speedup_ratio,
        rephrase_already_attempted=rephrase_already_attempted,
        max_overflow_ms=max_overflow_ms,
    )
    apply_time_stretch(
        raw_audio_path,
        aligned_audio_path,
        timing_plan.stretch_ratio if timing_plan.action in {"time_stretch", "time_stretch_overflow"} else 1.0,
    )
    row = {
        "segment_id": segment["segment_id"],
        "target_start_ms": int(segment["start_ms"]) + start_delay_ms,
        "target_end_ms": _target_end_ms(
            segment,
            timing_plan.action,
            tts_duration_ms,
            timing_plan.overflow_ms,
            start_delay_ms=start_delay_ms,
        ),
        "original_start_ms": int(segment["start_ms"]),
        "original_end_ms": int(segment["end_ms"]),
        "commentary_delay_ms": start_delay_ms,
        "orig_duration_ms": orig_ms,
        "tts_duration_ms": tts_duration_ms,
        "alignment_action": timing_plan.action,
        "stretch_ratio": timing_plan.stretch_ratio,
        "overflow_ms": timing_plan.overflow_ms,
        "available_overflow_ms": max_overflow_ms,
        "target_window_ms": orig_ms + max_overflow_ms,
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
        row["final_text"] = final_text
        row["final_text_chars"] = len(final_text)
    if source_text is not None:
        row["source_text_chars"] = len(source_text)
    if semantic_metrics is not None:
        row["semantic_metrics"] = semantic_metrics.to_dict()
    return row


def _overflow_budget_ms(
    segment: dict[str, Any],
    next_segment_start_ms: int | None,
    *,
    max_overflow_ms: int,
    overflow_reserve_ms: int,
    start_delay_ms: int = 0,
) -> int:
    if next_segment_start_ms is None:
        return 0
    target_start_ms = int(segment["start_ms"]) + start_delay_ms
    orig_ms = int(segment["duration_ms"])
    available_ms = max(0, next_segment_start_ms - target_start_ms - max(0, overflow_reserve_ms))
    return min(max(0, max_overflow_ms), max(0, available_ms - orig_ms))


def _target_end_ms(
    segment: dict[str, Any],
    action: str,
    tts_duration_ms: int,
    overflow_ms: int,
    *,
    start_delay_ms: int = 0,
) -> int:
    if action == "overflow":
        return int(segment["start_ms"]) + start_delay_ms + tts_duration_ms
    if action == "time_stretch_overflow":
        return int(segment["start_ms"]) + start_delay_ms + int(segment["duration_ms"]) + overflow_ms
    return int(segment["start_ms"]) + start_delay_ms + int(segment["duration_ms"])


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


def _write_quality_trace(
    paths: WorkspacePaths,
    segment_id: str,
    final_text: str,
    quality_attempts: list[dict[str, object]],
    *,
    status: str,
    final_error: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "cue_id": segment_id,
        "status": status,
        "final_text": final_text,
        "quality_attempts": quality_attempts,
    }
    if final_error is not None:
        payload["final_error"] = final_error
    write_json_atomic(paths.raw_dir / "tts" / f"{segment_id}.quality.json", payload)
