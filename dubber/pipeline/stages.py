from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections.abc import Awaitable
from pathlib import Path

from dubber.asr.timestamps import NormalizedASRTimestamps, TimestampUnit, normalize_asr_timestamps
from dubber.audio.vad import VadConfig as AudioVadConfig, detect_segments
from dubber.core.enums import StageName, StageStatus
from dubber.core.io import read_json, write_json_atomic
from dubber.orchestrator.segment_checkpoint_store import SegmentCheckpointStore
from dubber.orchestrator.stage_artifacts import StageArtifacts
from dubber.pipeline.stage_context import StageContext
from dubber.providers.llm_openai_compatible import LLMStructuredOutputError
from dubber.subtitles.ass import build_subtitle_cues, render_ass
from dubber.tts.clause_builder import TTSWorkItem, build_tts_work_items
from dubber.tts.segment_producer import produce_mock_tts_segment, produce_provider_tts_rows
from dubber.tts.track_mixer import assemble_commentary_track
from dubber.transcript.segmentation import build_transcript_segments
from dubber.translation.block_builder import TranslationContextBlock, build_translation_context_blocks
from dubber.translation.compressor import compress_segment_translation
from dubber.translation.validator import validate_translations

logger = logging.getLogger(__name__)


def _translation_blocks(ctx: StageContext, segments: list[dict[str, object]]) -> list[TranslationContextBlock]:
    config = ctx.config.translation
    return build_translation_context_blocks(
        segments,
        min_context_words=config.min_context_words,
        max_context_words=config.max_context_words,
        context_overlap_words=config.context_overlap_words,
        target_segment_count=config.target_segment_count,
    )


def _glossary_response_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "terms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "term_id": {"type": "string"},
                        "original": {"type": "string"},
                        "vietnamese": {"type": "string"},
                        "category": {"type": "string"},
                        "confidence": {"type": "number"},
                        "source_segments": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "notes": {"type": "string"},
                    },
                    "required": [
                        "term_id",
                        "original",
                        "vietnamese",
                        "category",
                        "confidence",
                        "source_segments",
                        "notes",
                    ],
                },
            }
        },
        "required": ["terms"],
    }


def _normalize_glossary_term(
    term: object,
    *,
    block_index: int,
    term_index: int,
    block_segment_ids: list[str],
    glossary_review: bool,
) -> dict[str, object] | None:
    fallback_id = f"term_{block_index:02d}_{term_index:04d}"
    if isinstance(term, str):
        original = term.strip()
        if not original:
            return None
        return {
            "term_id": fallback_id,
            "original": original,
            "vietnamese": "",
            "category": "term",
            "confidence": 0.5,
            "locked": not glossary_review,
            "source_segments": block_segment_ids,
            "notes": "LLM returned this glossary item as a string instead of an object.",
        }
    if not isinstance(term, dict):
        logger.warning(
            "stage glossary term skipped block=%s index=%s reason=invalid_type type=%s",
            block_index,
            term_index,
            type(term).__name__,
        )
        return None
    original = str(term.get("original", "")).strip()
    if not original:
        logger.warning("stage glossary term skipped block=%s index=%s reason=empty_original", block_index, term_index)
        return None
    source_segments = term.get("source_segments", block_segment_ids)
    if not isinstance(source_segments, list):
        source_segments = block_segment_ids
    return {
        "term_id": term.get("term_id", fallback_id),
        "original": original,
        "vietnamese": term.get("vietnamese", ""),
        "category": term.get("category", "term"),
        "confidence": term.get("confidence", 1.0),
        "locked": not glossary_review,
        "source_segments": source_segments,
        "notes": term.get("notes", ""),
    }


def _fallback_glossary_terms_from_text(
    content: str,
    *,
    block_index: int,
    block_segment_ids: list[str],
    glossary_review: bool,
) -> list[dict[str, object]]:
    terms: list[dict[str, object]] = []
    for line in content.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        is_bullet = candidate.startswith(("*", "-"))
        candidate = candidate.lstrip("-* ").strip()
        if not is_bullet and not candidate.startswith("**"):
            continue
        if candidate.startswith("**") and "**" in candidate[2:]:
            candidate = candidate[2:]
            original, _, notes = candidate.partition("**")
            original = original.strip(" :.-")
            notes = notes.lstrip(" :.-")
        elif ":" in candidate:
            original, _, notes = candidate.partition(":")
            original = original.strip(" *`.-")
            notes = notes.strip()
        else:
            continue
        if not original or len(original) > 120:
            continue
        terms.append(
            {
                "term_id": f"term_{block_index:02d}_{len(terms) + 1:04d}",
                "original": original,
                "vietnamese": "",
                "category": "term",
                "confidence": 0.4,
                "locked": not glossary_review,
                "source_segments": block_segment_ids,
                "notes": notes[:500],
            }
        )
    logger.warning("stage glossary recovered terms from non-json llm output block=%s terms=%s", block_index, len(terms))
    return terms


def _translation_response_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "segment_id": {"type": "string"},
                        "vi_text": {"type": "string"},
                        "used_terms": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "length_ratio": {"type": "number"},
                        "translation_warnings": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "segment_id",
                        "vi_text",
                        "used_terms",
                        "length_ratio",
                        "translation_warnings",
                    ],
                },
            }
        },
        "required": ["segments"],
    }


def _request_structured_json(
    provider: object,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, object],
    *,
    response_name: str,
    response_description: str,
) -> Awaitable[dict[str, object]]:
    if hasattr(provider, "complete_structured_json"):
        return provider.complete_structured_json(
            system_prompt,
            user_prompt,
            schema=schema,
            response_name=response_name,
            response_description=response_description,
        )
    if hasattr(provider, "complete_json"):
        return provider.complete_json(system_prompt, user_prompt, schema=schema)
    raise AttributeError("LLM provider does not support structured JSON completion")


def _effective_domain(ctx: StageContext) -> str:
    return (ctx.config.project.domain or "general").strip() or "general"


def _merge_glossary_terms(existing: list[dict[str, object]], new_terms: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[str, str, str], dict[str, object]] = {}
    for term in existing + new_terms:
        original = str(term.get("original", "")).strip().lower()
        vietnamese = str(term.get("vietnamese", "")).strip().lower()
        category = str(term.get("category", "term")).strip().lower()
        key = (original, vietnamese, category)
        current = merged.get(key)
        source_segments = list({str(segment) for segment in term.get("source_segments", [])})
        if current is None:
            merged[key] = {
                "term_id": str(term.get("term_id", f"term_{len(merged) + 1:04d}")),
                "original": str(term.get("original", "")),
                "vietnamese": str(term.get("vietnamese", "")),
                "category": str(term.get("category", "term")),
                "confidence": float(term.get("confidence", 1.0)),
                "locked": bool(term.get("locked", True)),
                "source_segments": source_segments,
                "notes": str(term.get("notes", "")),
            }
            continue
        current["confidence"] = max(float(current.get("confidence", 1.0)), float(term.get("confidence", 1.0)))
        current["source_segments"] = sorted({*map(str, current.get("source_segments", [])), *source_segments})
        if not current.get("notes") and term.get("notes"):
            current["notes"] = str(term.get("notes", ""))
    return list(merged.values())


def run_job_init(ctx: StageContext, copied_input: Path) -> None:
    logger.info("stage job_init started input=%s", copied_input)
    ctx.store.mark_stage(StageName.JOB_INIT, StageStatus.RUNNING)
    ctx.store.save()
    metadata_path = ctx.paths.input_dir / "input_metadata.v1.json"
    write_json_atomic(metadata_path, ctx.ffmpeg.probe(copied_input))
    StageArtifacts(ctx.paths, ctx.store, ctx.manifest).publish_file(
        stage=StageName.JOB_INIT,
        name="input_metadata",
        path=metadata_path,
    )
    logger.info("stage job_init completed artifact=%s", ctx.paths.to_relative(metadata_path))


def run_audio_extract(ctx: StageContext, copied_input: Path) -> int:
    logger.info("stage audio_extract started")
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
    logger.info("stage audio_extract completed duration_ms=%s artifact=%s", duration_ms, ctx.paths.to_relative(original_wav))
    return duration_ms


def run_vad(ctx: StageContext) -> None:
    logger.info("stage vad started mode=%s frame_ms=%s threshold_ratio=%s min_speech_duration_ms=%s preferred_max_chunk_ms=%s hard_max_chunk_ms=%s silence_merge_threshold_ms=%s context_padding_ms=%s", ctx.config.vad.mode, ctx.config.vad.frame_ms, ctx.config.vad.threshold_ratio, ctx.config.vad.min_speech_duration_ms, ctx.config.vad.preferred_max_chunk_ms, ctx.config.vad.hard_max_chunk_ms, ctx.config.vad.silence_merge_threshold_ms, ctx.config.vad.context_padding_ms)
    ctx.store.mark_stage(StageName.VAD, StageStatus.RUNNING)
    ctx.store.save()
    vad = ctx.config.vad
    segments = detect_segments(
        ctx.paths.audio_dir / "vocals.wav",
        AudioVadConfig(
            mode=vad.mode,
            frame_ms=vad.frame_ms,
            threshold_ratio=vad.threshold_ratio,
            min_duration_ms=vad.min_duration_ms,
            max_duration_ms=vad.max_duration_ms,
            min_speech_duration_ms=vad.min_speech_duration_ms,
            target_min_chunk_ms=vad.target_min_chunk_ms,
            preferred_max_chunk_ms=vad.preferred_max_chunk_ms,
            hard_max_chunk_ms=vad.hard_max_chunk_ms,
            silence_merge_threshold_ms=vad.silence_merge_threshold_ms,
            context_padding_ms=vad.context_padding_ms,
            soft_split_allowed=vad.soft_split_allowed,
            silero_model_path=vad.silero_model_path,
            silero_model_url=vad.silero_model_url,
            silero_auto_download=vad.silero_auto_download,
            silero_threshold=vad.silero_threshold,
            min_silence_duration_ms=vad.min_silence_duration_ms,
            speech_padding_ms=vad.speech_padding_ms,
            max_vad_chunk_ms=vad.max_vad_chunk_ms,
            merge_gap_ms=vad.merge_gap_ms,
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

    logger.info("stage vad completed segments=%s artifact=artifacts/segments.v1.json", len(segments))


def run_asr(ctx: StageContext) -> None:
    segments = ctx.artifact_json("segments.v1.json")["segments"]
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    checkpoint_path = ctx.paths.artifact_path("asr_segments.v1.json")
    segment_ids = [str(segment["segment_id"]) for segment in segments]
    segment_store = SegmentCheckpointStore.load_or_create(
        checkpoint_path, stage=StageName.ASR.value, segment_ids=segment_ids
    )
    segment_store.invalidate_missing_artifacts(ctx.paths.root)
    segment_store.save()
    ctx.store.mark_stage(
        StageName.ASR,
        StageStatus.RUNNING,
        done=segment_store.done_count,
        total=len(segments),
    )
    ctx.store.save()
    source_audio = ctx.paths.audio_dir / "vocals.wav"
    provider_bundle = ctx.require_provider_bundle() if ctx.provider_mode == "openai_compatible" else None

    for index, segment in enumerate(segments, start=1):
        segment_id = str(segment["segment_id"])
        checkpoint = segment_store.segments[segment_id]
        raw_path = ctx.paths.raw_dir / "asr" / f"{segment_id}.json"
        if checkpoint.status == StageStatus.COMPLETED and raw_path.exists():
            continue
        try:
            if ctx.provider_mode == "openai_compatible":
                assert provider_bundle is not None
                clip_path = ctx.paths.audio_dir / "asr" / f"{segment_id}.wav"
                start_ms = int(segment["start_ms"])
                end_ms = int(segment["end_ms"])
                ctx.ffmpeg.extract_audio_segment(
                    source_audio,
                    clip_path,
                    start_ms=start_ms,
                    duration_ms=end_ms - start_ms,
                )
                result = asyncio.run(provider_bundle.asr.transcribe(clip_path, language="en"))
                raw_payload = dict(result.raw)
                if result.confidence is not None:
                    raw_payload.setdefault("confidence", result.confidence)
                write_json_atomic(raw_path, raw_payload)
            else:
                text = f"Mock transcript segment {index} for vertical slice."
                write_json_atomic(raw_path, {"text": text, "confidence": 1.0})
            segment_store.mark(
                segment_id,
                StageStatus.COMPLETED,
                artifact=ctx.paths.to_relative(raw_path),
            )
            segment_store.save()
            logger.info(
                "stage asr segment completed segment=%s index=%s/%s progress=%.1f%% eta_ms=0",
                segment_id,
                index,
                len(segments),
                (segment_store.done_count / len(segments)) * 100 if segments else 100.0,
            )
            ctx.store.mark_stage(
                StageName.ASR,
                StageStatus.RUNNING,
                done=segment_store.done_count,
                total=len(segments),
            )
            ctx.store.save()
        except Exception as exc:
            segment_store.mark(segment_id, StageStatus.FAILED, error=str(exc))
            segment_store.save()
            raise

    asr_chunks: list[dict[str, object]] = []
    for segment in segments:
        segment_id = str(segment["segment_id"])
        raw_path = ctx.paths.raw_dir / "asr" / f"{segment_id}.json"
        raw = read_json(raw_path)
        confidence = float(raw.get("confidence", 1.0))
        if ctx.provider_mode == "openai_compatible":
            normalized = normalize_asr_timestamps(
                raw,
                chunk_start_ms=int(segment["start_ms"]),
                chunk_end_ms=int(segment["end_ms"]),
                require_timestamps=ctx.config.asr_service.require_timestamps,
                allow_chunk_text_fallback=ctx.config.asr_service.allow_chunk_text_fallback,
            )
        else:
            text = str(raw.get("text", ""))
            normalized = NormalizedASRTimestamps(
                source="mock",
                quality="mock",
                units=[TimestampUnit(text=text, start_ms=int(segment["start_ms"]), end_ms=int(segment["end_ms"]))],
                risk_flags=[],
            )
        asr_chunks.append(
            {
                "chunk_id": segment_id,
                "raw_response_path": ctx.paths.to_relative(raw_path),
                "timestamps": normalized,
                "confidence": confidence,
            }
        )

    transcript_segments = build_transcript_segments(asr_chunks, ctx.config.transcript_segmentation)
    publisher.publish_file(
        stage=StageName.ASR,
        name="asr_segments",
        path=segment_store.path,
        status=StageStatus.RUNNING,
    )
    publisher.publish_json(
        stage=StageName.ASR,
        name="transcript",
        filename="transcript.v1.json",
        payload={
            "schema_version": "1.0",
            "provider": {
                "type": ctx.provider_mode,
                "model": "provider-asr" if ctx.provider_mode == "openai_compatible" else "mock-asr",
            },
            "segments": transcript_segments,
        },
        done=len(segments),
        total=len(segments),
    )


def run_glossary(ctx: StageContext, *, glossary_review: bool) -> bool:
    domain = _effective_domain(ctx)
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    transcript = ctx.artifact_json("transcript.v1.json")
    blocks = _translation_blocks(ctx, transcript["segments"])
    block_ids = [f"block_{index:06d}" for index in range(1, max(1, len(blocks)) + 1)]
    checkpoint = SegmentCheckpointStore.load_or_create(
        ctx.paths.artifact_path("glossary_blocks.v1.json"),
        stage=StageName.GLOSSARY.value,
        segment_ids=block_ids,
    )
    checkpoint.invalidate_missing_artifacts(ctx.paths.root)
    checkpoint.save()
    ctx.store.mark_stage(
        StageName.GLOSSARY,
        StageStatus.RUNNING,
        done=checkpoint.done_count,
        total=checkpoint.total_count,
    )
    ctx.store.save()
    source_segments = [str(segment["segment_id"]) for segment in transcript["segments"]]
    terms: list[dict[str, object]] = []

    if ctx.provider_mode == "openai_compatible":
        for block_index, block in enumerate(blocks, start=1):
            block_id = block_ids[block_index - 1]
            raw_path = ctx.paths.raw_dir / "glossary" / f"{block_id}.json"
            block_segments = block.all_segments
            block_segment_ids = [str(segment["segment_id"]) for segment in block_segments]
            if checkpoint.segments[block_id].status == StageStatus.COMPLETED and raw_path.exists():
                block_terms = list(read_json(raw_path).get("terms", []))
            else:
                try:
                    result = asyncio.run(
                        _request_structured_json(
                            ctx.require_provider_bundle().llm,
                            f"Extract glossary terms for a {domain} video transcript block. Return raw JSON only.",
                            json.dumps(
                                {
                                    "instruction": "Return only a JSON object with a terms array.",
                                    "domain": domain,
                                    "segments": block_segments,
                                },
                                ensure_ascii=False,
                            ),
                            _glossary_response_schema(),
                            response_name="glossary_output",
                            response_description="Glossary terms extracted from the transcript.",
                        )
                    )
                    raw_terms = result.get("terms", [])
                except LLMStructuredOutputError as exc:
                    raw_terms = _fallback_glossary_terms_from_text(
                        exc.content,
                        block_index=block_index,
                        block_segment_ids=block_segment_ids or source_segments,
                        glossary_review=glossary_review,
                    )
                if not isinstance(raw_terms, list):
                    raw_terms = []
                block_terms = []
                for term_index, term in enumerate(raw_terms, start=1):
                    normalized = _normalize_glossary_term(
                        term,
                        block_index=block_index,
                        term_index=term_index,
                        block_segment_ids=block_segment_ids or source_segments,
                        glossary_review=glossary_review,
                    )
                    if normalized is not None:
                        block_terms.append(normalized)
                write_json_atomic(raw_path, {"schema_version": "1.0", "terms": block_terms})
                checkpoint.mark(block_id, StageStatus.COMPLETED, artifact=ctx.paths.to_relative(raw_path))
                checkpoint.save()
            terms = _merge_glossary_terms(terms, block_terms)
    else:
        block_id = block_ids[0]
        raw_path = ctx.paths.raw_dir / "glossary" / f"{block_id}.json"
        terms = [{
            "term_id": "term_0001",
            "original": "vertical slice",
            "vietnamese": "lát cắt dọc",
            "category": "phrase",
            "locked": not glossary_review,
            "source_segments": source_segments,
            "notes": "Mock glossary entry.",
        }]
        if checkpoint.segments[block_id].status != StageStatus.COMPLETED or not raw_path.exists():
            write_json_atomic(raw_path, {"schema_version": "1.0", "terms": terms})
            checkpoint.mark(block_id, StageStatus.COMPLETED, artifact=ctx.paths.to_relative(raw_path))
            checkpoint.save()

    publisher.publish_file(
        stage=StageName.GLOSSARY,
        name="glossary_blocks",
        path=checkpoint.path,
        status=StageStatus.RUNNING,
        done=checkpoint.done_count,
        total=checkpoint.total_count,
    )
    filename = "glossary.draft.json" if glossary_review else "glossary.locked.json"
    glossary_path = ctx.paths.artifact_path(filename)
    write_json_atomic(
        glossary_path,
        {
            "schema_version": "1.0",
            "domain": domain,
            "status": "draft" if glossary_review else "locked",
            "terms": terms,
        },
    )
    if glossary_review:
        publisher.publish_file(
            stage=StageName.GLOSSARY,
            name="glossary_draft",
            path=glossary_path,
            status=StageStatus.WAITING_REVIEW,
        )
        return True
    publisher.publish_file(stage=StageName.GLOSSARY, name="glossary", path=glossary_path)
    return False


def run_translation(ctx: StageContext) -> None:
    domain = _effective_domain(ctx)
    transcript = ctx.artifact_json("transcript.v1.json")
    glossary = ctx.artifact_json("glossary.locked.json")
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    blocks = _translation_blocks(ctx, transcript["segments"])
    block_ids = [f"block_{index:06d}" for index in range(1, len(blocks) + 1)]
    checkpoint = SegmentCheckpointStore.load_or_create(
        ctx.paths.artifact_path("translation_blocks.v1.json"),
        stage=StageName.TRANSLATION.value,
        segment_ids=block_ids,
    )
    checkpoint.invalidate_missing_artifacts(ctx.paths.root)
    checkpoint.save()
    ctx.store.mark_stage(
        StageName.TRANSLATION,
        StageStatus.RUNNING,
        done=checkpoint.done_count,
        total=checkpoint.total_count,
    )
    ctx.store.save()
    provider_translations: dict[str, dict] = {}

    for block_index, block in enumerate(blocks, start=1):
        block_id = block_ids[block_index - 1]
        raw_path = ctx.paths.raw_dir / "translation" / f"{block_id}.json"
        target_ids = [str(segment["segment_id"]) for segment in block.target_segments]
        if checkpoint.segments[block_id].status == StageStatus.COMPLETED and raw_path.exists():
            block_result = read_json(raw_path)
        elif ctx.provider_mode == "openai_compatible":
            collected: dict[str, dict[str, object]] = {}
            targets_by_id = {str(item["segment_id"]): item for item in block.target_segments}
            max_attempts = max(2, ctx.config.runtime.retry_max_attempts)
            for attempt in range(1, max_attempts + 1):
                missing = [segment_id for segment_id in target_ids if segment_id not in collected]
                if not missing:
                    break
                retry_instruction = (
                    " This is a retry. Return only the still-missing segment IDs: " + ", ".join(missing)
                    if attempt > 1 else ""
                )
                result = asyncio.run(
                    _request_structured_json(
                        ctx.require_provider_bundle().llm,
                        f"Translate the target_segments of a {domain} video transcript block to Vietnamese. Use context only for continuity. Return raw JSON only.{retry_instruction}",
                        json.dumps(
                            {
                                "instruction": "Return only a JSON object with a segments array. Return exactly one object for every required target ID.",
                                "required_target_segment_ids": missing,
                                "domain": domain,
                                "target_segments": [targets_by_id[segment_id] for segment_id in missing],
                                "context_before": block.context_before,
                                "context_after": block.context_after,
                                "glossary": glossary["terms"],
                            },
                            ensure_ascii=False,
                        ),
                        _translation_response_schema(),
                        response_name="translation_output",
                        response_description="Vietnamese translation segments.",
                    )
                )
                for item in result.get("segments", []):
                    segment_id = str(item.get("segment_id", ""))
                    if segment_id in missing:
                        collected[segment_id] = item
            missing = [segment_id for segment_id in target_ids if segment_id not in collected]
            if missing:
                raise ValueError(
                    f"Translation response missing segment_ids after {max_attempts} attempts: {', '.join(missing[:5])}"
                )
            block_result = {
                "schema_version": "1.0",
                "segments": [collected[segment_id] for segment_id in target_ids],
            }
            write_json_atomic(raw_path, block_result)
            checkpoint.mark(block_id, StageStatus.COMPLETED, artifact=ctx.paths.to_relative(raw_path))
            checkpoint.save()
        else:
            block_result = {
                "schema_version": "1.0",
                "segments": [
                    {
                        "segment_id": source["segment_id"],
                        "vi_text": f"Đây là lát cắt dọc thuyết minh tiếng Việt mẫu cho {source['segment_id']}.",
                        "used_terms": ["vertical slice"],
                        "length_ratio": 1.0,
                        "translation_warnings": [],
                    }
                    for source in block.target_segments
                ],
            }
            write_json_atomic(raw_path, block_result)
            checkpoint.mark(block_id, StageStatus.COMPLETED, artifact=ctx.paths.to_relative(raw_path))
            checkpoint.save()
        for item in block_result.get("segments", []):
            provider_translations[str(item["segment_id"])] = item

    translated_segments: list[dict[str, object]] = []
    for source in transcript["segments"]:
        provider_segment = provider_translations.get(str(source["segment_id"]), {})
        candidate = {
            "segment_id": source["segment_id"],
            "source_text": source["source_text"],
            "vi_text": provider_segment.get("vi_text", ""),
            "used_terms": provider_segment.get("used_terms", []),
            "length_ratio": provider_segment.get("length_ratio", 1.0),
            "translation_warnings": provider_segment.get("translation_warnings", []),
        }
        compressed = compress_segment_translation(candidate, glossary["terms"], max_length_ratio=3.0)
        candidate["vi_text"] = compressed.vi_text
        candidate["translation_warnings"] = list(candidate["translation_warnings"]) + compressed.warnings
        translated_segments.append(candidate)

    validation = validate_translations(transcript["segments"], translated_segments, glossary["terms"], max_length_ratio=3.0)
    publisher.publish_file(
        stage=StageName.TRANSLATION,
        name="translation_blocks",
        path=checkpoint.path,
        status=StageStatus.RUNNING,
        done=checkpoint.done_count,
        total=checkpoint.total_count,
    )
    publisher.publish_json(
        stage=StageName.TRANSLATION,
        name="translated",
        filename="translated.v1.json",
        payload={
            "schema_version": "1.0",
            "segments": translated_segments,
            "validation_warnings": validation.warnings,
        },
        done=len(transcript["segments"]),
        total=len(transcript["segments"]),
    )


def run_tts(
    ctx: StageContext,
    duration_ms: int,
    *,
    crash_stage: str | None = None,
    crash_after_segments: int | None = None,
) -> Path:
    segments = ctx.artifact_json("transcript.v1.json")["segments"]
    segment_ids = [str(segment["segment_id"]) for segment in segments]
    checkpoint = SegmentCheckpointStore.load_or_create(
        ctx.paths.artifact_path("tts_segments.v1.json"),
        stage=StageName.TTS.value,
        segment_ids=segment_ids,
    )
    checkpoint.invalidate_missing_artifacts(ctx.paths.root)
    checkpoint.save()
    ctx.store.mark_stage(
        StageName.TTS,
        StageStatus.RUNNING,
        done=checkpoint.done_count,
        total=len(segments),
    )
    ctx.store.save()
    if crash_stage == StageName.TTS.value and crash_after_segments == 0:
        raise RuntimeError("simulated crash at tts after 0 segments")

    translations = {
        str(segment["segment_id"]): segment
        for segment in ctx.artifact_json("translated.v1.json")["segments"]
    }
    if ctx.provider_mode == "openai_compatible":
        work_items = build_tts_work_items(segments, translations)
    else:
        work_items = [
            TTSWorkItem(segment=segment, translated_text="", parent_segment_ids=(str(segment["segment_id"]),))
            for segment in segments
        ]
    manifest_path = ctx.paths.artifact_path("tts_manifest.v1.json")
    existing_rows = list(read_json(manifest_path).get("segments", [])) if manifest_path.exists() else []
    rows_by_parents: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in existing_rows:
        parents = tuple(str(item) for item in row.get("parent_segment_ids", [row.get("segment_id")]))
        rows_by_parents.setdefault(parents, []).append(row)

    tts_segments: list[dict[str, object]] = []
    completed_work_items = 0
    for index, work_item in enumerate(work_items, start=1):
        parents = tuple(work_item.parent_segment_ids)
        prior_rows = rows_by_parents.get(parents, [])
        reusable = all(
            checkpoint.segments[parent].status == StageStatus.COMPLETED
            for parent in parents
        ) and bool(prior_rows) and all(
            ctx.paths.resolve_relative(str(row["audio_path"])).exists()
            for row in prior_rows
        )
        if reusable:
            rows = prior_rows
        elif ctx.provider_mode == "openai_compatible":
            tts_config = ctx.config.tts_service
            rows = asyncio.run(
                produce_provider_tts_rows(
                    paths=ctx.paths,
                    segment=work_item.segment,
                    text=(
                        work_item.translated_text
                        if str(work_item.segment.get("source_text", "")).strip()
                        else ""
                    ),
                    provider_bundle=ctx.require_provider_bundle(),
                    ffmpeg=ctx.ffmpeg,
                    quality_retry_attempts=tts_config.quality_retry_attempts,
                    rephrase_attempts=tts_config.rephrase_attempts,
                    max_speedup_ratio=tts_config.max_speedup_ratio,
                    min_rms=tts_config.min_rms,
                    silence_rms_threshold=tts_config.silence_rms_threshold,
                    max_edge_silence_ms=tts_config.max_edge_silence_ms,
                    max_internal_silence_ms=tts_config.max_internal_silence_ms,
                    clipping_peak_threshold=tts_config.clipping_peak_threshold,
                    max_clipped_sample_ratio=tts_config.max_clipped_sample_ratio,
                    clause_pause_threshold_ms=tts_config.clause_pause_threshold_ms,
                    next_segment_start_ms=(
                        int(work_items[index].segment["start_ms"]) + 500
                        if index < len(work_items) else duration_ms
                    ),
                )
            )
        else:
            rows = [produce_mock_tts_segment(paths=ctx.paths, segment=work_item.segment)]
        for row in rows:
            row["parent_segment_ids"] = list(parents)
        tts_segments.extend(rows)
        if not reusable:
            for parent in parents:
                checkpoint.mark(parent, StageStatus.COMPLETED, artifact=str(rows[0]["audio_path"]))
            checkpoint.save()
            write_json_atomic(manifest_path, {"schema_version": "1.0", "segments": tts_segments + [
                row for key, values in rows_by_parents.items() if key not in {tuple(item.parent_segment_ids) for item in work_items[:index]} for row in values
            ]})
        completed_work_items += 1
        ctx.store.mark_stage(
            StageName.TTS,
            StageStatus.RUNNING,
            done=checkpoint.done_count,
            total=len(segments),
        )
        ctx.store.save()
        if crash_stage == StageName.TTS.value and crash_after_segments == completed_work_items:
            raise RuntimeError(f"simulated crash at tts after {completed_work_items} segments")

    mix_audio_path = ctx.paths.tts_dir / "mix.wav"
    assemble_commentary_track(
        paths=ctx.paths,
        ffmpeg=ctx.ffmpeg,
        tts_segments=tts_segments,
        output_audio=mix_audio_path,
        max_speedup_ratio=ctx.config.tts_service.max_speedup_ratio,
    )
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    publisher.publish_file(
        stage=StageName.TTS,
        name="tts_segments",
        path=checkpoint.path,
        status=StageStatus.RUNNING,
    )
    publisher.publish_json(
        stage=StageName.TTS,
        name="tts_manifest",
        filename="tts_manifest.v1.json",
        payload={"schema_version": "1.0", "segments": tts_segments},
        done=len(segments),
        total=len(segments),
    )
    return mix_audio_path


def run_mixing(ctx: StageContext, copied_input: Path, tts_audio: Path, duration_ms: int) -> Path:
    logger.info("stage mixing started tts_audio=%s", tts_audio)
    ctx.store.mark_stage(StageName.MIXING, StageStatus.RUNNING)
    ctx.store.save()
    final_audio = ctx.paths.audio_dir / "final_mix.wav"
    ctx.ffmpeg.mix_commentary_audio(
        ctx.paths.audio_dir / "original.wav",
        tts_audio,
        final_audio,
        original_ducking_db=ctx.config.mixing.original_ducking_db,
        tts_boost_db=ctx.config.mixing.tts_boost_db,
        final_loudness_normalization=ctx.config.mixing.final_loudness_normalization,
    )
    output_video = ctx.paths.output_dir / f"{copied_input.stem}_vi.mp4"
    muxed_video = output_video
    if ctx.config.subtitles.enabled:
        muxed_video = ctx.paths.output_dir / f"{copied_input.stem}_vi.mux.mp4"
    ctx.ffmpeg.mux_video_audio(copied_input, final_audio, muxed_video)
    subtitle_ass_path: Path | None = None
    if ctx.config.subtitles.enabled:
        if ctx.config.subtitles.mode != "burn_in":
            raise ValueError(f"Unsupported subtitles.mode: {ctx.config.subtitles.mode}")
        subtitle_ass_path = ctx.paths.artifact_path("subtitles.ass")
        video_width, video_height = _video_dimensions(ctx.ffmpeg.probe(muxed_video))
        subtitle_ass_path.write_text(
            render_ass(
                build_subtitle_cues(
                    ctx.artifact_json("transcript.v1.json"),
                    ctx.artifact_json("translated.v1.json"),
                    ctx.config.subtitles,
                ),
                ctx.config.subtitles,
                video_width=video_width,
                video_height=video_height,
            ),
            encoding="utf-8",
        )
        ctx.ffmpeg.burn_in_subtitles(muxed_video, subtitle_ass_path, output_video)
    qa_path = ctx.paths.output_dir / f"{copied_input.stem}_vi.qa.json"
    write_json_atomic(
        qa_path,
        {
            "schema_version": "1.0",
            "job_id": ctx.paths.root.name,
            "input_duration_ms": duration_ms,
            "output_duration_ms": ctx.ffmpeg.duration_ms(output_video),
            "segments_total": len(ctx.artifact_json("segments.v1.json")["segments"]),
            "low_confidence_segments": 0,
            "glossary_terms": 1,
            "tts_overflow_segments": 0,
            "max_overflow_ms": 0,
            "sync_drift_p95_ms": 0,
            "warnings": [],
        },
    )
    publisher = StageArtifacts(ctx.paths, ctx.store, ctx.manifest)
    publisher.publish_file(
        stage=StageName.MIXING,
        name="final_audio",
        path=final_audio,
        status=StageStatus.RUNNING,
    )
    publisher.publish_file(
        stage=StageName.MIXING,
        name="qa_report",
        path=qa_path,
        status=StageStatus.RUNNING,
    )
    if subtitle_ass_path is not None and ctx.config.subtitles.output_sidecar:
        publisher.publish_file(
            stage=StageName.MIXING,
            name="subtitle_ass",
            path=subtitle_ass_path,
            status=StageStatus.RUNNING,
        )
    publisher.publish_file(
        stage=StageName.MIXING,
        name="output_video",
        path=output_video,
        status=StageStatus.COMPLETED,
    )
    logger.info("stage mixing completed output=%s qa=%s", output_video, qa_path)
    return output_video


def _video_dimensions(metadata: dict[str, object]) -> tuple[int, int]:
    streams = metadata.get("streams", [])
    if not isinstance(streams, list):
        return 1920, 1080
    for stream in streams:
        if not isinstance(stream, dict) or stream.get("codec_type") != "video":
            continue
        try:
            width = int(stream.get("width", 1920))
            height = int(stream.get("height", 1080))
        except (TypeError, ValueError):
            return 1920, 1080
        if width > 0 and height > 0:
            return width, height
    return 1920, 1080
